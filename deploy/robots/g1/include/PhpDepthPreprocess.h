#pragma once

// PHP 深度预处理：与 holosoma `depth_shm_plugin` 逐项一致
//   原始平面度量深度 (106x60, m) ->
//     nan/inf->far, crop[2:,4:-4] (58x98), clip[0.3,3.0],
//     双三次抗锯齿重采样 (58x98 -> 58x87), min(far),
//     where(<0.15 -> far), 归一化 (d-near)/(far-near)-0.5
//
// 双三次核为 Catmull-Rom (a=-0.75)，antialias 采用 ATen 的做法：
//   下采样时 support = 2*scale，权重取 cubic((j-center)/scale) 后归一化。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace php_depth
{
constexpr int   RAW_W = 106, RAW_H = 60;
constexpr int   CROP_TOP = 2, CROP_LEFT = 4, CROP_RIGHT = 4;
constexpr int   CROP_W = RAW_W - CROP_LEFT - CROP_RIGHT;  // 98
constexpr int   CROP_H = RAW_H - CROP_TOP;                // 58
constexpr int   OUT_W = 87, OUT_H = 58;
constexpr float NEAR = 0.3f, FAR = 3.0f;
constexpr float EMPTY_THRESH = 0.15f;

inline double cubic_kernel(double x)
{
    double ax = std::fabs(x);
    if (ax <= 1.0) {
        return ((1.25 * ax - 2.25) * ax) * ax + 1.0;
    } else if (ax < 2.0) {
        return (((-0.75 * ax + 3.75) * ax - 6.0) * ax + 3.0);
    }
    return 0.0;
}

struct Resize1D
{
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<double>> weights;
};

inline Resize1D build_weights_aa(int in_size, int out_size)
{
    Resize1D r;
    r.indices.resize(out_size);
    r.weights.resize(out_size);
    double scale = static_cast<double>(in_size) / out_size;
    double aa = (scale > 1.0) ? scale : 1.0;
    double support = 2.0 * aa;
    for (int o = 0; o < out_size; ++o) {
        double center = scale * (o + 0.5) - 0.5;
        int left = static_cast<int>(std::floor(center - support + 0.5));
        int right = static_cast<int>(std::ceil(center + support - 0.5));
        left = std::max(left, 0);
        right = std::min(right, in_size - 1);
        double sum = 0.0;
        std::vector<int> idx;
        std::vector<double> w;
        for (int j = left; j <= right; ++j) {
            double x = (j - center) / aa;
            double weight = cubic_kernel(x);
            idx.push_back(j);
            w.push_back(weight);
            sum += weight;
        }
        if (sum != 0.0) {
            for (auto& v : w) v /= sum;
        }
        r.indices[o] = std::move(idx);
        r.weights[o] = std::move(w);
    }
    return r;
}

inline void resize_bicubic_aa(const std::vector<float>& in, int in_h, int in_w,
                              std::vector<float>& out, int out_h, int out_w)
{
    Resize1D wx = build_weights_aa(in_w, out_w);
    Resize1D wy = build_weights_aa(in_h, out_h);

    std::vector<float> tmp(static_cast<size_t>(in_h) * out_w);
    for (int y = 0; y < in_h; ++y) {
        const float* row = &in[static_cast<size_t>(y) * in_w];
        for (int ox = 0; ox < out_w; ++ox) {
            double acc = 0.0;
            const auto& idx = wx.indices[ox];
            const auto& w = wx.weights[ox];
            for (size_t k = 0; k < idx.size(); ++k) acc += w[k] * row[idx[k]];
            tmp[static_cast<size_t>(y) * out_w + ox] = static_cast<float>(acc);
        }
    }
    out.assign(static_cast<size_t>(out_h) * out_w, 0.0f);
    for (int oy = 0; oy < out_h; ++oy) {
        const auto& idx = wy.indices[oy];
        const auto& w = wy.weights[oy];
        for (int ox = 0; ox < out_w; ++ox) {
            double acc = 0.0;
            for (size_t k = 0; k < idx.size(); ++k) {
                acc += w[k] * tmp[static_cast<size_t>(idx[k]) * out_w + ox];
            }
            out[static_cast<size_t>(oy) * out_w + ox] = static_cast<float>(acc);
        }
    }
}

inline void preprocess_depth(const std::vector<float>& raw, std::vector<float>& out_norm)
{
    std::vector<float> crop(static_cast<size_t>(CROP_H) * CROP_W);
    for (int y = 0; y < CROP_H; ++y) {
        for (int x = 0; x < CROP_W; ++x) {
            float v = raw[static_cast<size_t>(y + CROP_TOP) * RAW_W + (x + CROP_LEFT)];
            if (std::isnan(v) || std::isinf(v)) v = FAR;
            v = std::clamp(v, NEAR, FAR);
            crop[static_cast<size_t>(y) * CROP_W + x] = v;
        }
    }
    std::vector<float> rs;
    resize_bicubic_aa(crop, CROP_H, CROP_W, rs, OUT_H, OUT_W);
    out_norm.resize(static_cast<size_t>(OUT_H) * OUT_W);
    for (size_t i = 0; i < rs.size(); ++i) {
        float d = std::min(rs[i], FAR);
        if (d < EMPTY_THRESH) d = FAR;
        out_norm[i] = (d - NEAR) / (FAR - NEAR) - 0.5f;
    }
}

}  // namespace php_depth
