// Minimal POSIX shared-memory writer (header-only).
//
// Mirrors the contract used by holosoma's DepthShmPlugin / DepthShmSensor: a
// plain float32 array living at /dev/shm/<name>.  We only write the raw frame;
// the consumer attaches with the shape/dtype it expects.
#pragma once

#include <cstddef>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace depth_shm {

class Writer {
 public:
  Writer() = default;
  ~Writer() { Close(); }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  // Create (or truncate) the block. Returns false on failure.
  bool Create(const std::string& name, size_t bytes) {
    Close();
    name_ = name;
    bytes_ = bytes;
    // Unlink any stale block first so a previous run with a different size
    // does not leave us with a mismatched mapping.
    shm_unlink(name_.c_str());
    fd_ = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
      std::perror("[depth_shm] shm_open");
      return false;
    }
    if (::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
      std::perror("[depth_shm] ftruncate");
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    void* p = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      std::perror("[depth_shm] mmap");
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    ptr_ = static_cast<float*>(p);
    std::printf("[depth_shm] created '%s' (%zu bytes)\n", name_.c_str(), bytes_);
    return true;
  }

  float* data() { return ptr_; }
  bool ok() const { return ptr_ != nullptr; }
  size_t bytes() const { return bytes_; }

  void Close() {
    if (ptr_) {
      ::munmap(ptr_, bytes_);
      ptr_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  void Unlink() {
    if (!name_.empty()) {
      shm_unlink(name_.c_str());
    }
  }

 private:
  std::string name_;
  int fd_ = -1;
  size_t bytes_ = 0;
  float* ptr_ = nullptr;
};

}  // namespace depth_shm
