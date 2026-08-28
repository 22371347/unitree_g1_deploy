# Trajectory7 Reference Stance

Root:

```text
position xyz = [0.0, 0.0, 0.7098747491836548]
quaternion wxyz = [1.0, 0.0, 0.0, 0.0]
```

The exact 29D vectors are provided as:

- `TRAJECTORY7_STANCE_POLICY.csv`
- `TRAJECTORY7_STANCE_SDK.csv`

The reference motion satisfies:

```text
motion frame 0   == trajectory7 stance
motion frame 158 == trajectory7 stance
```

with max joint-position error = `0`.

## File conventions

`trajectory7_original_default_stance.csv` is the original uploaded stance CSV.

Its layout is:

```text
x, y, z, qx, qy, qz, qw, 29 grouped/SDK-order joints
```

`trajectory7_stance_policy_order.npz` stores the deployment stance in policy
joint order and root quaternion in `wxyz` order.

The generated punch CSV is headerless and uses:

```text
x, y, z, qx, qy, qz, qw, 29 grouped/SDK-order joints
```

The final `motion.npz` stores joint vectors in policy order.
