# File Sources

## From the newly uploaded Sim2Real input archive

- `01_POLICY/model_30000.pt`
- `01_POLICY/policy.onnx`
- final generated punch CSV
- final `motion.npz`
- `agent.yaml`, `env.yaml`, pickle configs
- current collected training source snapshot
- export/train scripts
- model/server metadata

## From files supplied earlier in the same conversation

- `trajectory7_original_default_stance.csv`
  - source: uploaded `trajectory (7).csv`
- `trajectory7_stance_policy_order.npz`
  - source: generated from that trajectory7 stance and previously returned to the user
- `g1_29dof_unitree_official_REFERENCE_ONLY.xml`
  - source: previous `WBT_G1_full_handoff_2026-08-18_under50MB.zip`
  - included only as historical naming/reference; meshes are not included

## Generated during final handoff packaging

- README / model card / deployment checklist / validation report
- interface tables and mapping files
- deployment reference math script
- SHA256 manifest
