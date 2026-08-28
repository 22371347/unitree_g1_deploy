# Robot Resource Note

The newly collected `WBT_G1_trajectory7_model30000_sim2real_inputs` archive had
an empty `05_ROBOT_RESOURCE` directory.

This final handoff therefore supplements:

```text
g1_29dof_unitree_official_REFERENCE_ONLY.xml
```

from the previously supplied project handoff.

Important limitations:

- it is included for joint/body naming and historical Sim2Sim reference;
- it references external STL mesh files that are not included here;
- it is not a self-contained simulation scene;
- it is not authoritative over the Sim2Real team's current Unitree SDK/model.

For hardware deployment, use the team's official current G1 29DoF robot
description and verify every joint name/index/sign against
`03_INTERFACE_SPEC/JOINT_INTERFACE.csv`.
