# Control Timing

Training/resolved environment:

```text
simulation dt = 0.005 s
decimation    = 4
policy dt     = 0.020 s
policy rate   = 50 Hz
motion fps    = 50 Hz
frames        = 159
duration      = 3.18 s
```

## Deployment rule

Run policy inference every 20 ms.

At each policy tick:

1. obtain current reference for `time_step`;
2. build the raw 154D observation;
3. infer 29D raw action;
4. convert action to `q_des_policy`;
5. reorder to SDK/hardware order;
6. hold/track that command until the next policy tick;
7. store raw action as `last_action`;
8. increment `time_step`.

Reference step range is `0..158`.

For initial Sim2Real bring-up, when the punch finishes, hold frame `158`.
Do not automatically wrap to frame `0` unless looping is explicitly desired and
the transition has been validated.
