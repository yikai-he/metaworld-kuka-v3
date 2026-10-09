# KUKA-v3 Robotiq push reward audit

## Method

`scripts/diagnose_kuka_robotiq_push.py` ran 50 deterministic configurations
(seeds 0–49), 215 control steps each. Every timestep records TCP distance,
minimum MuJoCo geom distance from any inner pad to the puck, physical contact
pairs, displacement, goal distance and progress, aperture, gripper action,
caging, in-place shaping, legacy gate state, reward, and success. Positive
progress is a per-step goal-distance reduction greater than `1e-5 m`.

All 50 rollouts established pad contact and positive goal progress; 22 reached
the unchanged 5 cm success condition.

## Measured compatibility

| Population | Timesteps | Min | Median | Max | Fraction below 2 cm |
|---|---:|---:|---:|---:|---:|
| Physical pad contact | 4,316 | 4.029 cm | 4.927 cm | 5.398 cm | 0% |
| Positive goal progress | 5,610 | 4.029 cm | 4.974 cm | 6.296 cm | 0% |

All 5,610 positive-progress timesteps occurred while the legacy gate was
false. First-contact TCP distance across rollouts was 4.626–5.365 cm, median
4.992 cm. Final rollout progress was 11.24–24.78 cm, median 15.61 cm.

The 2 cm threshold is therefore fundamentally mismatched to the Robotiq TCP,
not merely overly strict. The TCP is the midpoint of the articulated fingertip
sites; the active pushing face is displaced from that midpoint.

## Offline gate comparison

| Distance gate | Contact coverage | Progress coverage | Pre-contact false activations |
|---:|---:|---:|---:|
| 2.0–4.0 cm | 0% | 0% | 0 |
| 5.0 cm | 68.1% | 55.6% | 72 |
| 5.5 cm | 100% | 98.0% | 315 |
| 6.0 cm | 100% | 99.8% | 383 |

A 5.5 cm threshold is the smallest tested distance gate covering all measured
contact, but it rewards pre-contact hovering and remains sensitive to linkage,
aperture, and contact-side changes.

## Ranked alternatives

1. **Physical contact gate (selected).** Exact match to useful Robotiq push
   contact, no distance calibration constant, and no no-contact hover bonus.
   This changes only the legacy gate; caging, in-place shaping, scale, and
   success remain unchanged.
2. **Progress reward independent of a gate.** Directly rewards the objective
   and covers inertial progress after contact ends, but introduces temporal
   state/noise sensitivity and is a larger reward redesign. Suitable for a
   later ablation if contact-gated SAC learning is inadequate.
3. **Soft distance weighting.** Removes a discontinuity but still uses the
   geometrically mismatched TCP proxy and can reward hovering.
4. **Relaxed 5.5 cm gate.** Small textual change, but has 315 measured
   pre-contact false activations and incomplete progress coverage.
5. **Existing 2 cm gate.** Incompatible: zero contact/progress coverage.

## Implemented minimal correction

`KukaPushEnvV3.compute_reward()` now gates the existing shaping bonus on any
physical Robotiq inner-pad/puck contact. `KukaXYZEnv.pad_contacting_object()`
provides the single-pad predicate; bilateral `touching_object()` remains intact
for grasp semantics. The success override remains exactly 10 at the unchanged
5 cm object-to-goal threshold.

After the change, audited rewards remained finite and bounded from
`0.002855` to `10.0` over the 50 rollouts.

Regression tests cover finite/bounded rewards, unchanged success maximum,
monotonic in-place progress, moving-away ordering, real contact activation,
no-contact hover rejection, and deterministic repeated evaluation.
