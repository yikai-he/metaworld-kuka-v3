# KUKA-v3 / Robotiq 2F-85 geometry calibration

This report records the post-migration geometry pass. Measurements come from
MuJoCo runtime state using `scripts/diagnose_kuka_geometry.py`; they are not
nominal XML-only estimates. Reward functions, success thresholds, objects,
SAC settings, and the shared robot transform were not changed.

## Runtime Robotiq reference

| Policy command | Actuator control | Aperture |
|---:|---:|---:|
| -1.0 | 0.0 | 84.19 mm |
| -0.5 | 63.75 | 64.67 mm |
| 0.0 | 127.5 | 43.01 mm |
| +0.5 | 191.25 | 20.01 mm |
| +1.0 | 255.0 | 0.27 mm |

The collision-pad envelope extends approximately 18.9 mm below the TCP over
the tested aperture range. `hand` is approximately 13.7 mm above the open TCP
and approximately 1.2 mm above the closed TCP because the TCP follows the
physical linkage. The gripper housing remains above the TCP/pinch plane. The
pad boxes are 22 mm wide along their local x contact dimension and each of the
two pad segments is 18.75 mm tall (full dimensions).

## Task inventory

| Task | Class / model | Workspace | Randomized object / goal | Objective and success |
|---|---|---|---|---|
| kuka-reach-v3 | `KukaReachEnvV3` / `kuka_reach_v3.xml` | x ±.25, y .42–.82, z .10–.40 | object x ±.10/y .48–.58; goal x ±.15/y .62–.78/z .12–.32 | Reach; TCP ≤ .05 m |
| kuka-push-v3 | `KukaPushEnvV3` / `kuka_push_v3.xml` | x ±.25, y .42–.75, z **.04–.50** | puck x ±.10/y **.484–.55**; goal x ±.10/y .65–.75 | Push; object ≤ .05 m |
| kuka-handle-press-side-v3 | `KukaHandlePressSideEnvV3` / `kuka_handle_press_side_v3.xml` | x ±.25, y .42–.75, z .05–.50 | fixture x .22–.32/y .45–.60 | Press; handle z ≤ .02 m |
| kuka-button-press-v3 | `KukaButtonPressEnvV3` / `kuka_button_press_v3.xml` | x ±.25, y .42–.75, z .05–.50 | box x ±.10/y .69–.75 | Press; button ≤ .02 m |
| kuka-hammer-v3 | `KukaHammerEnvV3` / `kuka_hammer_v3.xml` | x ±.25, y .42–.75, z **.03–.50** | hammer x ±.10/y .42–.50; fixed nail | Grasp/lift/tool use; nail slide > .09 m |
| kuka-drawer-close-v3 | `KukaDrawerCloseEnvV3` / `kuka_drawer_close_v3.xml` | x ±.25, y .42–.75, z .05–.50 | drawer x ±.10/y .90 | Push; handle ≤ .055 m |
| kuka-faucet-open-v3 | `KukaFaucetOpenEnvV3` / `kuka_faucet_open_v3.xml` | x ±.25, y .42–.75, z .05–.50 | faucet x ±.10/y .65–.70 | Rotate open; handle ≤ .07 m |
| kuka-faucet-close-v3 | `KukaFaucetCloseEnvV3` / shared faucet model | same | same | Rotate close; handle ≤ .07 m |
| kuka-window-open-v3 | `KukaWindowOpenEnvV3` / `kuka_window_open_v3.xml` | x ±.25, y .42–.75, z .05–.50 | window x ±.10/y .70–.85 | Slide open; handle ≤ .05 m |
| kuka-window-close-v3 | `KukaWindowCloseEnvV3` / shared window model | same | same | Slide close; handle ≤ .05 m |
| kuka-door-open-v3 | `KukaDoorOpenEnvV3` / `kuka_door_open_v3.xml` | x ±.25, y .42–.75, z .05–.50 | door x 0–.10/y .65–.75 | Pull/rotate; handle x ≤ .08 m |
| kuka-pick-place-v3 | `KukaPickPlaceEnvV3` / `kuka_pick_place_v3.xml` | x ±.25, y .42–.75, z .03–.50 | object x ±.10/y .45–.55; goal y .65–.75/z .05–.30 | Grasp/lift/place; object ≤ .07 m |

## Isolated calibration results

| Task | Original issue | Root cause class | Change | Old | New | Scripted result | Remaining issue |
|---|---|---|---|---:|---:|---|---|
| Reach | None | — | None | — | — | PASS; task-bank goals/reset finite | None |
| Push | pad above puck | workspace infeasibility | task-specific TCP z floor | .05 m | .04 m | pad contact at step 42; displacement .144 m; goal progress .115 m; no table contact | deterministic controller ends .059 m from goal; physical contact has TCP distance .0505 m, so the unchanged .02 m reward gate does not activate |
| Handle press | stopped .0318 m from target | trajectory/contact mismatch | controller contact offset and deeper endpoint | centered / -.10 m | x +.04, y +.02 / -.14 m | existing .02 m success reached; joint near -.111 m | None |
| Button press | None | — | None | — | — | PASS | None |
| Hammer | bilateral contact but slip/pivot | workspace and grasp geometry | task-specific TCP z floor; head-side grasp and staged transport | .05 m / x +.07 | .03 m / x +.13 | full tool clears table, stays grasped, and nail slide exceeds .09 m | None; no friction change required |
| Drawer close | None | — | None | — | — | PASS | None |
| Faucet open | legacy arc rotates wrong way | trajectory/TCP mismatch | mechanism-kinematic Cartesian sweep | synthetic fixed-radius arc | q=0→1.5 handle path | current success reached | None |
| Faucet close | None in existing close diagnostic | — | None | — | — | PASS | None |
| Window open | None | — | None | — | — | PASS | None |
| Window close | None | — | None | — | — | PASS | None |
| Door open | legacy pull under-rotates | trajectory/TCP mismatch | mechanism-kinematic sweep with pad-face offset | straight pull | q=0→-1.4, offset (-.04,+.08,0) m | hinge < -1 rad and current success reached | None |
| Pick-place | None | — | None | — | — | grasp/lift/transport PASS | None |

## Push workspace feasibility

The push task retains its task-specific Cartesian workspace at x ±0.25 m,
y 0.42–0.75 m, and z 0.04–0.50 m. This is a hardware-informed,
simulation-validated Cartesian task workspace, not a claim of real-world
safety.

An audit tested lower y bounds from 0.420 m through 0.365 m using the fixed
Robotiq orientation. Although lower poses remained free of arm/floor and
Robotiq/floor contact, KUKA joint A4 reached and then slightly exceeded its
modeled/public -120 degree limit near the required worst-case approach pose.
The workspace was therefore not expanded. Instead, the push puck's sampled
lower y bound was raised from 0.450 m to 0.484 m. With the conservative
54 mm pre-contact approach offset, the full continuous sampler now has at
least 10 mm of Cartesian clearance from the y workspace boundary. Runtime
probes at TCP y=0.430 m across x=-0.10/0/+0.10 m and z=0.042/0.047/0.052 m
tracked within 0.1 mm, had a worst measured joint-limit margin of 5.06 degrees,
and produced no arm/floor, Robotiq/floor, or robot self collision.

Public LBR iiwa 7 R800 limits and reach constrain this simulation decision;
the 800 mm nominal maximum reach is not treated as a safe Cartesian sphere.
Real-robot validation is still required before deployment.

The push reward gate is a reward-shaping issue, not a remaining physical
feasibility issue: first real pad contact occurs with `tcp_to_obj≈0.0505 m`,
whereas reward v2 requires `<0.02 m`. This pass intentionally does not change
that gate. The task success radius remains physically attainable, although the
simple fixed-line diagnostic finishes just outside it.

## Regression audit

Fifty randomized resets per registered task were checked. All 600 observations
were finite and all goals matched `obs[36:39]`. The full KUKA test suite and the
dedicated Robotiq integration suite should be run with `MUJOCO_GL=egl` in a
headless environment.
