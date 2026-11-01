# KUKA 40 mm Cube Push baseline

`kuka-push-cube-v3` is a single Sawyer Push v2 adaptation. It uses the KUKA
iiwa, Robotiq 2F-85, 4D Cartesian/gripper actions, 39D observations, and the
normal 200-step horizon. There are no curricula or alternate initializations.

The MuJoCo box half-size is `(0.02, 0.02, 0.02)`, giving a 40 mm cube. Its
100 g mass is a retained simulation assumption; box inertia is the analytic
uniform-cube value for that assumed mass. Friction and solver/contact parameters
are likewise reasonable simulation assumptions, not a physical calibration. The upright
quaternion is fixed on every reset; position and goal are randomized in the
KUKA-safe Push workspace with at least 0.15 m separation.

For distance `d` from cube center to goal, `in_place` is MetaWorld's long-tail
tolerance with bounds `(0, 0.05)` and margin equal to initial goal distance.
The reward is `2*caging`; when TCP distance is below 0.055 m and the gripper is
open it becomes `1 + 4*caging + 5*in_place`; when `d < 0.05` it is 10. Reported
success uses `d <= 0.05`, exactly matching Sawyer's boundary operators.

The Robotiq pads open along world Y, so Sawyer's pad-axis and X-Z Hamacher
construction is retained. The scalar cube approximation is its center-corner
radius `sqrt(3)*0.02 = 0.03464 m`; pad threshold is 0.055 m. X-Z tolerance is
0.025 m to include the table-supported cube's 20 mm center-height offset. These
are geometry-derived baseline assumptions requiring future hardware calibration.

Former progress, contact, retention, handoff, near-start, and curriculum modes
are intentionally unsupported. Checkpoints trained with them must not be
reported as policies for this baseline.
