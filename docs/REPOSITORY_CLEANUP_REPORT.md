# Repository Cleanup Report

Date: 2026-10-13
Primary repository: `metaworld-kuka-v3`
Read-only consumer inspected: `continual-learning`

## 1. Initial repository state

The primary repository was on `main` at `2ed78a0` (`Initial MetaWorld v3 KUKA
adaptation`) and tracked `origin/main`. Before this cleanup it had 15 modified
tracked files and 10 untracked files. Those changes implement or test active
KUKA work and were treated as protected user work. No branch, commit, index,
checkpoint, run output, or Git history was changed.

The consumer repository was on `main` at `598f526` and also had extensive
uncommitted configuration, documentation, script, and test work. It was
inspected only; this cleanup did not modify it.

At audit start the primary checkout contained 470 tracked files (387 under the
vendored `metaworld` runtime, 52 under `metaworld_kuka`, 13 tests, four scripts,
and three documents) plus the 10 protected untracked files. Its total size was
approximately 64 MiB, dominated by required MuJoCo textures and meshes.

## 2. Directory inventory and classification

| Path | Classification | Purpose | Action | Risk |
| --- | --- | --- | --- | --- |
| `metaworld/` | Production, vendored runtime | MetaWorld v3.1.1 runtime, utilities, Sawyer task classes, and shared assets used by KUKA XML and reward code | KEEP | HIGH |
| `metaworld_kuka/envs/` | Production | Fifteen registered KUKA task implementations and reward variants | KEEP | HIGH |
| `metaworld_kuka/assets/kuka_xyz/` | Production | Task MJCF entry points and shared task fragments | KEEP | HIGH |
| `metaworld_kuka/assets/objects/` | Production | KUKA base MJCF and IIWA collision/visual meshes | KEEP | HIGH |
| `metaworld_kuka/assets/meshes/` | Production, third party | Robotiq 2F-85 model, meshes, image, provenance, and license | KEEP | HIGH |
| `tests/metaworld_kuka/` | Production validation | Registration, robot, task, reward, determinism, and controller regression tests | KEEP | LOW |
| `scripts/` | Active diagnostics | Headless geometry/workspace diagnostics and an explicit manual GUI utility | KEEP | MEDIUM |
| `docs/` | Current engineering and historical evidence | Asset provenance and calibrated reward/geometry audit records | KEEP | LOW |
| ignored caches | Generated development artifacts | Python bytecode and pytest cache | KEEP locally; already ignored | LOW |

The package currently has 19 Python files, its KUKA asset subtree has 20 XML
files and 16 STL files, and the test tree has 15 test modules. Packaging rules
include KUKA XML, STL, PNG, license, and README assets.

## 3. Active environments

The single public Gymnasium entry point is `Meta-World-KUKA/MT1`. Its ordered
task registry contains:

1. `kuka-reach-v3`
2. `kuka-push-v3`
3. `kuka-push-cube-v3`
4. `kuka-handle-press-side-v3`
5. `kuka-button-press-v3`
6. `kuka-button-press-topdown-v3`
7. `kuka-peg-unplug-side-v3`
8. `kuka-hammer-v3`
9. `kuka-drawer-close-v3`
10. `kuka-faucet-open-v3`
11. `kuka-faucet-close-v3`
12. `kuka-window-open-v3`
13. `kuka-window-close-v3`
14. `kuka-door-open-v3`
15. `kuka-pick-place-v3`

The public contract is a `(39,)` `float32` observation, `(4,)` action, and
200-step horizon. Shared reward versions `v1` and `v2` remain available.
`v3`, `v3_1`, and `v3_2_retention` remain scoped to the two Push tasks;
`v3_3_contact_push_progress` and `sawyer_push_v2_adapted` remain Cube-only.
Hammer and Peg Unplug variant controls are also retained.

## 4. Asset and reference audit

Every active environment class resolves its MJCF entry point beneath
`metaworld_kuka/assets/kuka_xyz`. The shared KUKA model includes the IIWA and
Robotiq resources beneath `metaworld_kuka/assets`, while task XML deliberately
references shared upstream scene, texture, and object assets beneath
`metaworld/assets`. These relative cross-tree references make asset moves a
high-risk compatibility change.

The close variants intentionally reuse their corresponding open-task XML
(`kuka_faucet_open_v3.xml` and `kuka_window_open_v3.xml`). This is code reuse,
not an orphaned or missing model. `kuka_base_smoke.xml` is referenced by the
base-environment test. The Robotiq README, image, and BSD-2-Clause license are
third-party provenance material and must stay with the meshes.

## 5. Redundant and obsolete candidates

The tracked-file hash audit found six duplicate groups. None is safe to delete:

| Candidate | Evidence and references | Action | Risk |
| --- | --- | --- | --- |
| `handle_press_col1.stl`, `handle_press_col2.stl` | Byte-identical upstream meshes with distinct semantic mesh names used by the object model | DEFER | HIGH |
| `objects/assets/table.xml`, `table_hole.xml` | Byte-identical upstream include-layer files retained under distinct public asset names | DEFER | HIGH |
| button and coffee-machine `button.stl` | Byte-identical upstream assets in separate object namespaces | DEFER | HIGH |
| `objects/block.xml`, `objects/sphere.xml` | Byte-identical upstream compatibility wrappers | DEFER | HIGH |
| button and coffee-machine `buttonring.stl` | Byte-identical upstream assets in separate object namespaces | DEFER | HIGH |
| empty `py.typed` and `__init__.py` marker files | Intentionally empty package/type markers with different packaging roles | KEEP | LOW |

No tracked temporary files, logs, run outputs, checkpoints, videos, backup
files, or generated reports were found. No exact duplicate documentation was
found. Static unused-import cleanup was not performed: Ruff is not installed
in the validated environment, and changing imports without equivalent tooling
and regression evidence would provide little cleanup value.

Ignored `.pytest_cache` and `__pycache__` directories are confirmed generated
development artifacts. They are already covered by `.gitignore`; the request
explicitly disallows bulk deletion of ignored caches, so they were reported and
left intact. Existing ignore rules also cover runs, checkpoints, logs,
TensorBoard data, local environments, editor state, test reports, and secrets.
No `.gitignore` change was necessary.

## 6. Changes performed

- Corrected the README's outdated 12-task list to the 15-task active registry.
- Documented observation/action/horizon contracts, reward-version scope,
  MuJoCo asset ownership, tests, and the consumer-repository boundary.
- Added this audit and cleanup report.

No files were deleted, moved, consolidated, or reorganized. No Python, reward,
physics, collision, XML, controller, sampling, success, or registration code
was changed during cleanup.

## 7. Cross-repository dependencies

The read-only `continual-learning` inspection found direct dependencies on:

- importing `metaworld_kuka` to register `Meta-World-KUKA/MT1`;
- the exact ordered `KUKA_ENVIRONMENTS` task bank and `_make_tasks` payloads;
- all 15 task IDs in the KUKA v3 adapter;
- `(39,)` float32 observations, goal values at indices `36:39`, `(4,)`
  actions, and a 200-step horizon;
- reward versions `v1`, `v2`, `v3`, `v3_1`, `v3_2_retention`,
  `v3_3_contact_push_progress`, and `sawyer_push_v2_adapted`;
- Hammer and Peg Unplug reward-variant arguments;
- `_last_rand_vec`, `_target_pos`, task payload serialization, and stable
  seeded task reconstruction;
- KUKA base-environment reward diagnostics used by viewers and evaluation;
- checkpoint/config compatibility for SAC, sequential SAC, DiffCRL, scripted
  controllers, fixed-task evaluation, and policy rendering.

The consumer also contains historical configurations and experiment documents
whose exact task/reward identities are reproducibility records. Public module
moves, task renames, registry reordering, reward consolidation, or asset-path
changes therefore require a coordinated two-repository migration and are
deferred. No consumer file was modified.

## 8. Before/after validation

The first attempted baseline used the similarly named legacy Conda environment
`metaworld-kuka`. It has Python 3.8 and no `mujoco` package, so collection
failed with 15 import errors. This is a pre-existing local-environment issue,
not a repository failure. The validated environment is
`/home/yikai/anaconda3/envs/metaworld` (Python 3.10.20, MuJoCo 3.3.0,
Gymnasium 1.3.0, pytest 9.1.1).

Validation is run headlessly with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  /home/yikai/anaconda3/envs/metaworld/bin/python -m pytest -q
```

The contract snapshot constructs every registered task twice with seed 42,
loads every MuJoCo model, verifies identical reset observations, records the
task and observation hashes, and checks the public spaces. Push object geometry
is inspected directly from the compiled MuJoCo model to distinguish the
40 mm Cube box from the Cylinder. Focused reward tests cover the historical and
new reward decompositions and finiteness.

Pre-cleanup results in the validated environment:

- Primary full suite: **122 passed** in 211.27 seconds.
- Fifteen registered models loaded successfully; repeated seed-42 resets were
  byte-identical for every environment.
- Every public observation/action shape was `(39,)`/`(4,)`, and every reset
  observation was `float32`.
- Cube Push `objGeom` was MuJoCo box type 6 with half-sizes
  `[0.02, 0.02, 0.02]` (40 mm full side length).
- Cylinder Push `objGeom` was MuJoCo cylinder type 5 with size
  `[0.03, 0.015, 0.0]`.

Post-cleanup results:

- Primary full suite: **122 passed** in 210.54 seconds. This matches the
  pre-cleanup pass count.
- Consumer validation (read-only): **126 tests and 44 subtests passed**. It
  covered KUKA SAC adapters, Cube integration/configurations, all Cube reward
  configurations, reward-version validation, and the checkpoint viewer CLI.
- Because cleanup changed documentation only, the previously recorded seeded
  task/reset hashes and compiled model geometry are unchanged by construction;
  the full environment tests independently reconstructed the models again.

## 9. Remaining technical debt and recommendations

1. **DEFER / HIGH:** Do not deduplicate or relocate vendored upstream assets
   without building a complete MJCF reference graph and coordinating the
   consumer repository.
2. **DEFER / HIGH:** Do not reorganize `metaworld_kuka.envs` or change exports;
   external code imports the current classes and registry directly.
3. **DEFER / MEDIUM:** Decide separately whether diagnostic scripts need a
   supported CLI namespace. Moving them now would break documented commands.
4. **DEFER / MEDIUM:** Consider adding Ruff to the declared development tools,
   then perform unused-import cleanup in its own behavior-neutral change.
5. **KEEP:** Retain historical reward versions, calibration reports, and task
   ordering for existing checkpoints and fixed-task hashes.
6. **KEEP:** Retain ignored caches and local outputs unless the owner requests a
   targeted, recoverable deletion in a separate operation.

The repository is already organized around the conventional package, tests,
scripts, and docs boundaries. Further cosmetic movement would add more
compatibility risk than cleanup benefit at this stage.
