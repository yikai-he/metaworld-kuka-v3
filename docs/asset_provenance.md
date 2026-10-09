# Simulation asset provenance

This file records the provenance that could be verified for third-party robot
assets distributed with this repository. It does not assign authorship beyond
what the cited upstream repositories state.

## KUKA IIWA link meshes

The following files were imported into this repository in commit
`c2def3789e869dcabd3f6485b3b6b7ca08f310d6`:

- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_0.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_1.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_2.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_3.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_4.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_5.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_6.stl`
- `metaworld_kuka/assets/objects/meshes/kuka/kuka_link_7.stl`

Their direct source is the public
[`Ghiara/Metaworld-KUKA-IIWA-R800`](https://github.com/Ghiara/Metaworld-KUKA-IIWA-R800)
repository, where the same eight filenames are stored under
`metaworld/envs/assets/meshes/kuka/`. That repository identifies itself as a
modified MetaWorld environment containing KUKA IIWA R800 tasks and declares an
MIT license at repository level. No narrower or conflicting notice is present
in the mesh directory. The repository-level MIT grant therefore covers the
copy obtained from that source; retain the top-level MIT `LICENSE` when
redistributing it.

The source repository does not identify an earlier author or CAD source for
these eight files. This record therefore confirms the direct source and its
published redistribution license, but makes no claim about earlier provenance.

## Robotiq 2F-85 model

The Robotiq model under
`metaworld_kuka/assets/meshes/kuka/robotiq_2f85/` was imported from
`Ghiara/Metaworld-KUKA-IIWA-R800` commit
`256a721408df1585a6c04e9c5556e857cc760e06`. Its local `README.md` records the
model derivation, and its local `LICENSE` is the ROS-Industrial BSD-2-Clause
license. Both files must remain with the model.
