"""Reusable runtime geometry audit for the formal KUKA/Robotiq task family.

Examples:
    python scripts/diagnose_kuka_geometry.py --task all --samples 1
    python scripts/diagnose_kuka_geometry.py --task kuka-push-v3 --samples 50
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mujoco
import numpy as np

from metaworld_kuka.registration import KUKA_ENVIRONMENTS


def configured(env_cls, seed: int):
    env = env_cls(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    return env


def geom_points(env, name: str) -> np.ndarray:
    """Return runtime world-space support points for an active collision geom."""
    geom = env.model.geom(name)
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = env.model.mesh(geom.dataid[0])
        local = env.model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
    else:
        sx, sy, sz = geom.size
        if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            sx = sy = sz = sx
        elif geom.type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            sy, sz = sx, sy + sx
        local = np.array(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
        )
    data_geom = env.data.geom(name)
    return local @ data_geom.xmat.reshape(3, 3).T + data_geom.xpos


def bounds(env, names: tuple[str, ...]) -> dict[str, Any]:
    points = np.vstack([geom_points(env, name) for name in names])
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def nearest_pad_distance(env, object_pos: np.ndarray) -> float:
    points = np.vstack([geom_points(env, name) for name in env.ROBOTIQ_PAD_GEOMS])
    return float(np.min(np.linalg.norm(points - object_pos, axis=1)))


def contact_pairs(env) -> list[tuple[str, str]]:
    return [
        (env.model.geom(c.geom1).name, env.model.geom(c.geom2).name)
        for c in env.data.contact
    ]


def robot_reference() -> dict[str, Any]:
    env = configured(next(iter(KUKA_ENVIRONMENTS.values())), 1)
    try:
        env.reset()
        result = {}
        for command in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for _ in range(80):
                env.step(np.array([0, 0, 0, command], dtype=np.float32))
            tcp = env.tcp_center.copy()
            pads = bounds(env, env.ROBOTIQ_PAD_GEOMS)
            housing = bounds(
                env, ("robotiq_base_mount_collision", "robotiq_base_collision")
            )
            result[str(command)] = {
                "control": float(env.data.ctrl[0]),
                "tcp": tcp.tolist(),
                "hand": env.data.body("hand").xpos.tolist(),
                "right_inner_pad": env._get_site_pos("rightEndEffector").tolist(),
                "left_inner_pad": env._get_site_pos("leftEndEffector").tolist(),
                "aperture": env.gripper_aperture,
                "normalized_aperture": float(env._get_obs()[3]),
                "pad_bounds": pads,
                "housing_bounds": housing,
                "lowest_pad_below_tcp": float(tcp[2] - pads["min"][2]),
                "lowest_housing_below_tcp": float(tcp[2] - housing["min"][2]),
            }
        return result
    finally:
        env.close()


def task_snapshot(name: str, env, sample: int) -> dict[str, Any]:
    obs, _ = env.reset()
    obj = np.asarray(env._get_pos_objects(), dtype=float).reshape(-1)[:3]
    target = np.asarray(env._target_pos, dtype=float)
    return {
        "task": name,
        "sample": sample,
        "tcp": env.tcp_center.tolist(),
        "hand": env.data.body("hand").xpos.tolist(),
        "aperture": env.gripper_aperture,
        "object": obj.tolist(),
        "target": target.tolist(),
        "workspace_low": env.mocap_low.tolist(),
        "workspace_high": env.mocap_high.tolist(),
        "nearest_pad_object_distance": nearest_pad_distance(env, obj),
        "contacts": contact_pairs(env),
        "observation_finite": bool(np.isfinite(obs).all()),
        "goal_matches_observation": bool(np.allclose(obs[36:39], target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=("all", *KUKA_ENVIRONMENTS))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    selected = KUKA_ENVIRONMENTS.items()
    if args.task != "all":
        selected = ((args.task, KUKA_ENVIRONMENTS[args.task]),)
    report: dict[str, Any] = {"robotiq_reference": robot_reference(), "tasks": {}}
    for name, env_cls in selected:
        env = configured(env_cls, args.seed)
        try:
            report["tasks"][name] = [
                task_snapshot(name, env, sample) for sample in range(args.samples)
            ]
        finally:
            env.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
