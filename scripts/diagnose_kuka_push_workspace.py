"""Audit the KUKA push Cartesian workspace against the compiled MuJoCo model.

This diagnostic deliberately reports simulation evidence, not real-robot safety.
It drives the fixed KUKA/Robotiq orientation to task-relevant TCP poses, holds
each pose, and records tracking, joint-limit, contact, and solver-health data.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from typing import Any

import mujoco
import numpy as np

from metaworld import _make_tasks
from metaworld_kuka.registration import KUKA_ENVIRONMENTS


TASK = "kuka-push-v3"
APPROACH_OFFSET = 0.054
CONTACT_OFFSET = 0.044
PUSH_Z = 0.047
GRID_X = (-0.10, 0.0, 0.10)
GRID_Z = (0.042, PUSH_Z, 0.052)
DEFAULT_Y_LOW = (0.420, 0.400, 0.390, 0.385, 0.375, 0.365)
ARM_JOINTS = tuple(f"kuka_joint_{i}" for i in range(1, 8))
PAD_GEOMS = {"right_pad1", "right_pad2", "left_pad1", "left_pad2"}
GRIPPER_BODIES = {
    "base_mount",
    "base",
    "hand",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
    "right_silicone_pad",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
    "left_silicone_pad",
}


def task_bank(seed: int):
    args = {TASK: {"args": [], "kwargs": {"task_id": 0}}}
    return _make_tasks(
        KUKA_ENVIRONMENTS,
        args,
        {"partially_observable": False},
        seed=seed,
    )


def make_env(task=None):
    env = KUKA_ENVIRONMENTS[TASK](reward_function_version="v2")
    env._set_task_called = True
    env._partially_observable = False
    if task is not None:
        env.set_task(task)
    else:
        env.seed(7)
        env.seeded_rand_vec = True
        env._freeze_rand_vec = False
    env.reset()
    return env


def geom_body(env, geom_id: int) -> str:
    body_id = int(env.model.geom_bodyid[geom_id])
    return env.model.body(body_id).name or "world"


def contact_audit(env) -> dict[str, Any]:
    pairs = []
    arm_floor = False
    gripper_floor = False
    self_collision = False
    object_contact = False
    for contact in env.data.contact:
        geom1 = env.model.geom(contact.geom1).name or f"geom:{contact.geom1}"
        geom2 = env.model.geom(contact.geom2).name or f"geom:{contact.geom2}"
        body1 = geom_body(env, contact.geom1)
        body2 = geom_body(env, contact.geom2)
        names = {geom1, geom2}
        bodies = {body1, body2}
        pairs.append((geom1, geom2))
        object_contact |= "objGeom" in names and bool(names & PAD_GEOMS)
        if "floor" in names:
            gripper_floor |= bool(bodies & GRIPPER_BODIES)
            arm_floor |= any(body.startswith("kuka_link_") for body in bodies)
        arm1 = body1.startswith("kuka_link_")
        arm2 = body2.startswith("kuka_link_")
        gripper1 = body1 in GRIPPER_BODIES
        gripper2 = body2 in GRIPPER_BODIES
        # Contacts between the two closed 2F-85 pad chains are expected and
        # are not robot self-collision. Link 7 and the mounted gripper are also
        # an adjacent assembly. Retain non-adjacent arm/arm and arm/gripper
        # contacts as self-collision evidence.
        adjacent_mount = (body1 == "kuka_link_7" and gripper2) or (
            body2 == "kuka_link_7" and gripper1
        )
        self_collision |= (
            (arm1 and arm2 and body1 != body2)
            or ((arm1 and gripper2) or (arm2 and gripper1)) and not adjacent_mount
        )
    return {
        "pairs": pairs,
        "arm_floor": arm_floor,
        "gripper_floor": gripper_floor,
        "self_collision": self_collision,
        "robotiq_object": object_contact,
    }


def joint_audit(env) -> dict[str, Any]:
    angles = []
    margins = []
    for name in ARM_JOINTS:
        joint = env.model.joint(name)
        value = float(env.data.joint(name).qpos[0])
        low, high = map(float, joint.range)
        angles.append(value)
        margins.append(min(value - low, high - value))
    return {
        "angles_rad": angles,
        "angles_deg": np.rad2deg(angles).tolist(),
        "limit_margin_rad": margins,
        "limit_margin_deg": np.rad2deg(margins).tolist(),
        "minimum_limit_margin_rad": min(margins),
        "minimum_limit_margin_deg": float(np.rad2deg(min(margins))),
    }


def move_and_hold(env, target: np.ndarray, *, move_steps=100, hold_steps=30):
    elevated = target.copy()
    elevated[2] = max(target[2] + 0.10, 0.15)
    max_qvel = 0.0
    warnings_before = int(env._did_see_sim_exception)
    for waypoint, steps in ((elevated, move_steps), (target, move_steps)):
        for _ in range(steps):
            action = np.r_[
                np.clip((waypoint - env.tcp_center) / env.action_scale, -1, 1),
                1.0,
            ].astype(np.float32)
            env.step(action)
            max_qvel = max(max_qvel, float(np.max(np.abs(env.data.qvel))))
    held = []
    contact_flags = []
    for _ in range(hold_steps):
        action = np.r_[
            np.clip((target - env.tcp_center) / env.action_scale, -1, 1), 1.0
        ].astype(np.float32)
        env.step(action)
        held.append(env.tcp_center.copy())
        contact_flags.append(contact_audit(env))
        max_qvel = max(max_qvel, float(np.max(np.abs(env.data.qvel))))
    tcp = env.tcp_center.copy()
    tracking_error = float(np.linalg.norm(tcp - target))
    finite = bool(
        np.isfinite(env.data.qpos).all()
        and np.isfinite(env.data.qvel).all()
        and np.isfinite(tcp).all()
    )
    contacts = {
        key: any(row[key] for row in contact_flags)
        for key in ("arm_floor", "gripper_floor", "self_collision", "robotiq_object")
    }
    contacts["pairs"] = sorted(
        {tuple(pair) for row in contact_flags for pair in row["pairs"]}
    )
    joints = joint_audit(env)
    hold_drift = float(np.max(np.linalg.norm(np.asarray(held) - held[-1], axis=1)))
    solver_warning = bool(env._did_see_sim_exception) and not warnings_before
    if (
        not finite
        or solver_warning
        or joints["minimum_limit_margin_deg"] < 0.0
        or any(
            contacts[key]
            for key in ("arm_floor", "gripper_floor", "self_collision")
        )
        or tracking_error > 0.010
    ):
        classification = "FAIL"
    elif tracking_error > 0.005 or joints["minimum_limit_margin_deg"] < 5.0:
        classification = "MARGINAL"
    else:
        classification = "PASS"
    return {
        "requested_tcp": target.tolist(),
        "realized_tcp": tcp.tolist(),
        "mocap": env.data.mocap_pos[0].tolist(),
        "tracking_error_m": tracking_error,
        "hold_drift_m": hold_drift,
        "maximum_abs_qvel": max_qvel,
        "finite": finite,
        "solver_warning": solver_warning,
        "joints": joints,
        "contacts": contacts,
        "classification": classification,
    }


def geometry_reference() -> dict[str, Any]:
    env = make_env()
    try:
        joints = []
        for name in ARM_JOINTS:
            joint = env.model.joint(name)
            joints.append(
                {
                    "name": name,
                    "range_rad": joint.range.tolist(),
                    "range_deg": np.rad2deg(joint.range).tolist(),
                }
            )
        for _ in range(100):
            env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        open_aperture = env.gripper_aperture
        for _ in range(100):
            env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return {
            "arm_joints": joints,
            "open_aperture_m": open_aperture,
            "closed_aperture_m": env.gripper_aperture,
            "base_position": env.data.body("kuka_link_0").xpos.tolist(),
            "link7_position": env.data.body("kuka_link_7").xpos.tolist(),
            "mount_position": env.data.body("base_mount").xpos.tolist(),
            "tcp_position": env.tcp_center.tolist(),
            "mocap_quaternion": env.MOCAP_QUATERNION.tolist(),
        }
    finally:
        env.close()


def task_poses(task) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rand_vec = np.asarray(pickle.loads(task.data)["rand_vec"], dtype=float)
    obj = rand_vec[:2]
    goal = rand_vec[3:5]
    direction = goal - obj
    direction /= np.linalg.norm(direction)
    return obj, goal, {
        "approach": np.r_[obj - APPROACH_OFFSET * direction, PUSH_Z],
        "contact": np.r_[obj - CONTACT_OFFSET * direction, PUSH_Z],
        "final": np.r_[goal - CONTACT_OFFSET * direction, PUSH_Z],
    }


def audit_pose(task, target: np.ndarray, y_low: float):
    env = make_env(task)
    try:
        env.mocap_low[1] = y_low
        return move_and_hold(env, target)
    finally:
        env.close()


def audit_bank(tasks, y_low: float) -> dict[str, Any]:
    rows = []
    for index, task in enumerate(tasks):
        obj, goal, poses = task_poses(task)
        env = make_env(task)
        try:
            env.mocap_low[1] = y_low
            pose_results = {}
            for name, target in poses.items():
                env.reset()
                pose_results[name] = move_and_hold(env, target)
        finally:
            env.close()
        worst_tracking = max(row["tracking_error_m"] for row in pose_results.values())
        joint_margin = min(
            row["joints"]["minimum_limit_margin_deg"] for row in pose_results.values()
        )
        cartesian_margin = min(
            min(target[0] + 0.25, 0.25 - target[0], target[1] - y_low, 0.75 - target[1])
            for target in poses.values()
        )
        classes = {row["classification"] for row in pose_results.values()}
        if "FAIL" in classes or cartesian_margin < 0:
            classification = "FAIL"
        elif "MARGINAL" in classes or cartesian_margin < 0.010:
            classification = "MARGINAL"
        else:
            classification = "PASS"
        rows.append(
            {
                "index": index,
                "object": obj.tolist(),
                "goal": goal.tolist(),
                "poses": {name: value.tolist() for name, value in poses.items()},
                "minimum_workspace_margin_m": cartesian_margin,
                "minimum_joint_limit_margin_deg": joint_margin,
                "maximum_tracking_error_m": worst_tracking,
                "collision": any(
                    result["contacts"][key]
                    for result in pose_results.values()
                    for key in ("arm_floor", "gripper_floor", "self_collision")
                ),
                "pose_results": pose_results,
                "classification": classification,
            }
        )
    counts = Counter(row["classification"] for row in rows)
    return {
        "y_low": y_low,
        "counts": {key: counts.get(key, 0) for key in ("PASS", "MARGINAL", "FAIL")},
        "worst_workspace_margin_m": min(row["minimum_workspace_margin_m"] for row in rows),
        "worst_joint_limit_margin_deg": min(
            row["minimum_joint_limit_margin_deg"] for row in rows
        ),
        "worst_tracking_error_m": max(row["maximum_tracking_error_m"] for row in rows),
        "collision_rows": [row["index"] for row in rows if row["collision"]],
        "rows": rows,
    }


def audit_grid(y_low: float) -> dict[str, Any]:
    rows = []
    env = make_env()
    try:
        for x in GRID_X:
            for z in GRID_Z:
                env.reset()
                env.mocap_low[1] = y_low
                row = move_and_hold(env, np.array([x, y_low, z], dtype=float))
                row.update({"x": x, "y": y_low, "z": z})
                rows.append(row)
    finally:
        env.close()
    counts = Counter(row["classification"] for row in rows)
    return {
        "y_low": y_low,
        "counts": {key: counts.get(key, 0) for key in ("PASS", "MARGINAL", "FAIL")},
        "worst_joint_limit_margin_deg": min(
            row["joints"]["minimum_limit_margin_deg"] for row in rows
        ),
        "worst_tracking_error_m": max(row["tracking_error_m"] for row in rows),
        "collision_points": [
            [row["x"], row["y"], row["z"]]
            for row in rows
            if any(
                row["contacts"][key]
                for key in ("arm_floor", "gripper_floor", "self_collision")
            )
        ],
        "rows": rows,
    }


def continuous_geometry(y_low: float, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    configurations = []
    # Include sampler corners and axis-extreme combinations before random draws.
    for ox in (-0.10, 0.10):
        for oy in (0.484, 0.55):
            for gx in (-0.10, 0.10):
                for gy in (0.65, 0.75):
                    configurations.append((np.array([ox, oy]), np.array([gx, gy])))
    while len(configurations) < samples + 16:
        obj = rng.uniform((-0.10, 0.484), (0.10, 0.55))
        goal = rng.uniform((-0.10, 0.65), (0.10, 0.75))
        if np.linalg.norm(goal - obj) >= 0.15:
            configurations.append((obj, goal))
    margins = []
    worst = None
    for obj, goal in configurations:
        direction = goal - obj
        direction /= np.linalg.norm(direction)
        poses = (
            obj - APPROACH_OFFSET * direction,
            obj - CONTACT_OFFSET * direction,
            goal - CONTACT_OFFSET * direction,
        )
        margin = min(
            min(point[0] + 0.25, 0.25 - point[0], point[1] - y_low, 0.75 - point[1])
            for point in poses
        )
        margins.append(margin)
        if worst is None or margin < worst["margin_m"]:
            worst = {
                "object": obj.tolist(),
                "goal": goal.tolist(),
                "direction": direction.tolist(),
                "approach": poses[0].tolist(),
                "margin_m": margin,
            }
    return {
        "sample_count": len(configurations),
        "minimum_margin_m": min(margins),
        "below_zero": sum(value < 0 for value in margins),
        "below_10mm": sum(value < 0.010 for value in margins),
        "worst": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument("--continuous-samples", type=int, default=10_000)
    parser.add_argument("--output")
    parser.add_argument("--y-low", type=float, nargs="*", default=DEFAULT_Y_LOW)
    args = parser.parse_args()
    tasks = task_bank(args.task_set_seed)
    report = {
        "scope": "hardware-informed simulation audit; not real-world safety",
        "task": TASK,
        "task_set_seed": args.task_set_seed,
        "approach_offset_m": APPROACH_OFFSET,
        "contact_offset_m": CONTACT_OFFSET,
        "push_z_m": PUSH_Z,
        "classification": {
            "tracking_fail_above_m": 0.010,
            "tracking_marginal_above_m": 0.005,
            "joint_marginal_below_deg": 5.0,
            "workspace_marginal_below_m": 0.010,
        },
        "geometry": geometry_reference(),
        "candidates": [],
    }
    for y_low in args.y_low:
        report["candidates"].append(
            {
                "y_low": y_low,
                "grid": audit_grid(y_low),
                "bank": audit_bank(tasks, y_low),
                "continuous": continuous_geometry(
                    y_low, args.continuous_samples, args.task_set_seed
                ),
            }
        )
    encoded = json.dumps(
        report,
        indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else value,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
