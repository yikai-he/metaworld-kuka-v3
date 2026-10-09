"""Deterministic per-timestep reward audit for KUKA/Robotiq push."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from metaworld_kuka.envs import KukaPushEnvV3

GATE_CANDIDATES = (0.02, 0.025, 0.03, 0.04, 0.05, 0.055, 0.06)
PROGRESS_EPSILON = 1e-5


def make_env(seed: int) -> KukaPushEnvV3:
    env = KukaPushEnvV3(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    return env


def pad_metrics(env: KukaPushEnvV3) -> tuple[float, list[tuple[str, str]]]:
    obj_id = env._get_id_main_object()
    pad_ids = {env.data.geom(name).id for name in env.ROBOTIQ_PAD_GEOMS}
    distances = [
        mujoco.mj_geomDistance(env.model, env.data, pad_id, obj_id, 1.0, None)
        for pad_id in pad_ids
    ]
    contacts = []
    for contact in env.data.contact:
        pair = {contact.geom1, contact.geom2}
        if obj_id in pair and pair & pad_ids:
            contacts.append(
                (
                    env.model.geom(contact.geom1).name,
                    env.model.geom(contact.geom2).name,
                )
            )
    return float(min(distances)), contacts


def rollout(seed: int) -> dict[str, Any]:
    env = make_env(seed)
    try:
        obs, _ = env.reset()
        initial_obj = env.get_body_com("obj").copy()
        previous_obj = initial_obj.copy()
        previous_goal_distance = float(np.linalg.norm(initial_obj - env._target_pos))
        initial_goal_distance = previous_goal_distance
        direction = env._target_pos[:2] - initial_obj[:2]
        direction /= np.linalg.norm(direction)
        behind = initial_obj.copy()
        behind[:2] -= 0.040 * direction
        behind[2] = 0.047
        stages = [("above", behind + [0, 0, 0.10], 25), ("approach", behind, 40)]
        stages += [
            ("push", behind + np.r_[direction * distance, 0.0], 1)
            for distance in np.linspace(0.0, 0.25, 150)
        ]

        rows = []
        timestep = 0
        for stage, target, steps in stages:
            for _ in range(steps):
                action = np.r_[
                    np.clip((target - env.tcp_center) / env.action_scale, -1, 1),
                    1.0,
                ].astype(np.float32)
                obs, reward, _, _, info = env.step(action)
                timestep += 1
                obj = env.get_body_com("obj").copy()
                goal_distance = float(np.linalg.norm(obj - env._target_pos))
                step_progress = previous_goal_distance - goal_distance
                min_pad_distance, contacts = pad_metrics(env)
                components = env.compute_reward(action, obs)
                rows.append(
                    {
                        "timestep": timestep,
                        "stage": stage,
                        "tcp_to_obj": float(components[1]),
                        "minimum_pad_to_object_distance": min_pad_distance,
                        "physical_pad_contact": bool(contacts),
                        "contact_pairs": contacts,
                        "object_displacement": float(np.linalg.norm(obj - initial_obj)),
                        "object_step_displacement": float(np.linalg.norm(obj - previous_obj)),
                        "object_to_goal": goal_distance,
                        "object_to_goal_progress": initial_goal_distance - goal_distance,
                        "step_progress": step_progress,
                        "positive_progress": bool(step_progress > PROGRESS_EPSILON),
                        "gripper_aperture": env.gripper_aperture,
                        "action_gripper": float(action[3]),
                        "object_grasped": float(components[4]),
                        "in_place": float(components[5]),
                        "passes_2cm_gate": bool(components[1] < 0.02),
                        "reward": float(reward),
                        "success": bool(info["success"]),
                    }
                )
                previous_obj = obj
                previous_goal_distance = goal_distance
        contact_rows = [row for row in rows if row["physical_pad_contact"]]
        return {
            "seed": seed,
            "initial_object": initial_obj.tolist(),
            "target": env._target_pos.tolist(),
            "first_contact": contact_rows[0] if contact_rows else None,
            "final_success": rows[-1]["success"],
            "final_object_displacement": rows[-1]["object_displacement"],
            "final_goal_progress": rows[-1]["object_to_goal_progress"],
            "rows": rows,
        }
    finally:
        env.close()


def distribution(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = np.asarray([row["tcp_to_obj"] for row in rows])
    return {
        "count": len(rows),
        "min": float(values.min()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def summarize(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for rollout_data in rollouts for row in rollout_data["rows"]]
    contact = [row for row in rows if row["physical_pad_contact"]]
    progress = [row for row in rows if row["positive_progress"]]

    def gate_fractions(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            str(gate): float(np.mean([row["tcp_to_obj"] < gate for row in selected]))
            if selected
            else 0.0
            for gate in GATE_CANDIDATES
        }

    return {
        "rollouts": len(rollouts),
        "rollouts_with_contact": sum(r["first_contact"] is not None for r in rollouts),
        "rollouts_with_positive_progress": sum(
            any(row["positive_progress"] for row in r["rows"]) for r in rollouts
        ),
        "successful_rollouts": sum(r["final_success"] for r in rollouts),
        "progress_epsilon": PROGRESS_EPSILON,
        "tcp_to_obj_during_contact": distribution(contact),
        "tcp_to_obj_during_positive_progress": distribution(progress),
        "gate_fraction_during_contact": gate_fractions(contact),
        "gate_fraction_during_positive_progress": gate_fractions(progress),
        "contact_progress_rows": sum(
            row["physical_pad_contact"] and row["positive_progress"] for row in rows
        ),
        "progress_while_2cm_gate_false": sum(
            row["positive_progress"] and not row["passes_2cm_gate"] for row in rows
        ),
        "total_positive_progress_rows": len(progress),
        "reward_min": min(row["reward"] for row in rows),
        "reward_max": max(row["reward"] for row in rows),
        "all_rewards_finite": bool(all(np.isfinite(row["reward"]) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rollouts = [rollout(seed) for seed in range(args.seed_start, args.seed_start + args.rollouts)]
    report = {"summary": summarize(rollouts), "rollouts": rollouts}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
