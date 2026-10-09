"""Regression coverage for diagnostic-only KUKA manual stepping."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import metaworld_kuka  # noqa: F401  Register the isolated KUKA environments.
from metaworld_kuka.registration import KUKA_MT1_ID
from scripts.manual_control_kuka import rollover_manual_bookkeeping, step_manual


def make_push_env() -> gym.Env:
    return gym.make(
        KUKA_MT1_ID,
        env_name="kuka-push-v3",
        seed=10000,
        reward_function_version="v2",
        terminate_on_success=False,
        disable_env_checker=True,
    )


def physical_state(env: gym.Env) -> tuple[np.ndarray, ...]:
    base = env.unwrapped
    return (
        base.data.qpos.copy(),
        base.data.qvel.copy(),
        np.asarray(base._get_pos_objects()).copy(),
        np.asarray(base.tcp_center).copy(),
        np.asarray([base._get_obs()[3]]),
    )


def test_manual_step_runs_1000_actions_without_reset_or_state_jump() -> None:
    env = make_push_env()
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    try:
        obs, _ = env.reset(seed=10000)
        task_configuration = env.unwrapped._last_rand_vec.copy()
        states: dict[int, tuple[np.ndarray, ...]] = {0: physical_state(env)}

        for manual_step in range(1, 1001):
            if env.unwrapped.curr_path_length >= env.unwrapped.max_path_length:
                before_rollover = physical_state(env)
                assert rollover_manual_bookkeeping(env)
                after_rollover = physical_state(env)
                for before_value, after_value in zip(
                    before_rollover, after_rollover, strict=True
                ):
                    assert np.array_equal(before_value, after_value)
            obs, reward, _, _, _ = step_manual(env, action)
            assert obs.shape == (39,)
            assert action.shape == (4,)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert np.array_equal(env.unwrapped._last_rand_vec, task_configuration)
            if manual_step in {
                199, 200, 201, 399, 400, 401, 499, 500, 501,
                599, 600, 601, 799, 800, 801, 999, 1000,
            }:
                states[manual_step] = physical_state(env)

        assert env.unwrapped.curr_path_length == env.unwrapped.max_path_length
        for boundary in (200, 400, 500, 600, 800):
            before = states[boundary]
            after = states[boundary + 1]
            assert np.linalg.norm(after[0] - before[0]) < 0.05
            assert np.linalg.norm(after[2] - before[2]) < 0.01
            assert np.linalg.norm(after[3] - before[3]) < 0.01
            assert np.linalg.norm(after[4] - before[4]) < 0.05
    finally:
        env.close()


def test_registered_environment_retains_formal_episode_semantics() -> None:
    env = make_push_env()
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    try:
        env.reset(seed=10000)
        for step in range(1, 201):
            _, _, _, truncated, _ = env.step(action)
        assert step == 200
        assert truncated

        for _ in range(300):
            env.step(action)
        with pytest.raises(
            ValueError, match="You must reset the env manually once truncate==True"
        ):
            env.step(action)
    finally:
        env.close()
