"""Contract and physical sanity tests for KUKA side-handle press V3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld_kuka.envs import KukaHandlePressSideEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID


def make_direct(seed: int = 7) -> KukaHandlePressSideEnvV3:
    env = KukaHandlePressSideEnvV3(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move_tcp(env, target: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        action = np.r_[
            np.clip((target - env.tcp_center) / env.action_scale, -1, 1), -1.0
        ].astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated and not truncated
        assert "success" in info


def test_direct_contract_resets_steps_and_physical_press() -> None:
    env = make_direct()
    try:
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (39,) and np.isfinite(obs).all()
            assert info == {} and env.action_space.shape == (4,)
            np.testing.assert_allclose(obs[:18], obs[18:36])
            np.testing.assert_allclose(obs[36:39], env._target_pos)
            assert env._last_rand_vec.shape == (3,)

        # Use a fixed representative task for the deterministic controller;
        # the resets above separately cover the full randomization contract.
        env.seed(2)
        env.reset()
        handle_start = env._get_pos_objects().copy()
        joint_start = float(env.data.joint("goal_slidez").qpos[0])
        # Offset places the inner pad face over the side handle rather than
        # centering the wider Robotiq housing on it.
        contact_offset = np.array([0.04, 0.02, 0.0])
        move_tcp(env, handle_start + contact_offset + [0.0, 0.0, 0.04], 30)
        move_tcp(env, handle_start + contact_offset + [0.0, 0.0, -0.14], 120)
        joint_end = float(env.data.joint("goal_slidez").qpos[0])
        target_dist = abs(env._get_pos_objects()[2] - env._target_pos[2])
        assert joint_end < joint_start - 0.05
        assert target_dist <= env.TARGET_RADIUS
    finally:
        env.close()


def test_registered_horizon_and_rgb_render() -> None:
    assert "kuka-handle-press-side-v3" not in ALL_V3_ENVIRONMENTS
    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-handle-press-side-v3",
        seed=17,
        reward_function_version="v2",
        disable_env_checker=True,
    )
    try:
        obs, _ = env.reset()
        assert obs.shape == (39,) and obs.dtype == np.float32
        assert env.action_space.shape == (4,)
        for step in range(1, 201):
            result = env.step(env.action_space.sample())
            assert len(result) == 5
            obs, reward, terminated, truncated, info = result
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated and truncated is (step == 200)
            assert "success" in info
    finally:
        env.close()

    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-handle-press-side-v3",
        seed=19,
        reward_function_version="v2",
        render_mode="rgb_array",
        camera_name="corner",
        disable_env_checker=True,
    )
    try:
        env.reset()
        assert env.render().shape == (480, 480, 3)
    finally:
        env.close()
