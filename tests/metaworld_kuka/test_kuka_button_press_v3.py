"""Contract and horizontal-press physics tests for KUKA Button Press V3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld_kuka.envs import KukaButtonPressEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID


def make_direct(seed: int = 3) -> KukaButtonPressEnvV3:
    env = KukaButtonPressEnvV3(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move_tcp(env, target: np.ndarray, steps: int) -> dict:
    info = {}
    for _ in range(steps):
        action = np.r_[
            np.clip((target - env.tcp_center) / env.action_scale, -1, 1), 1.0
        ].astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated and not truncated
    return info


def test_direct_resets_and_horizontal_button_is_physically_pressable() -> None:
    env = make_direct()
    try:
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (39,) and np.isfinite(obs).all() and info == {}
            assert env.action_space.shape == (4,)
            np.testing.assert_allclose(obs[:18], obs[18:36])
            np.testing.assert_allclose(obs[36:39], env._target_pos)
            assert env._last_rand_vec.shape == (3,)

        env.seed(3)
        env.reset()
        start = env._get_pos_objects().copy()
        joint_start = float(env.data.joint("btnbox_joint").qpos[0])
        move_tcp(env, start + np.array([0.0, -0.05, 0.0]), 35)
        info = move_tcp(env, start + np.array([0.0, 0.20, 0.0]), 120)
        joint_end = float(env.data.joint("btnbox_joint").qpos[0])
        assert joint_end < joint_start - 0.05
        assert env._get_pos_objects()[1] > start[1]  # negative-Y axis times negative q
        assert info["success"] == 1.0
    finally:
        env.close()


def test_registered_horizon_and_rgb_render() -> None:
    assert "kuka-button-press-v3" not in ALL_V3_ENVIRONMENTS
    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-button-press-v3",
        seed=17,
        reward_function_version="v2",
        disable_env_checker=True,
    )
    try:
        obs, _ = env.reset()
        assert obs.shape == (39,) and obs.dtype == np.float32
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
        env_name="kuka-button-press-v3",
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
