"""Contract and manipulation physics tests for KUKA Hammer V3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld_kuka.envs import KukaHammerEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID


def make_direct(seed: int = 4) -> KukaHammerEnvV3:
    env = KukaHammerEnvV3(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move_tcp(env, target: np.ndarray, steps: int, gripper: float) -> None:
    for _ in range(steps):
        action = np.r_[
            np.clip((target - env.tcp_center) / env.action_scale, -1, 1), gripper
        ].astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated and not truncated and "success" in info


def move_hammer_head(env, target: np.ndarray, steps: int) -> None:
    start = env.data.geom("HammerHead").xpos.copy()
    tcp_from_head = env.tcp_center - start
    for waypoint in np.linspace(start, target, steps):
        move_tcp(env, waypoint + tcp_from_head, 1, 1.0)


def test_direct_resets_grasp_lift_and_nail_motion() -> None:
    env = make_direct()
    try:
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (39,) and np.isfinite(obs).all() and info == {}
            assert env.action_space.shape == (4,)
            np.testing.assert_allclose(obs[:18], obs[18:36])
            np.testing.assert_allclose(obs[36:39], env._target_pos)
            assert env._last_rand_vec.shape == (3,)

        env.seed(4)
        env.reset()
        hammer_start = env.get_body_com("hammer").copy()
        # Grasp near the head so the long tool clears the table instead of
        # pivoting around a handle-only grasp.
        grip = hammer_start + np.array([0.13, 0.0, 0.01])
        move_tcp(env, grip + np.array([0.0, 0.0, 0.08]), 20, -1.0)
        move_tcp(env, grip, 40, -1.0)
        move_tcp(env, grip, 50, 1.0)
        for dz in np.linspace(0.0, 0.08, 100):
            move_tcp(env, grip + np.array([0.0, 0.0, dz]), 1, 1.0)
        assert env.get_body_com("hammer")[2] > hammer_start[2] + 0.07

        nail_start = env._get_site_pos("nailHead").copy()
        move_hammer_head(env, nail_start + np.array([0.0, -0.05, 0.04]), 180)
        assert env.touching_main_object
        move_hammer_head(env, nail_start + np.array([0.0, 0.08, 0.0]), 80)
        assert float(env.data.joint("NailSlideJoint").qpos[0]) > 0.09
    finally:
        env.close()


def test_registered_horizon_and_rgb_render() -> None:
    assert "kuka-hammer-v3" not in ALL_V3_ENVIRONMENTS
    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-hammer-v3",
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
        env_name="kuka-hammer-v3",
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


def test_hammer_reward_ablation_math_and_unchanged_success_threshold() -> None:
    original = object.__new__(KukaHammerEnvV3)
    original.hammer_reward_variant = "original"
    original.hammer_nail_progress_weight = 0.0
    assert original._shape_nail_progress_reward(3.25, 0.08) == 3.25

    assert KukaHammerEnvV3._nail_progress(0.0) == 0.0
    assert KukaHammerEnvV3._nail_progress(0.045) == 0.5
    assert KukaHammerEnvV3._nail_progress(0.10) == 1.0

    progress = object.__new__(KukaHammerEnvV3)
    progress.hammer_reward_variant = "nail_progress"
    progress.hammer_nail_progress_weight = 2.0
    assert progress._shape_nail_progress_reward(3.25, 0.045) == 4.25
    progress.hammer_nail_progress_weight = 4.0
    assert progress._shape_nail_progress_reward(3.25, 0.045) == 5.25

    assert not KukaHammerEnvV3._is_success(0.09)
    assert KukaHammerEnvV3._is_success(np.nextafter(0.09, 1.0))
