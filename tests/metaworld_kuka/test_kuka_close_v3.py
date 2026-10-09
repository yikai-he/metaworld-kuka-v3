"""Contracts and action-based physics sanity for KUKA-v3 close tasks."""

from __future__ import annotations

import inspect

import gymnasium as gym
import numpy as np
import pytest

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld_kuka.envs import KukaFaucetCloseEnvV3, KukaWindowCloseEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID

TASKS = {
    "kuka-faucet-close-v3": KukaFaucetCloseEnvV3,
    "kuka-window-close-v3": KukaWindowCloseEnvV3,
}


def direct(cls, seed: int = 7):
    env = cls(reward_function_version="v2")
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move(env, target, grip: float, steps: int):
    last = None
    for _ in range(steps):
        action = np.r_[
            np.clip((target - env.tcp_center) / env.action_scale, -1, 1), grip
        ].astype(np.float32)
        last = env.step(action)
        assert len(last) == 5
        assert np.isfinite(last[0]).all() and np.isfinite(last[1])
        assert "success" in last[4]
    return last


@pytest.mark.parametrize("name,cls", TASKS.items())
def test_direct_reset_random_step_and_visible_goal(name, cls):
    env = direct(cls)
    try:
        random_vectors = []
        for _ in range(4):
            observation, info = env.reset()
            random_vectors.append(env._last_rand_vec.copy())
            assert observation.shape == (39,)
            assert env.action_space.shape == (4,)
            assert info == {}
            assert np.isfinite(observation).all()
            np.testing.assert_allclose(observation[36:39], env._target_pos)
            transition = env.step(env.action_space.sample())
            assert len(transition) == 5
            assert not transition[2]
            assert np.isfinite(transition[0]).all()
            assert np.isfinite(transition[1])
            assert "success" in transition[4]
        assert any(
            not np.array_equal(random_vectors[0], vector)
            for vector in random_vectors[1:]
        )
        assert name not in ALL_V3_ENVIRONMENTS
    finally:
        env.close()


@pytest.mark.parametrize("name", TASKS)
def test_registered_horizon(name):
    env = gym.make(
        KUKA_MT1_ID,
        env_name=name,
        seed=17,
        reward_function_version="v2",
        terminate_on_success=False,
        disable_env_checker=True,
    )
    try:
        observation, _ = env.reset()
        assert observation.shape == (39,)
        assert observation.dtype == np.float32
        assert env.action_space.shape == (4,)
        for step in range(1, 201):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert not terminated
            assert truncated is (step == 200)
            assert "success" in info
    finally:
        env.close()


@pytest.mark.parametrize("name", TASKS)
def test_registered_rgb(name):
    env = gym.make(
        KUKA_MT1_ID,
        env_name=name,
        seed=19,
        render_mode="rgb_array",
        camera_name="corner",
        disable_env_checker=True,
    )
    try:
        env.reset()
        image = env.render()
        assert image.shape == (480, 480, 3)
        assert image.dtype == np.uint8
    finally:
        env.close()


def test_faucet_contact_closes_toward_target():
    env = direct(KukaFaucetCloseEnvV3, 2)
    try:
        observation, _ = env.reset()
        start_joint = float(env.data.joint("knob_Joint_1").qpos[0])
        start_error = env.compute_reward(np.zeros(4), observation)[3]
        center = env.obj_init_pos + [0.0, 0.0, 0.124]
        handle = env._get_pos_objects().copy()
        move(env, handle + [0.04, -0.023, -0.02], 1, 50)
        for angle in np.linspace(0, -1.40, 30):
            target = center + [
                0.175 * np.sin(angle),
                -0.175 * np.cos(angle) - 0.023,
                -0.01,
            ]
            move(env, target, 1, 10)
        end_error = env.compute_reward(np.zeros(4), env._get_obs())[3]
        assert float(env.data.joint("knob_Joint_1").qpos[0]) < start_joint - 0.5
        assert end_error < env._target_radius
        assert end_error < start_error
    finally:
        env.close()


def test_window_contact_closes_toward_target():
    env = direct(KukaWindowCloseEnvV3, 2)
    try:
        observation, _ = env.reset()
        start_joint = float(env.data.joint("window_slide").qpos[0])
        start_error = env.compute_reward(np.zeros(4), observation)[3]
        handle = env._get_pos_objects().copy()
        move(env, handle + [0.03, 0.02, 0.0], 1, 40)
        move(env, handle + [-0.10, 0.02, 0.0], 1, 80)
        end_error = env.compute_reward(np.zeros(4), env._get_obs())[3]
        assert float(env.data.joint("window_slide").qpos[0]) < start_joint - 0.1
        assert end_error < env.TARGET_RADIUS
        assert end_error < start_error
    finally:
        env.close()


@pytest.mark.parametrize("cls", TASKS.values())
def test_native_mujoco_only(cls):
    source = inspect.getsource(cls)
    assert "import gym\n" not in source
    assert "mujoco_py" not in source
