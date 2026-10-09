"""Integration tests for the isolated KUKA MT1 Gymnasium entry point."""

from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np

import metaworld_kuka
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld.wrappers import AutoTerminateOnSuccessWrapper, RandomTaskSelectWrapper
from metaworld_kuka.registration import Float32ObservationWrapper, KUKA_MT1_ID


def make_registered_env(**kwargs):
    return gym.make(
        KUKA_MT1_ID,
        env_name="kuka-reach-v3",
        seed=17,
        use_one_hot=False,
        terminate_on_success=False,
        reward_function_version="v2",
        disable_env_checker=True,
        **kwargs,
    )


def find_wrapper(env, wrapper_type):
    current = env
    while True:
        if isinstance(current, wrapper_type):
            return current
        if not hasattr(current, "env"):
            raise AssertionError(f"{wrapper_type.__name__} not found")
        current = current.env


def test_public_mt1_lifecycle_contract_and_horizon() -> None:
    assert KUKA_MT1_ID in gym.envs.registry
    assert "kuka-reach-v3" not in ALL_V3_ENVIRONMENTS
    env = make_registered_env()
    try:
        assert isinstance(find_wrapper(env, Float32ObservationWrapper), Float32ObservationWrapper)
        task_wrapper = find_wrapper(env, RandomTaskSelectWrapper)
        termination_wrapper = find_wrapper(env, AutoTerminateOnSuccessWrapper)
        assert termination_wrapper.terminate_on_success is False
        assert len(task_wrapper.tasks) == 50

        for task in task_wrapper.tasks:
            data = pickle.loads(task.data)
            rand_vec = data["rand_vec"]
            assert rand_vec.shape == (6,)
            assert np.isfinite(rand_vec).all()
            assert data["partially_observable"] is False

        obs, info = env.reset()
        assert obs.shape == (39,)
        assert obs.dtype == np.float32
        assert info == {}
        assert env.observation_space.shape == (39,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.shape == (4,)
        assert np.any(obs[36:39])
        assert env.unwrapped.reward_function_version == "v2"

        transition = env.step(np.zeros(4, dtype=np.float32))
        assert len(transition) == 5
        assert transition[0].dtype == np.float32
        assert "success" in transition[4]
        assert transition[2] is False

        # Reusing one official Task through public set_task() reproduces reset
        # state without touching any lifecycle flags.
        task_wrapper.toggle_sample_tasks_on_reset(False)
        task = task_wrapper.tasks[0]
        env.unwrapped.set_task(task)
        first_obs, _ = env.reset()
        env.unwrapped.set_task(task)
        second_obs, _ = env.reset()
        np.testing.assert_allclose(first_obs, second_obs)
        np.testing.assert_allclose(first_obs[36:39], second_obs[36:39])

        env.unwrapped.set_task(task)
        env.reset()
        for step in range(1, 201):
            _, _, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            assert terminated is False
            assert "success" in info
            assert truncated is (step == 200)
    finally:
        env.close()


def test_registered_rgb_array_rendering() -> None:
    env = make_registered_env(render_mode="rgb_array", camera_name="corner")
    try:
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        frame = env.render()
        assert frame.shape == (480, 480, 3)
    finally:
        env.close()
