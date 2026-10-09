"""Contract, registration, rendering, and contact-physics tests for Phase 2."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld_kuka.envs import (
    KukaDoorOpenEnvV3, KukaDrawerCloseEnvV3, KukaFaucetOpenEnvV3,
    KukaPickPlaceEnvV3, KukaWindowOpenEnvV3,
)
from metaworld_kuka.registration import KUKA_MT1_ID

TASKS = {
    "kuka-drawer-close-v3": KukaDrawerCloseEnvV3,
    "kuka-faucet-open-v3": KukaFaucetOpenEnvV3,
    "kuka-window-open-v3": KukaWindowOpenEnvV3,
    "kuka-door-open-v3": KukaDoorOpenEnvV3,
    "kuka-pick-place-v3": KukaPickPlaceEnvV3,
}


def direct(cls, seed=7):
    env = cls(reward_function_version="v2")
    env.seed(seed); env.seeded_rand_vec = True; env._freeze_rand_vec = False
    env._set_task_called = True; env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move(env, target, grip, steps):
    last = None
    for _ in range(steps):
        action = np.r_[np.clip((target - env.tcp_center) / env.action_scale, -1, 1), grip].astype(np.float32)
        last = env.step(action)
        assert np.isfinite(last[0]).all() and np.isfinite(last[1]) and "success" in last[4]
    return last


@pytest.mark.parametrize("name,cls", TASKS.items())
def test_direct_reset_random_step_and_visible_goal(name, cls):
    env = direct(cls)
    try:
        samples = []
        for _ in range(4):
            obs, info = env.reset(); samples.append(env._last_rand_vec.copy())
            assert obs.shape == (39,) and env.action_space.shape == (4,) and info == {}
            assert np.isfinite(obs).all(); np.testing.assert_allclose(obs[36:39], env._target_pos)
            out = env.step(env.action_space.sample())
            assert len(out) == 5 and not out[2] and "success" in out[4]
        assert any(not np.array_equal(samples[0], sample) for sample in samples[1:])
        assert name not in ALL_V3_ENVIRONMENTS
    finally:
        env.close()


@pytest.mark.parametrize("name", TASKS)
def test_registered_horizon(name):
    env = gym.make(KUKA_MT1_ID, env_name=name, seed=17, reward_function_version="v2", disable_env_checker=True)
    try:
        obs, _ = env.reset(); assert obs.shape == (39,) and obs.dtype == np.float32
        for step in range(1, 201):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward) and not terminated
            assert truncated is (step == 200) and "success" in info
    finally: env.close()


@pytest.mark.parametrize("name", TASKS)
def test_registered_rgb(name):
    env = gym.make(KUKA_MT1_ID, env_name=name, seed=19, render_mode="rgb_array", camera_name="corner", disable_env_checker=True)
    try:
        env.reset(); assert env.render().shape == (480, 480, 3)
    finally: env.close()


def test_drawer_contact_closes_joint():
    env = direct(KukaDrawerCloseEnvV3, 2)
    try:
        env.reset(); start = float(env.data.joint("goal_slidey").qpos[0]); handle = env._get_pos_objects().copy()
        move(env, handle + [0, -.03, -.03], 1, 40); move(env, handle + [0, .25, -.03], 1, 100)
        assert float(env.data.joint("goal_slidey").qpos[0]) > start + .10
    finally: env.close()


def test_faucet_contact_rotates_joint():
    env = direct(KukaFaucetOpenEnvV3, 2)
    try:
        env.reset(); joint = env.data.joint("knob_Joint_1")
        path = []
        for qpos in np.linspace(0.0, 1.5, 30):
            joint.qpos[0] = qpos; mujoco.mj_forward(env.model, env.data)
            path.append(env._get_pos_objects().copy())
        joint.qpos[0] = 0.0; joint.qvel[0] = 0.0; mujoco.mj_forward(env.model, env.data)
        move(env, path[0] + [0, 0, .06], -1, 20); move(env, path[0], -1, 30)
        for target in path:
            move(env, target, -1, 5)
        _, info = env.evaluate_state(env._get_obs(), np.zeros(4, dtype=np.float32))
        assert float(env.data.joint("knob_Joint_1").qpos[0]) > 0.5
        assert info["success"] == 1.0
    finally: env.close()


def test_window_contact_opens_slide():
    env = direct(KukaWindowOpenEnvV3, 2)
    try:
        env.reset(); handle = env._get_pos_objects().copy(); move(env, handle + [-.03, .02, 0], 1, 40)
        move(env, handle + [.25, .02, 0], 1, 120)
        assert float(env.data.joint("window_slide").qpos[0]) > .12
    finally: env.close()


def test_door_contact_opens_hinge():
    env = direct(KukaDoorOpenEnvV3, 2)
    try:
        env.reset(); joint = env.data.joint("doorjoint"); path = []
        for qpos in np.linspace(0.0, -1.4, 35):
            joint.qpos[0] = qpos; mujoco.mj_forward(env.model, env.data)
            path.append(env._get_pos_objects().copy())
        joint.qpos[0] = 0.0; joint.qvel[0] = 0.0; mujoco.mj_forward(env.model, env.data)
        offset = np.array([-0.04, 0.08, 0.0])
        move(env, path[0] + offset + [0, 0, .06], -1, 20)
        move(env, path[0] + offset, -1, 30)
        for target in path:
            move(env, target + offset, -1, 5)
        _, info = env.evaluate_state(env._get_obs(), np.zeros(4, dtype=np.float32))
        assert float(env.data.joint("doorjoint").qpos[0]) < -1.0
        assert info["success"] == 1.0
    finally: env.close()


def test_pick_place_approach_grasp_lift_transport():
    env = direct(KukaPickPlaceEnvV3, 2)
    try:
        env.reset(); obj0 = env._get_pos_objects().copy(); move(env, obj0 + [0, 0, .06], -1, 30)
        move(env, obj0 + [0, 0, .015], -1, 30); move(env, obj0 + [0, 0, .015], 1, 40)
        move(env, obj0 + [0, 0, .15], 1, 60); lifted = env._get_pos_objects().copy()
        assert lifted[2] > obj0[2] + .04
        move(env, env._target_pos, 1, 70)
        assert np.linalg.norm(env._get_pos_objects() - env._target_pos) < np.linalg.norm(lifted - env._target_pos)
    finally: env.close()
