"""Direct semantic tests for KUKA Reach V3 before registry integration."""

from __future__ import annotations

import numpy as np

from metaworld.utils import reward_utils
from metaworld_kuka.envs import KukaReachEnvV3
from metaworld_kuka.kuka_xyz_env import KukaXYZEnv


INFO_KEYS = {
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
}


def make_env(*, seed: int = 7, render_mode: str | None = None) -> KukaReachEnvV3:
    """Configure direct construction like an observable, freely sampled MT task."""
    env = KukaReachEnvV3(
        render_mode=render_mode,
        camera_name="corner" if render_mode is not None else None,
        reward_function_version="v2",
    )
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def test_construction_reset_observation_and_sampling() -> None:
    assert issubclass(KukaReachEnvV3, KukaXYZEnv)
    env = make_env()
    try:
        assert env.action_space.shape == (4,)
        assert env.observation_space.shape == (39,)

        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == (39,)
            assert info == {}
            assert np.isfinite(obs).all()
            np.testing.assert_allclose(obs[:18], obs[18:36])
            np.testing.assert_allclose(obs[36:39], env._target_pos)
            assert env.goal_space.contains(env._target_pos)

            tcp = env.tcp_center
            assert tcp.shape == (3,)
            assert np.isfinite(tcp).all()
            assert np.linalg.norm(tcp) < 2.0

            # Mocap drives the hand body while reach is scored at TCP center.
            # Account for their fixed-orientation offset when checking whether
            # each sampled TCP goal is inside the configured KUKA workspace.
            hand_to_tcp = env.get_endeff_pos() - tcp
            required_hand_position = env._target_pos + hand_to_tcp
            assert np.all(required_hand_position >= env.mocap_low - 1e-9)
            assert np.all(required_hand_position <= env.mocap_high + 1e-9)
    finally:
        env.close()


def test_v2_reward_success_and_info_match_metaworld_reach() -> None:
    env = make_env()
    try:
        obs, _ = env.reset()
        action = np.zeros(4, dtype=np.float32)

        env._target_pos = env.tcp_center + np.array([0.051, 0.0, 0.0])
        reward, info = env.evaluate_state(obs, action)
        expected_distance = 0.051
        expected_margin = np.linalg.norm(env.hand_init_pos - env._target_pos)
        expected_in_place = reward_utils.tolerance(
            expected_distance,
            bounds=(0, 0.05),
            margin=expected_margin,
            sigmoid="long_tail",
        )
        assert set(info) == INFO_KEYS
        assert info["success"] == 0.0
        assert info["grasp_success"] == 1.0
        np.testing.assert_allclose(info["near_object"], expected_distance)
        np.testing.assert_allclose(info["grasp_reward"], expected_distance)
        np.testing.assert_allclose(info["obj_to_target"], expected_distance)
        np.testing.assert_allclose(info["in_place_reward"], expected_in_place)
        np.testing.assert_allclose(reward, 10 * expected_in_place)
        np.testing.assert_allclose(info["unscaled_reward"], reward)

        env._target_pos = env.tcp_center.copy()
        success_reward, success_info = env.evaluate_state(obs, action)
        assert success_info["success"] == 1.0
        np.testing.assert_allclose(success_reward, 10.0)
    finally:
        env.close()


def test_reward_increases_as_physical_tcp_moves_toward_goal() -> None:
    env = make_env(seed=13)
    try:
        obs, _ = env.reset()
        zero_action = np.zeros(4, dtype=np.float32)
        reward_before, distance_before, _ = env.compute_reward(zero_action, obs)

        for _ in range(25):
            mocap_delta = env._target_pos - env.data.mocap_pos[0]
            action = np.concatenate(
                (np.clip(mocap_delta / env.action_scale, -1.0, 1.0), [0.0])
            ).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        reward_after, distance_after, _ = env.compute_reward(zero_action, obs)
        assert distance_after < distance_before
        assert reward_after > reward_before
    finally:
        env.close()


def test_random_steps_repeated_resets_and_rendering() -> None:
    env = make_env(seed=23)
    try:
        for _ in range(3):
            obs, _ = env.reset()
            assert np.isfinite(obs).all()
            for _ in range(10):
                obs, _, _, _, info = env.step(env.action_space.sample())
                assert np.isfinite(obs).all()
                assert set(info) == INFO_KEYS
    finally:
        env.close()

    render_env = make_env(seed=29, render_mode="rgb_array")
    try:
        render_env.reset()
        frame = render_env.render()
        assert frame.shape == (480, 480, 3)
    finally:
        render_env.close()
