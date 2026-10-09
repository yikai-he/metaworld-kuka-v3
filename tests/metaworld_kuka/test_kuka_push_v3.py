"""Direct and registered integration tests for KUKA Push V3."""

from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np

import metaworld_kuka  # noqa: F401
from metaworld import ALL_V3_ENVIRONMENTS
from metaworld.wrappers import RandomTaskSelectWrapper
from metaworld_kuka.envs import KukaPushEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID


INFO_KEYS = {
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
}
V3_INFO_KEYS = {
    "reward_v3_caging",
    "reward_v3_contact",
    "reward_v3_signed_progress",
    "reward_v3_clipped_progress",
    "reward_v3_success_latched",
    "reward_v3_first_success",
    "reward_v3_total",
}


def make_direct(
    *,
    seed: int = 7,
    render_mode: str | None = None,
    reward_function_version: str = "v2",
) -> KukaPushEnvV3:
    env = KukaPushEnvV3(
        render_mode=render_mode,
        camera_name="corner" if render_mode else None,
        reward_function_version=reward_function_version,
    )
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def find_wrapper(env, wrapper_type):
    current = env
    while hasattr(current, "env"):
        if isinstance(current, wrapper_type):
            return current
        current = current.env
    raise AssertionError(f"{wrapper_type.__name__} not found")


def test_direct_contract_random_resets_steps_reward_and_success() -> None:
    env = make_direct()
    try:
        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == (39,)
            assert info == {}
            assert np.isfinite(obs).all()
            assert env.action_space.shape == (4,)
            assert env.goal_space.contains(env._target_pos)
            np.testing.assert_allclose(obs[36:39], env._target_pos)

        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert obs.shape == (39,)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert not terminated
            assert not truncated
            assert set(info) == INFO_KEYS

        # The Sawyer-v3 success definition is achievable at the physical goal.
        env._set_obj_xyz(env._target_pos)
        obs = env._get_obs()
        reward, info = env.evaluate_state(obs, np.zeros(4, dtype=np.float32))
        assert info["success"] == 1.0
        assert info["obj_to_target"] <= env.TARGET_RADIUS
        assert reward == 10.0
    finally:
        env.close()


def test_sampled_puck_has_conservative_robotiq_approach_clearance() -> None:
    env = make_direct(seed=10_000)
    try:
        np.testing.assert_allclose(
            env._random_reset_space.low[:3], [-0.10, 0.484, 0.02]
        )
        for _ in range(50):
            env.reset()
            direction = env._target_pos[:2] - env.obj_init_pos[:2]
            direction /= np.linalg.norm(direction)
            approach = env.obj_init_pos[:2] - 0.054 * direction
            assert approach[1] >= env.mocap_low[1] + 0.010 - 1e-12
    finally:
        env.close()


def test_calibrated_robotiq_push_establishes_contact_and_progress() -> None:
    env = make_direct(seed=7)
    try:
        env.reset()
        initial = env.get_body_com("obj").copy()
        initial_distance = np.linalg.norm(initial - env._target_pos)
        direction = env._target_pos[:2] - initial[:2]
        direction /= np.linalg.norm(direction)
        behind = initial.copy()
        behind[:2] -= 0.04 * direction
        behind[2] = 0.047
        pad_contact = False
        table_contact = False
        for target, steps in (
            (behind + [0, 0, 0.10], 25),
            (behind, 40),
            *((behind + np.r_[direction * d, 0.0], 1) for d in np.linspace(0, 0.25, 150)),
        ):
            for _ in range(steps):
                action = np.r_[
                    np.clip((target - env.tcp_center) / env.action_scale, -1, 1),
                    1.0,
                ].astype(np.float32)
                env.step(action)
            for contact in env.data.contact:
                names = {
                    env.model.geom(contact.geom1).name,
                    env.model.geom(contact.geom2).name,
                }
                pad_contact |= "objGeom" in names and bool(
                    names & set(env.ROBOTIQ_PAD_GEOMS)
                )
                table_contact |= "floor" in names and bool(
                    names & set(env.ROBOTIQ_PAD_GEOMS)
                )
        final = env.get_body_com("obj").copy()
        assert pad_contact and not table_contact
        assert np.linalg.norm(final - initial) > 0.10
        assert np.linalg.norm(final - env._target_pos) < initial_distance - 0.08
    finally:
        env.close()


def test_contact_gated_reward_is_finite_monotonic_and_not_hover_exploitable(
    monkeypatch,
) -> None:
    env = make_direct(seed=7)
    try:
        obs, _ = env.reset()
        action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # Hold caging constant to isolate the gate and in-place progress term.
        monkeypatch.setattr(env, "_gripper_caging_reward", lambda *args, **kwargs: 0.4)

        direction = env.obj_init_pos - env._target_pos
        direction /= np.linalg.norm(direction)
        far_obs = obs.copy()
        far_obs[4:7] = env._target_pos + direction * 0.12
        near_obs = obs.copy()
        near_obs[4:7] = env._target_pos + direction * 0.06

        monkeypatch.setattr(env, "pad_contacting_object", lambda _: False)
        hover_reward, *_ = env.compute_reward(action, near_obs)
        # No-contact proximity receives only the unchanged caging term.
        np.testing.assert_allclose(hover_reward, 0.8)

        monkeypatch.setattr(env, "pad_contacting_object", lambda _: True)
        far_reward = env.compute_reward(action, far_obs)[0]
        near_reward = env.compute_reward(action, near_obs)[0]
        repeat_reward = env.compute_reward(action, near_obs)[0]
        assert np.isfinite([far_reward, near_reward, repeat_reward]).all()
        assert 0.0 <= far_reward < near_reward < 10.0
        np.testing.assert_allclose(near_reward, repeat_reward)

        success_obs = obs.copy()
        success_obs[4:7] = env._target_pos
        assert env.compute_reward(action, success_obs)[0] == 10.0
    finally:
        env.close()


def _v3_reward_fixture(monkeypatch, *, contact: bool, caging: float = 0.4):
    env = make_direct(reward_function_version="v3")
    obs, _ = env.reset()
    monkeypatch.setattr(
        env, "_gripper_caging_reward", lambda *args, **kwargs: caging
    )
    monkeypatch.setattr(env, "pad_contacting_object", lambda _: contact)
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return env, obs, action


def _v3_1_reward_fixture(monkeypatch, *, contact: bool, caging: float = 0.4):
    env = make_direct(reward_function_version="v3_1")
    obs, _ = env.reset()
    monkeypatch.setattr(
        env, "_gripper_caging_reward", lambda *args, **kwargs: caging
    )
    monkeypatch.setattr(env, "pad_contacting_object", lambda _: contact)
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return env, obs, action


def _set_v3_progress(env, obs, progress: float):
    result = obs.copy()
    goal = env._target_pos.copy()
    current_distance = 0.10
    result[36:39] = goal
    result[4:7] = goal + np.array([current_distance, 0.0, 0.0])
    result[22:25] = goal + np.array([current_distance + progress, 0.0, 0.0])
    result[3] = 1.0
    return result


def test_reward_v3_stationary_no_contact_is_small_and_finite(monkeypatch) -> None:
    env, obs, action = _v3_reward_fixture(monkeypatch, contact=False)
    try:
        reward = env.compute_reward(action, _set_v3_progress(env, obs, 0.0))[0]
        assert np.isfinite(reward)
        np.testing.assert_allclose(reward, 0.04)
    finally:
        env.close()


def test_reward_v3_stationary_contact_only_gets_baseline(monkeypatch) -> None:
    env, obs, action = _v3_reward_fixture(monkeypatch, contact=True)
    try:
        reward = env.compute_reward(action, _set_v3_progress(env, obs, 0.0))[0]
        np.testing.assert_allclose(reward, 0.09)
    finally:
        env.close()


def test_reward_v3_signed_progress_and_clipping(monkeypatch) -> None:
    env, obs, action = _v3_reward_fixture(monkeypatch, contact=True)
    try:
        positive = env.compute_reward(
            action, _set_v3_progress(env, obs, 0.002)
        )[0]
        negative = env.compute_reward(
            action, _set_v3_progress(env, obs, -0.002)
        )[0]
        clipped_positive = env.compute_reward(
            action, _set_v3_progress(env, obs, 0.020)
        )[0]
        clipped_negative = env.compute_reward(
            action, _set_v3_progress(env, obs, -0.020)
        )[0]
        np.testing.assert_allclose(positive, 2.09)
        np.testing.assert_allclose(negative, -1.91)
        np.testing.assert_allclose(clipped_positive, 5.09)
        np.testing.assert_allclose(clipped_negative, -4.91)
        assert positive > 0.09 > negative
    finally:
        env.close()


def test_reward_v3_first_success_post_success_and_reset(monkeypatch) -> None:
    env, obs, action = _v3_reward_fixture(monkeypatch, contact=True)
    try:
        success_obs = obs.copy()
        success_obs[4:7] = env._target_pos
        success_obs[22:25] = env._target_pos + np.array([0.01, 0.0, 0.0])
        success_obs[36:39] = env._target_pos
        assert env.compute_reward(action, success_obs)[0] == 100.0
        assert env._reward_v3_success_latched
        assert env.compute_reward(action, success_obs)[0] == 0.0
        reset_obs, _ = env.reset()
        assert not env._reward_v3_success_latched
        reset_success_obs = reset_obs.copy()
        reset_success_obs[4:7] = env._target_pos
        reset_success_obs[22:25] = env._target_pos + np.array([0.01, 0.0, 0.0])
        reset_success_obs[36:39] = env._target_pos
        assert env.compute_reward(action, reset_success_obs)[0] == 100.0
    finally:
        env.close()


def test_reward_v3_deterministic_finite_and_diagnostic_components(monkeypatch) -> None:
    env, obs, action = _v3_reward_fixture(monkeypatch, contact=True)
    try:
        progress_obs = _set_v3_progress(env, obs, 0.003)
        first, *_ = env.compute_reward(action, progress_obs)
        second, *_ = env.compute_reward(action, progress_obs)
        np.testing.assert_allclose(first, second)
        assert np.isfinite(first)
        reward, info = env.evaluate_state(progress_obs, action)
        assert reward == first
        assert V3_INFO_KEYS <= set(info)
        assert info["reward_v3_contact"] is True
        np.testing.assert_allclose(info["reward_v3_signed_progress"], 0.003)
        np.testing.assert_allclose(info["reward_v3_clipped_progress"], 0.003)
        assert info["reward_v3_success_latched"] is False
        np.testing.assert_allclose(info["reward_v3_total"], reward)
    finally:
        env.close()


def test_reward_v3_1_no_contact_equals_caging_and_increases(monkeypatch) -> None:
    env, obs, action = _v3_1_reward_fixture(
        monkeypatch, contact=False, caging=0.2
    )
    try:
        stationary = _set_v3_progress(env, obs, 0.0)
        low = env.compute_reward(action, stationary)[0]
        monkeypatch.setattr(
            env, "_gripper_caging_reward", lambda *args, **kwargs: 0.7
        )
        high = env.compute_reward(action, stationary)[0]
        np.testing.assert_allclose([low, high], [0.2, 0.7])
        assert high > low
    finally:
        env.close()


def test_reward_v3_1_contact_phase_excludes_caging(monkeypatch) -> None:
    env, obs, action = _v3_1_reward_fixture(
        monkeypatch, contact=True, caging=0.9
    )
    try:
        stationary = env.compute_reward(
            action, _set_v3_progress(env, obs, 0.0)
        )[0]
        positive = env.compute_reward(
            action, _set_v3_progress(env, obs, 0.001)
        )[0]
        negative = env.compute_reward(
            action, _set_v3_progress(env, obs, -0.001)
        )[0]
        clipped_positive = env.compute_reward(
            action, _set_v3_progress(env, obs, 0.020)
        )[0]
        clipped_negative = env.compute_reward(
            action, _set_v3_progress(env, obs, -0.020)
        )[0]
        assert np.isfinite(
            [stationary, positive, negative, clipped_positive, clipped_negative]
        ).all()
        np.testing.assert_allclose(
            [stationary, positive, negative, clipped_positive, clipped_negative],
            [0.05, 1.05, -0.95, 5.05, -4.95],
        )
    finally:
        env.close()


def test_reward_v3_1_first_success_post_success_and_reset(monkeypatch) -> None:
    env, obs, action = _v3_1_reward_fixture(monkeypatch, contact=True)
    try:
        success_obs = obs.copy()
        success_obs[4:7] = env._target_pos
        success_obs[22:25] = env._target_pos + np.array([0.01, 0.0, 0.0])
        success_obs[36:39] = env._target_pos
        assert env.compute_reward(action, success_obs)[0] == 100.0
        assert env._reward_v3_1_success_latched
        assert env.compute_reward(action, success_obs)[0] == 0.0
        reset_obs, _ = env.reset()
        assert not env._reward_v3_1_success_latched
        reset_success_obs = reset_obs.copy()
        reset_success_obs[4:7] = env._target_pos
        reset_success_obs[22:25] = env._target_pos + np.array([0.01, 0.0, 0.0])
        reset_success_obs[36:39] = env._target_pos
        assert env.compute_reward(action, reset_success_obs)[0] == 100.0
    finally:
        env.close()


def test_reward_v3_1_diagnostics_and_contact_predicate(monkeypatch) -> None:
    env, obs, action = _v3_1_reward_fixture(monkeypatch, contact=True)
    try:
        progress_obs = _set_v3_progress(env, obs, 0.001)
        reward, info = env.evaluate_state(progress_obs, action)
        np.testing.assert_allclose(reward, 1.05)
        assert info["reward_v3_1_contact"] is True
        np.testing.assert_allclose(info["reward_v3_1_caging"], 0.4)
        np.testing.assert_allclose(info["reward_v3_1_signed_progress"], 0.001)
        np.testing.assert_allclose(info["reward_v3_1_clipped_progress"], 0.001)
        np.testing.assert_allclose(info["reward_v3_1_total"], reward)
        # Closing the aperture gate disables the same physical-contact predicate.
        progress_obs[3] = 0.0
        np.testing.assert_allclose(env.compute_reward(action, progress_obs)[0], 0.4)
    finally:
        env.close()


def test_reward_v2_formula_is_unchanged(monkeypatch) -> None:
    env = make_direct(reward_function_version="v2")
    try:
        obs, _ = env.reset()
        obs[3] = 1.0
        obs[4:7] = env._target_pos + np.array([0.10, 0.0, 0.0])
        action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        monkeypatch.setattr(
            env, "_gripper_caging_reward", lambda *args, **kwargs: 0.4
        )
        monkeypatch.setattr(env, "pad_contacting_object", lambda _: True)
        reward, *_, in_place = env.compute_reward(action, obs)
        np.testing.assert_allclose(reward, 1.0 + 4.0 * 0.4 + 5.0 * in_place)
    finally:
        env.close()


def test_real_pad_contact_activates_push_bonus() -> None:
    env = make_direct(seed=7)
    try:
        env.reset()
        initial = env.get_body_com("obj").copy()
        direction = env._target_pos[:2] - initial[:2]
        direction /= np.linalg.norm(direction)
        behind = initial.copy()
        behind[:2] -= 0.04 * direction
        behind[2] = 0.047
        found = False
        for target, steps in (
            (behind + [0, 0, 0.10], 25),
            (behind, 40),
            *((behind + np.r_[direction * d, 0.0], 1) for d in np.linspace(0, 0.10, 80)),
        ):
            for _ in range(steps):
                action = np.r_[
                    np.clip((target - env.tcp_center) / env.action_scale, -1, 1),
                    1.0,
                ].astype(np.float32)
                obs, reward, *_ = env.step(action)
                if env.pad_contacting_object(env._get_id_main_object()):
                    components = env.compute_reward(action, obs)
                    base_reward = 2.0 * components[4]
                    assert np.isfinite(reward)
                    assert reward > base_reward
                    assert components[1] > 0.02
                    found = True
                    break
            if found:
                break
        assert found
    finally:
        env.close()


def test_registered_tasks_horizon_and_rendering() -> None:
    assert "kuka-push-v3" not in ALL_V3_ENVIRONMENTS
    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-push-v3",
        seed=17,
        reward_function_version="v2",
        disable_env_checker=True,
    )
    try:
        task_wrapper = find_wrapper(env, RandomTaskSelectWrapper)
        assert len(task_wrapper.tasks) == 50
        for task in task_wrapper.tasks:
            data = pickle.loads(task.data)
            assert data["rand_vec"].shape == (6,)
            assert np.isfinite(data["rand_vec"]).all()

        obs, _ = env.reset()
        assert obs.shape == (39,)
        assert obs.dtype == np.float32
        assert env.action_space.shape == (4,)
        for step in range(1, 201):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert not terminated
            assert truncated is (step == 200)
            assert "success" in info
    finally:
        env.close()

    render_env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-push-v3",
        seed=19,
        reward_function_version="v2",
        render_mode="rgb_array",
        camera_name="corner",
        disable_env_checker=True,
    )
    try:
        render_env.reset()
        assert render_env.render().shape == (480, 480, 3)
    finally:
        render_env.close()
