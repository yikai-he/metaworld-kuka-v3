"""Interface and task-physics coverage for the two added KUKA V3 tasks."""

import gymnasium as gym
import numpy as np
import pytest

import metaworld_kuka  # noqa: F401
from metaworld_kuka.envs import KukaButtonPressTopdownEnvV3, KukaPegUnplugSideEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID

TASKS = ("kuka-button-press-topdown-v3", "kuka-peg-unplug-side-v3")


def direct(cls, seed=3, **kwargs):
    env = cls(reward_function_version="v2", **kwargs)
    env.seed(seed)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def move(env, target, grip, steps):
    result = None
    for _ in range(steps):
        action = np.r_[np.clip((target - env.tcp_center) / env.action_scale, -1, 1), grip].astype(np.float32)
        result = env.step(action)
        assert np.isfinite(result[0]).all() and np.isfinite(result[1])
    return result


@pytest.mark.parametrize("name", TASKS)
def test_registered_interface_determinism_randomization_and_render(name):
    env = gym.make(KUKA_MT1_ID, env_name=name, seed=17,
                   reward_function_version="v2", render_mode="rgb_array",
                   camera_name="corner", disable_env_checker=True)
    try:
        wrapper = env
        while not hasattr(wrapper, "tasks"):
            wrapper = wrapper.env
        wrapper.toggle_sample_tasks_on_reset(False)
        task = wrapper.tasks[0]
        env.unwrapped.set_task(task)
        first, _ = env.reset()
        env.unwrapped.set_task(task)
        second, _ = env.reset()
        np.testing.assert_allclose(first, second)
        assert first.shape == (39,) and first.dtype == np.float32
        assert env.action_space.shape == (4,)
        assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
        assert env.render().shape == (480, 480, 3)
        assert len({bytes(task.data) for task in wrapper.tasks}) > 1
        transition = env.step(np.zeros(4, dtype=np.float32))
        assert np.isfinite(transition[0]).all() and np.isfinite(transition[1])
    finally:
        env.close()


def test_topdown_button_moves_down_reward_rises_and_success_is_physical():
    env = direct(KukaButtonPressTopdownEnvV3)
    try:
        obs, _ = env.reset()
        start = env._get_pos_objects().copy()
        initial_reward = env.compute_reward(np.zeros(4), obs)[0]
        move(env, start + [0, 0, 0.06], 1, 50)
        _, reward, _, _, info = move(env, env._target_pos + [0, 0, 0.005], 1, 70)
        assert env._get_pos_objects()[2] < start[2] - 0.05
        assert reward > initial_reward and info["success"] == 1.0
    finally:
        env.close()


def test_side_peg_starts_inserted_and_scripted_grasp_extracts_it():
    env = direct(KukaPegUnplugSideEnvV3)
    try:
        obs, _ = env.reset()
        start = env._get_pos_objects().copy()
        initial_reward = env.compute_reward(np.zeros(4), obs)[0]
        hole = env._get_site_pos("hole")
        assert abs(start[0] - hole[0]) < 0.08
        grip = start + [0.02, 0, 0]
        move(env, grip + [0, 0, 0.10], -1, 50)
        move(env, grip, -1, 60)
        move(env, grip, 1, 80)
        best = (initial_reward, 0.0, start[0])
        for _ in range(60):
            _, reward, _, _, info = move(env, grip + [0.10, 0, 0.01], 1, 1)
            best = max(best, (reward, info["success"], env._get_pos_objects()[0]))
        assert best[2] > start[0] + 0.04
        assert best[0] > initial_reward and best[1] == 1.0
    finally:
        env.close()


def test_side_peg_progress_potential_and_shaping_math():
    env = object.__new__(KukaPegUnplugSideEnvV3)
    env.obj_init_pos = np.array([-0.19, 0.60, 0.131])
    env._target_pos = np.array([-0.04, 0.60, 0.131])
    env.peg_unplug_reward_variant = "peg_unplug_progress_v1"
    env.peg_unplug_progress_weight = 2.0
    env.peg_unplug_progress_gamma = 0.99

    assert env._extraction_progress_potential(env.obj_init_pos) == 0.0
    assert env._extraction_progress_potential(env.obj_init_pos + [0.075, 0, 0]) == 0.5
    assert env._extraction_progress_potential(env.obj_init_pos + [1.0, 0, 0]) == 1.0
    assert env._extraction_progress_potential(env.obj_init_pos - [1.0, 0, 0]) == 0.0
    assert env._progress_shaping_reward(0.2, 0.4) == pytest.approx(0.392)
    assert env._progress_shaping_reward(0.4, 0.2) < 0.0
    assert env._progress_shaping_reward(0.4, 0.4) == pytest.approx(-0.008)

    env.peg_unplug_reward_variant = "original"
    assert env._progress_shaping_reward(0.2, 0.9) == 0.0


def test_side_peg_progress_reward_uses_consecutive_states_once(monkeypatch):
    env = direct(
        KukaPegUnplugSideEnvV3,
        peg_unplug_reward_variant="peg_unplug_progress_v1",
        peg_unplug_progress_weight=2.0,
        peg_unplug_progress_gamma=0.99,
    )
    try:
        obs, _ = env.reset()
        assert env._previous_extraction_potential == 0.0
        original = env.compute_reward(np.zeros(4, dtype=np.float32), obs)[0]
        monkeypatch.setattr(
            env,
            "compute_reward",
            lambda action, observation: (
                original, 0.1, 1.0, 0.1, 0.2, 0.3, 0.0
            ),
        )
        moved = obs.copy()
        moved[4] = env.obj_init_pos[0] + 0.075
        potential = env._extraction_progress_potential(moved[4:7])
        reward, info = env.evaluate_state(moved, np.zeros(4, dtype=np.float32))
        first_shaping = 2.0 * 0.99 * potential
        assert reward == pytest.approx(original + first_shaping)
        assert info["progress_shaping_reward"] == pytest.approx(first_shaping)
        reward, info = env.evaluate_state(moved, np.zeros(4, dtype=np.float32))
        stationary_shaping = 2.0 * (0.99 - 1.0) * potential
        assert reward == pytest.approx(original + stationary_shaping)
        assert info["progress_shaping_reward"] == pytest.approx(stationary_shaping)

        env.reset()
        assert env._previous_extraction_potential == 0.0
    finally:
        env.close()


def test_side_peg_original_reward_is_default_and_unchanged():
    env = direct(KukaPegUnplugSideEnvV3)
    try:
        obs, _ = env.reset()
        action = np.zeros(4, dtype=np.float32)
        original = env.compute_reward(action, obs)[0]
        reward, info = env.evaluate_state(obs, action)
        assert env.peg_unplug_reward_variant == "original"
        assert info["progress_shaping_reward"] == 0.0
        assert reward == original
    finally:
        env.close()


def test_side_peg_progress_variant_preserves_contract_and_success_threshold():
    env = gym.make(
        KUKA_MT1_ID,
        env_name="kuka-peg-unplug-side-v3",
        seed=23,
        reward_function_version="v2",
        peg_unplug_reward_variant="peg_unplug_progress_v1",
        peg_unplug_progress_weight=2.0,
        peg_unplug_progress_gamma=0.99,
        disable_env_checker=True,
    )
    try:
        obs, _ = env.reset()
        assert obs.shape == (39,) and env.action_space.shape == (4,)
        raw = env.unwrapped
        action = np.zeros(4, dtype=np.float32)
        success_obs = obs.astype(np.float64)
        success_obs[4:7] = raw._target_pos + np.array([0.069, 0.0, 0.0])
        assert raw.evaluate_state(success_obs, action)[1]["success"] == 1.0
        failure_obs = success_obs.copy()
        failure_obs[4] = raw._target_pos[0] + 0.071
        assert raw.evaluate_state(failure_obs, action)[1]["success"] == 0.0
    finally:
        env.close()
