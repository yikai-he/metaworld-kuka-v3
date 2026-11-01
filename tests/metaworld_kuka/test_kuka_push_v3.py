"""Behavioral contract for the Sawyer-inspired KUKA cylinder Push task."""
import pickle
import gymnasium as gym
import numpy as np
import pytest
from metaworld.wrappers import RandomTaskSelectWrapper
import metaworld_kuka  # noqa: F401
from metaworld_kuka.envs import KukaPushEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID

def direct(seed=7):
    env = KukaPushEnvV3(); env.seed(seed); env.seeded_rand_vec = True
    env._freeze_rand_vec = False; env._set_task_called = True
    env._partially_observable = False; env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env

def wrapper(env, kind):
    while hasattr(env, "env"):
        if isinstance(env, kind): return env
        env = env.env
    raise AssertionError(kind)

def test_construct_reset_step_and_spaces():
    env = direct()
    try:
        obs, info = env.reset(); assert obs.shape == (39,) and info == {}
        assert env.action_space.shape == (4,)
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated and not truncated and "success" in info
    finally: env.close()

def test_object_goal_distribution_and_separation():
    env = direct(10_000)
    try:
        for _ in range(50):
            env.reset()
            assert np.linalg.norm(env.obj_init_pos[:2] - env._target_pos[:2]) >= 0.15
            assert env._random_reset_space.contains(np.r_[env.obj_init_pos[:2], .02, env._target_pos])
    finally: env.close()

def test_reward_formula_progress_success_and_threshold(monkeypatch):
    env = direct()
    try:
        obs, _ = env.reset(); action = np.array([0, 0, 0, 1], np.float32)
        monkeypatch.setattr(env, "_robotiq_caging_reward", lambda *a, **k: 0.4)
        tcp = [env._target_pos + [0.30, 0, 0]]
        monkeypatch.setattr(type(env), "tcp_center", property(lambda self: tcp[0]))
        obs[3] = 1.0; obs[4:7] = env._target_pos + [0.10, 0, 0]
        assert env.compute_reward(action, obs)[0] == pytest.approx(0.8)
        obs[4:7] = 0.0
        tcp[0] = np.array([env.NEAR_OBJECT_RADIUS, 0.0, 0.0])
        assert env.compute_reward(action, obs)[0] == pytest.approx(0.8)
        tcp[0][0] = np.nextafter(env.NEAR_OBJECT_RADIUS, 0.0)
        near, *_, in_place = env.compute_reward(action, obs)
        assert near == pytest.approx(1 + 4 * 0.4 + 5 * in_place)
        obs[4:7] = env._target_pos + [0.05, 0, 0]
        reward, info = env.evaluate_state(obs, action)
        assert info["success"] == 1.0 and reward != 10.0
        obs[4:7] = env._target_pos + [0.049, 0, 0]
        assert env.evaluate_state(obs, action)[0] == 10.0
    finally: env.close()

def test_robotiq_caging_geometry_is_finite():
    env = direct()
    try:
        env.reset(); action = np.array([0, 0, 0, 1], np.float32)
        value = env._robotiq_caging_reward(action, env.obj_init_pos, env.OBJECT_RADIUS,
            env.PAD_SUCCESS_THRESHOLD, 0.01, 0.005, high_density=True)
        assert np.isfinite(value) and 0 <= value <= 1
        left, right = env.get_body_com("leftpad"), env.get_body_com("rightpad")
        assert abs(left[1] - right[1]) > abs(left[0] - right[0])
    finally: env.close()

@pytest.mark.parametrize("version", ["v3", "v3_1", "v3_2_retention"])
def test_obsolete_reward_modes_rejected_for_ordinary_push(version):
    with pytest.raises(ValueError, match="v2.*only"):
        KukaPushEnvV3(reward_function_version=version)
    with pytest.raises(ValueError, match="Unsupported reward_function_version"):
        gym.make(KUKA_MT1_ID, env_name="kuka-push-v3", seed=1,
                 reward_function_version=version, disable_env_checker=True)

def test_fixed_task_bank_is_deterministic_and_randomized():
    banks = []
    for _ in range(2):
        env = gym.make(KUKA_MT1_ID, env_name="kuka-push-v3", seed=10_000,
                       reward_function_version="v2", disable_env_checker=True)
        try:
            tasks = wrapper(env, RandomTaskSelectWrapper).tasks
            banks.append([pickle.loads(task.data)["rand_vec"] for task in tasks])
            assert len({tuple(x) for x in banks[-1]}) > 1
        finally: env.close()
    np.testing.assert_array_equal(banks[0], banks[1])

def test_scripted_push_moves_object_toward_goal():
    env = direct(7)
    try:
        env.reset(); initial = env.get_body_com("obj").copy()
        initial_distance = np.linalg.norm(initial - env._target_pos)
        direction = env._target_pos[:2] - initial[:2]; direction /= np.linalg.norm(direction)
        behind = initial.copy(); behind[:2] -= 0.04 * direction; behind[2] = 0.047
        for target, steps in ((behind + [0,0,.10], 25), (behind, 40)):
            for _ in range(steps):
                env.step(np.r_[np.clip((target-env.tcp_center)/env.action_scale,-1,1),1].astype(np.float32))
        for distance in np.linspace(0, .25, 150):
            target = behind + np.r_[direction * distance, 0]
            env.step(np.r_[np.clip((target-env.tcp_center)/env.action_scale,-1,1),1].astype(np.float32))
        assert np.linalg.norm(env.get_body_com("obj") - env._target_pos) < initial_distance
    finally: env.close()
