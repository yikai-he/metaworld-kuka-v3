"""Contract tests for the clean Sawyer-inspired 40 mm Cube Push task."""
import pickle
import gymnasium as gym
import mujoco
import numpy as np
import pytest
from metaworld.wrappers import RandomTaskSelectWrapper
import metaworld_kuka  # noqa: F401
from metaworld_kuka.envs import KukaPushCubeEnvV3
from metaworld_kuka.registration import KUKA_MT1_ID

def direct(seed=7):
    env = KukaPushCubeEnvV3(); env.seed(seed); env.seeded_rand_vec = True
    env._freeze_rand_vec = False; env._set_task_called = True
    env._partially_observable = False; env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env

def find_wrapper(env, kind):
    while hasattr(env, "env"):
        if isinstance(env, kind): return env
        env = env.env
    raise AssertionError(kind)

def test_contract_cube_geometry_and_stable_upright_reset():
    env = direct()
    try:
        obs, _ = env.reset(); geom = env.model.geom("objGeom")
        assert obs.shape == (39,) and obs.dtype == np.float64
        assert env.action_space.shape == (4,)
        np.testing.assert_allclose(geom.size, [0.02, 0.02, 0.02])
        np.testing.assert_allclose(env.data.joint("objjoint").qpos[3:], [1,0,0,0])
        initial = obs[4:7].copy()
        for _ in range(20): env.step(np.zeros(4, np.float32))
        assert np.linalg.norm(env._get_obs()[4:7] - initial) < 0.003
    finally: env.close()

def test_registered_float32_contract_and_deterministic_task_bank():
    banks = []
    for _ in range(2):
        env = gym.make(KUKA_MT1_ID, env_name="kuka-push-cube-v3", seed=10000,
                       reward_function_version="v2", disable_env_checker=True)
        try:
            obs, _ = env.reset(); assert obs.shape == (39,) and obs.dtype == np.float32
            wrapper = find_wrapper(env, RandomTaskSelectWrapper)
            banks.append([pickle.loads(task.data)["rand_vec"] for task in wrapper.tasks])
        finally: env.close()
    np.testing.assert_array_equal(banks[0], banks[1])

def test_distribution_separation_and_upright_orientation():
    env = direct(10000)
    try:
        for _ in range(50):
            env.reset()
            assert np.linalg.norm(env.obj_init_pos[:2] - env._target_pos[:2]) >= .15
            np.testing.assert_allclose(env.data.joint("objjoint").qpos[3:], [1,0,0,0])
    finally: env.close()

def test_reward_boundaries_and_near_gate(monkeypatch):
    env = direct()
    try:
        obs, _ = env.reset(); action=np.array([0,0,0,1],np.float32)
        monkeypatch.setattr(env, "_robotiq_caging_reward", lambda *a, **k: .4)
        tcp=[np.array([.055,0,0])]
        monkeypatch.setattr(type(env), "tcp_center", property(lambda self: tcp[0]))
        obs[3]=1; obs[4:7]=0
        assert env.compute_reward(action,obs)[0] == pytest.approx(.8)
        tcp[0][0]=np.nextafter(.055,0)
        assert env.compute_reward(action,obs)[0] > .8
        obs[4:7]=env._target_pos+[.05,0,0]
        reward,*_=env.compute_reward(action,obs); assert reward != 10
        assert env.evaluate_state(obs,action)[1]["success"] == 1
        obs[4:7]=env._target_pos+[.049,0,0]
        assert env.compute_reward(action,obs)[0] == 10
    finally: env.close()

def test_caging_constants_are_geometric_and_reachable():
    env=direct()
    try:
        env.reset(); assert env.OBJECT_RADIUS == pytest.approx(np.sqrt(3)*.02)
        assert env.OBJECT_RADIUS < env.PAD_SUCCESS_THRESHOLD
        values=[]
        for z in (.035,.04,.045):
            env.data.mocap_pos[0]=[env.obj_init_pos[0],env.obj_init_pos[1],z]
            mujoco.mj_forward(env.model,env.data)
            values.append(env._robotiq_caging_reward(np.array([0,0,0,1]),env._get_pos_objects(),
                env.OBJECT_RADIUS,env.PAD_SUCCESS_THRESHOLD,.01,env.XZ_ALIGNMENT_THRESHOLD,high_density=True))
        assert np.isfinite(values).all() and max(values) > 0
    finally: env.close()

@pytest.mark.parametrize("kwargs", [
    {"reward_function_version":"v3_1"}, {"cube_push_curriculum_level":2},
    {"cube_push_initialization":"near_contact"}, {"near_contact_gap_m":.005},
])
def test_obsolete_modes_are_rejected(kwargs):
    with pytest.raises((TypeError,ValueError)):
        gym.make(KUKA_MT1_ID,env_name="kuka-push-cube-v3",seed=1,
                 disable_env_checker=True,**kwargs)

def test_scripted_push_contacts_and_moves_cube_toward_goal():
    env=direct(7)
    try:
        env.reset(); initial=env._get_pos_objects().copy(); d0=np.linalg.norm(initial-env._target_pos)
        direction=env._target_pos[:2]-initial[:2]; direction/=np.linalg.norm(direction)
        behind=initial.copy(); behind[:2]-=.065*direction; behind[2]=.045
        contact=False
        for target,steps in ((behind+[0,0,.10],25),(behind,45)):
            for _ in range(steps):
                env.step(np.r_[np.clip((target-env.tcp_center)/env.action_scale,-1,1),1].astype(np.float32))
        for distance in np.linspace(0,.28,180):
            target=behind+np.r_[direction*distance,0]
            env.step(np.r_[np.clip((target-env.tcp_center)/env.action_scale,-1,1),1].astype(np.float32))
            contact |= env.robot_object_contact_diagnostics(env._get_id_main_object())[0]
        final=env._get_pos_objects()
        assert contact and np.linalg.norm(final-initial)>.01
        assert np.linalg.norm(final-env._target_pos)<d0
    finally: env.close()
