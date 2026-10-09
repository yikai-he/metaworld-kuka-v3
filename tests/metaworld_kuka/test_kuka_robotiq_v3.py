"""Shared Robotiq 2F-85 integration contracts for every formal KUKA-v3 task."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from metaworld_kuka.envs import KukaHammerEnvV3, KukaReachEnvV3
from metaworld_kuka.registration import KUKA_ENVIRONMENTS


def configured(env_cls=KukaReachEnvV3):
    env = env_cls(reward_function_version="v2")
    env.seed(11)
    env.seeded_rand_vec = True
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env.observation_space = env.sawyer_observation_space
    return env


def settle(env, command: float, steps: int = 80) -> None:
    action = np.array([0.0, 0.0, 0.0, command], dtype=np.float32)
    for _ in range(steps):
        obs, *_ = env.step(action)
        assert np.isfinite(obs).all()


def test_model_contains_provenance_backed_robotiq_mechanism() -> None:
    env = configured()
    try:
        body_names = {env.model.body(i).name for i in range(env.model.nbody)}
        joint_names = {env.model.joint(i).name for i in range(env.model.njnt)}
        assert {
            "base_mount",
            "base",
            "right_driver",
            "left_driver",
            "right_coupler",
            "left_coupler",
            "right_spring_link",
            "left_spring_link",
            "right_follower",
            "left_follower",
            "right_pad",
            "left_pad",
        } <= body_names
        assert {
            "right_driver_joint",
            "left_driver_joint",
            "right_coupler_joint",
            "left_coupler_joint",
            "right_spring_link_joint",
            "left_spring_link_joint",
            "right_follower_joint",
            "left_follower_joint",
        } <= joint_names
        assert "r_close" not in joint_names and "l_close" not in joint_names
        assert env.model.nu == 1
        assert env.model.actuator(0).name == "fingers_actuator"
        assert env.model.tendon(0).name == "split"
        assert env.model.ntendon == 1 and env.model.neq >= 4
        assert sum(env.model.eq_type == mujoco.mjtEq.mjEQ_CONNECT) == 2
        assert sum(env.model.eq_type == mujoco.mjtEq.mjEQ_JOINT) == 1
    finally:
        env.close()


def test_action_and_observation_aperture_semantics_are_monotonic() -> None:
    env = configured()
    try:
        obs, _ = env.reset()
        assert obs.shape == (39,) and env.action_space.shape == (4,)
        assert np.isfinite(obs).all()
        np.testing.assert_allclose(obs[36:39], env._target_pos)
        assert obs[3] > 0.9
        open_aperture = env.gripper_aperture

        settle(env, +1.0)
        closed_obs = env._get_obs()
        closed_aperture = env.gripper_aperture
        assert closed_aperture < open_aperture
        assert closed_obs[3] < 0.1
        np.testing.assert_allclose(env.data.ctrl, [255.0])

        settle(env, -1.0)
        reopened_obs = env._get_obs()
        assert env.gripper_aperture > closed_aperture
        assert reopened_obs[3] > 0.9
        np.testing.assert_allclose(env.data.ctrl, [0.0])
    finally:
        env.close()


def test_tcp_is_fingertip_midpoint_and_policy_ee() -> None:
    env = configured()
    try:
        env.reset()
        right = env._get_site_pos("rightEndEffector")
        left = env._get_site_pos("leftEndEffector")
        np.testing.assert_allclose(env.tcp_center, (right + left) / 2.0)
        np.testing.assert_allclose(env.get_endeff_pos(), env.tcp_center)
        # The fixed welded hand frame is the nominal pinch reference; linkage
        # compliance may shift the physical midpoint by less than 2 cm.
        assert np.linalg.norm(env.data.body("hand").xpos - env.tcp_center) < 0.02
    finally:
        env.close()


def test_robotiq_pad_contact_lookup_detects_bilateral_object_contact() -> None:
    env = configured(KukaHammerEnvV3)
    try:
        env.reset()
        hammer = env.get_body_com("hammer").copy()
        target = hammer + np.array([0.07, 0.0, 0.0])
        for gripper, steps, offset in (
            (-1.0, 30, np.array([0.0, 0.0, 0.06])),
            (-1.0, 40, np.zeros(3)),
            (+1.0, 60, np.zeros(3)),
        ):
            for _ in range(steps):
                action = np.r_[
                    np.clip(
                        (target + offset - env.tcp_center) / env.action_scale,
                        -1,
                        1,
                    ),
                    gripper,
                ].astype(np.float32)
                obs, *_ = env.step(action)
                assert np.isfinite(obs).all()
        assert env.touching_main_object
        contacting_names = set()
        object_id = env._get_id_main_object()
        for contact in env.data.contact:
            if object_id in (contact.geom1, contact.geom2):
                contacting_names.update(
                    (env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name)
                )
        assert contacting_names & {"right_pad1", "right_pad2"}
        assert contacting_names & {"left_pad1", "left_pad2"}
    finally:
        env.close()


@pytest.mark.parametrize("name,env_cls", tuple(KUKA_ENVIRONMENTS.items()))
def test_every_formal_kuka_task_uses_robotiq_and_resets_stably(name, env_cls) -> None:
    env = configured(env_cls)
    try:
        obs, info = env.reset()
        assert info == {} and obs.shape == (39,) and np.isfinite(obs).all(), name
        assert env.model.nu == 1
        assert env.model.actuator(0).name == "fingers_actuator"
        assert env.gripper_aperture > 0.075
        assert env.tcp_center[2] > 0.02
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(
                np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            )
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated and not truncated
    finally:
        env.close()
