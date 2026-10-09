"""Native-MuJoCo smoke tests for the isolated KUKA XYZ base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld_kuka.kuka_xyz_env import KukaXYZEnv


ASSET = (
    Path(__file__).parents[2]
    / "metaworld_kuka"
    / "assets"
    / "kuka_xyz"
    / "kuka_base_smoke.xml"
)


class SmokeKukaXYZEnv(KukaXYZEnv):
    """Small concrete environment used only to validate the robot base."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(
            hand_low=(-0.25, 0.42, 0.10),
            hand_high=(0.25, 0.82, 0.40),
            render_mode=render_mode,
            camera_name="corner",
        )
        self.hand_init_pos = np.array([0.0, 0.45, 0.20])
        self.obj_init_pos = np.array([0.0, 0.54, 0.02])
        self._target_pos = np.array([0.0, 0.70, 0.20])
        self.goal_space = Box(
            np.array([-0.15, 0.62, 0.12]),
            np.array([0.15, 0.78, 0.32]),
            dtype=np.float64,
        )
        self._partially_observable = False
        self._set_task_called = True
        self.__dict__.pop("sawyer_observation_space", None)
        self.observation_space = self.sawyer_observation_space

    @property
    def model_name(self) -> str:
        return str(ASSET)

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xpos.copy()

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat.copy()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def evaluate_state(self, obs, action):
        del obs, action
        return 0.0, {"success": 0.0}


def test_kuka_base_contract_and_control() -> None:
    assert issubclass(KukaXYZEnv, SawyerXYZEnv)
    env = SmokeKukaXYZEnv()
    try:
        obs, info = env.reset()
        assert env.action_space.shape == (4,)
        assert env.observation_space.shape == (39,)
        assert obs.shape == (39,)
        assert info == {}
        assert np.isfinite(obs).all()
        np.testing.assert_allclose(obs[:18], obs[18:36])

        initial_mocap = env.data.mocap_pos[0].copy()
        result = env.step(np.array([0.5, -0.5, 2.0, 0.25], dtype=np.float32))
        assert len(result) == 5
        next_obs, _, terminated, truncated, _ = result
        assert np.isfinite(next_obs).all()
        assert not terminated
        assert not truncated
        expected = np.clip(
            initial_mocap + np.array([0.005, -0.005, 0.01]),
            env.mocap_low,
            env.mocap_high,
        )
        np.testing.assert_allclose(env.data.mocap_pos[0], expected)
        np.testing.assert_allclose(env.data.mocap_quat[0], env.MOCAP_QUATERNION)
        np.testing.assert_allclose(env.data.ctrl, [159.375])

        for _ in range(20):
            transition = env.step(env.action_space.sample())
            assert np.isfinite(transition[0]).all()
        reset_obs, _ = env.reset()
        assert np.isfinite(reset_obs).all()
        np.testing.assert_allclose(reset_obs[:18], reset_obs[18:36])
    finally:
        env.close()


def test_rgb_array_rendering_initializes() -> None:
    env = SmokeKukaXYZEnv(render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert frame.shape == (480, 480, 3)
    finally:
        env.close()
