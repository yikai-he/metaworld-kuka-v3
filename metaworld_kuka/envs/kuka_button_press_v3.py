"""Horizontal KUKA button press with MetaWorld 3 semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils
from metaworld_kuka.kuka_xyz_env import KukaXYZEnv


_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_button_press_v3.xml"


class KukaButtonPressEnvV3(KukaXYZEnv):
    """Press a standard horizontal button along its negative-Y slide axis."""

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.25, 0.42, 0.05)
        hand_high = (0.25, 0.75, 0.50)
        # The horizontal button face is 0.1935 m in front of the fixture. This
        # places it in the historical KUKA operating region near y=0.50--0.56.
        obj_low = (-0.10, 0.69, 0.115)
        obj_high = (0.10, 0.75, 0.115)
        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version
        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0.0, 0.72, 0.115], dtype=np.float32),
            "hand_init_pos": np.array([0.0, 0.45, 0.20], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.62, 0.12])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(
            np.array(hand_low), np.array(hand_high), dtype=np.float64
        )

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_obj, tcp_open, obj_to_target, near_button, pressed = (
            self.compute_reward(action, obs)
        )
        return reward, {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_id_main_object(self) -> int:
        return self.model.geom("btnGeom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("button") + np.array([0.0, -0.193, 0.0])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("button").xquat.copy()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        self.data.joint("btnbox_joint").qpos[0] = float(np.asarray(pos))
        self.data.joint("btnbox_joint").qvel[0] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("box").pos = self.obj_init_pos
        self._set_obj_xyz(np.array(0.0))
        self._target_pos = self._get_site_pos("hole")
        self._obj_to_target_init = abs(
            self._target_pos[1] - self._get_site_pos("buttonStart")[1]
        )
        self.maxDist = self._obj_to_target_init
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        del action
        assert self._target_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA V3 tasks support reward_function_version='v2' only")
        obj = obs[4:7]
        tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
        tcp_to_obj_init = float(np.linalg.norm(obj - self.init_tcp))
        obj_to_target = float(abs(self._target_pos[1] - obj[1]))
        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid="long_tail",
        )
        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed
        return reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed
