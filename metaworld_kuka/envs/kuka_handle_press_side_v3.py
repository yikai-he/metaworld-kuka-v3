"""KUKA side-handle press with MetaWorld 3 semantics."""

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

_MODEL = (
    Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_handle_press_side_v3.xml"
)


class KukaHandlePressSideEnvV3(KukaXYZEnv):
    """Press the KUKA-side handle downward along its slide joint."""

    TARGET_RADIUS: float = 0.02

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
        # Historical Z=0.05 was inconsistent with the XML body origin. The
        # fixture remains table-mounted, matching Sawyer's randomization contract.
        obj_low = (0.22, 0.45, -0.001)
        obj_high = (0.32, 0.60, 0.001)
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
            "obj_init_pos": np.array([0.22, 0.52, 0.0]),
            "hand_init_pos": np.array([-0.05, 0.50, 0.20]),
        }
        self.goal = np.array([0.004, 0.52, 0.075])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(hand_low), np.array(hand_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_obj, _, target_to_obj, grasped, in_place = self.compute_reward(
            action, obs
        )
        return reward, {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("handleStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        self.data.joint("goal_slidez").qpos[0] = float(np.asarray(pos))
        self.data.joint("goal_slidez").qvel[0] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("box").pos = self.obj_init_pos
        self._set_obj_xyz(np.array(-0.001))
        self._target_pos = self._get_site_pos("goalPress")
        self._handle_init_pos = self._get_pos_objects()
        self.maxDist = abs(self.data.site("handleStart").xpos[2] - self._target_pos[2])
        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        del actions, obs
        assert self._target_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA V3 tasks support reward_function_version='v2' only")
        obj = self._get_pos_objects()
        tcp = self.tcp_center
        target_to_obj = float(abs(obj[2] - self._target_pos[2]))
        target_to_obj_init = float(abs(self._handle_init_pos[2] - self._target_pos[2]))
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        handle_radius = 0.02
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_init = float(np.linalg.norm(self._handle_init_pos - self.init_tcp))
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid="long_tail",
        )
        reward = 10 * reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS:
            reward = 10.0
        return reward, tcp_to_obj, 0.0, target_to_obj, reach, in_place
