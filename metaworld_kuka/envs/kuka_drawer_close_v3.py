"""KUKA drawer-close task with current MetaWorld V3 semantics."""

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

_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_drawer_close_v3.xml"


class KukaDrawerCloseEnvV3(KukaXYZEnv):
    """Push an initially open drawer to its closed position."""

    TARGET_RADIUS: float = 0.04

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        # HISTORICAL_KUKA: bounds, initial poses, fixture position and 0.20 m travel.
        hand_low, hand_high = (-0.25, 0.42, 0.05), (0.25, 0.75, 0.5)
        obj_low, obj_high = (-0.1, 0.9, 0.04), (0.1, 0.9, 0.04)
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
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.9, 0.04]),
            "hand_init_pos": np.array([0.0, 0.45, 0.2]),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(hand_low), np.array(hand_high), dtype=np.float64)
        self.maxDist = 0.2

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_obj, _, target_to_obj, object_grasped, in_place = (
            self.compute_reward(action, obs)
        )
        return reward, {
            "success": float(target_to_obj <= self.TARGET_RADIUS + 0.015),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("handleStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        self.data.joint("goal_slidey").qpos[0] = float(np.asarray(pos))
        self.data.joint("goal_slidey").qvel[0] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("drawer").pos = self.obj_init_pos
        self.model.body("drawer_cover").pos = self.obj_init_pos + np.array(
            [0.0, 0.0, -0.02]
        )
        # CURRENT_SAWYER semantics: target is the closed handle pose.
        self._target_pos = self.obj_init_pos + np.array([0.0, -0.2, 0.08])
        self._set_obj_xyz(np.array(-self.maxDist))
        self.obj_init_pos = self._get_pos_objects().copy()
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA V3 tasks support reward_function_version='v2' only")
        obj, tcp, target = obs[4:7], self.tcp_center, self._target_pos
        target_to_obj = float(np.linalg.norm(obj - target))
        target_to_obj_init = float(np.linalg.norm(self.obj_init_pos - target))
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_init = float(np.linalg.norm(self.obj_init_pos - self.init_tcp))
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.005),
            margin=abs(tcp_to_obj_init - 0.005),
            sigmoid="gaussian",
        )
        reach = reward_utils.hamacher_product(reach, min(max(0, action[-1]), 1))
        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.0
        return 10 * reward, tcp_to_obj, 0.0, target_to_obj, reach, in_place
