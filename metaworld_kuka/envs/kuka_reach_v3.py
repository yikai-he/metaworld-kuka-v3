"""KUKA reach task with MetaWorld 3 reach semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils
from metaworld_kuka.kuka_xyz_env import KukaXYZEnv


_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_reach_v3.xml"


class KukaReachEnvV3(KukaXYZEnv):
    """Reach a sampled Cartesian goal with the KUKA tool center point."""

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        # Sawyer's reach bounds extend to y=1.0. These bounds are the calibrated
        # subset supported by the legacy KUKA model and its fixed wrist pose.
        goal_low = (-0.15, 0.62, 0.12)
        goal_high = (0.15, 0.78, 0.32)
        hand_low = (-0.25, 0.42, 0.10)
        hand_high = (0.25, 0.82, 0.40)
        obj_low = (-0.10, 0.48, 0.02)
        obj_high = (0.10, 0.58, 0.02)

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
            "obj_init_pos": np.array([0.0, 0.54, 0.02]),
            "hand_init_pos": np.array([0.0, 0.45, 0.20]),
        }
        self.goal = np.array([0.0, 0.70, 0.20])
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)
        info = {
            "success": success,
            "near_object": reach_dist,
            "grasp_success": 1.0,
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        return np.array([adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))
        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        assert self._target_pos is not None
        if self.reward_function_version == "v2":
            target_radius = 0.05
            tcp_to_target = float(np.linalg.norm(self.tcp_center - self._target_pos))
            in_place_margin = float(np.linalg.norm(self.hand_init_pos - self._target_pos))
            in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, target_radius),
                margin=in_place_margin,
                sigmoid="long_tail",
            )
            return 10 * in_place, tcp_to_target, in_place

        right_finger = self._get_site_pos("rightEndEffector")
        left_finger = self._get_site_pos("leftEndEffector")
        finger_com = (right_finger + left_finger) / 2
        reach_dist = np.linalg.norm(finger_com - self._target_pos)
        del actions, obs
        reach_reward = 1000 * (self.maxReachDist - reach_dist) + 1000 * (
            np.exp(-(reach_dist**2) / 0.01) + np.exp(-(reach_dist**2) / 0.001)
        )
        return float(max(reach_reward, 0)), float(reach_dist), 0.0
