"""KUKA push task with MetaWorld 3 push semantics."""

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


_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_push_v3.xml"


class KukaPushEnvV3(KukaXYZEnv):
    """Push a puck to a sampled planar goal with the KUKA end effector."""

    TARGET_RADIUS: float = 0.05

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        # Geometry from fcc9e34; task semantics follow SawyerPushEnvV3.
        # ROBOTIQ_GEOMETRY_CALIBRATION: 0.05 m left the lowest 2F-85 pad
        # above the puck. 0.04 m permits side contact without table contact.
        hand_low = (-0.25, 0.42, 0.04)
        hand_high = (0.25, 0.75, 0.50)
        # Keep the conservative hardware-informed y workspace. With the
        # Robotiq/puck approach offset (0.054 m), y >= 0.484 leaves at least
        # 10 mm of task-space clearance and avoids driving KUKA A4 to its
        # -120 degree limit at the fixed push orientation.
        obj_low = (-0.10, 0.484, 0.02)
        obj_high = (0.10, 0.55, 0.02)
        goal_low = (-0.10, 0.65, 0.01)
        goal_high = (0.10, 0.75, 0.02)

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
        self._reward_v3_success_latched = False
        self._reward_v3_1_success_latched = False
        self._last_reward_components: dict[str, float | bool] = {}
        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.45, 0.02]),
            "hand_init_pos": np.array([0.0, 0.45, 0.20]),
        }
        self.goal = np.array([0.10, 0.65, 0.02])
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
        obj = obs[4:7]
        reward, tcp_to_obj, tcp_opened, target_to_obj, grasped, in_place = (
            self.compute_reward(action, obs)
        )
        assert self.obj_init_pos is not None
        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float(
                self.touching_main_object
                and tcp_opened > 0
                and obj[2] - 0.02 > self.obj_init_pos[2]
            ),
            "grasp_reward": grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }
        if self.reward_function_version in ("v3", "v3_1"):
            info.update(self._last_reward_components)
        return reward, info

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(self.data.geom("objGeom").xmat.reshape(3, 3)).as_quat()

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        return np.array([adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reward_v3_success_latched = False
        self._reward_v3_1_success_latched = False
        self._last_reward_components = {}
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = np.concatenate([goal_pos[-3:-1], [self.obj_init_pos[-1]]])
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])
        self.maxPlacingDist = (
            np.linalg.norm(
                np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget])
                - self._target_pos
            )
            + self.heightTarget
        )
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version not in ("v2", "v3", "v3_1"):
            raise ValueError(
                "KUKA push supports reward_function_version='v2', 'v3', or "
                "'v3_1' only"
            )
        obj = obs[4:7]
        tcp_opened = obs[3]
        tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
        target_to_obj = float(np.linalg.norm(obj - self._target_pos))
        target_to_obj_init = float(np.linalg.norm(self.obj_init_pos - self._target_pos))
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True,
        )
        contact = bool(
            self.pad_contacting_object(self._get_id_main_object()) and tcp_opened > 0
        )
        if self.reward_function_version in ("v3", "v3_1"):
            previous_obj = obs[22:25]
            goal = obs[36:39]
            previous_target_to_obj = float(np.linalg.norm(previous_obj - goal))
            signed_progress = previous_target_to_obj - target_to_obj
            clipped_progress = float(np.clip(signed_progress, -0.005, 0.005))
            is_v3_1 = self.reward_function_version == "v3_1"
            success_before_step = (
                self._reward_v3_1_success_latched
                if is_v3_1
                else self._reward_v3_success_latched
            )
            success_now = target_to_obj <= self.TARGET_RADIUS
            first_success = success_now and not success_before_step

            if success_before_step:
                reward = 0.0
            elif first_success:
                reward = 100.0
                if is_v3_1:
                    self._reward_v3_1_success_latched = True
                else:
                    self._reward_v3_success_latched = True
            elif is_v3_1 and contact:
                reward = 0.05 + 1000.0 * clipped_progress
            elif is_v3_1:
                reward = object_grasped
            else:
                reward = (
                    0.1 * object_grasped
                    + 0.05 * float(contact)
                    + 1000.0 * float(contact) * clipped_progress
                )
            prefix = "reward_v3_1" if is_v3_1 else "reward_v3"
            self._last_reward_components = {
                f"{prefix}_caging": float(object_grasped),
                f"{prefix}_contact": contact,
                f"{prefix}_signed_progress": signed_progress,
                f"{prefix}_clipped_progress": clipped_progress,
                f"{prefix}_success_latched": (
                    self._reward_v3_1_success_latched
                    if is_v3_1
                    else self._reward_v3_success_latched
                ),
                f"{prefix}_first_success": first_success,
                f"{prefix}_total": float(reward),
            }
            return (
                float(reward),
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
                in_place,
            )

        reward = 2 * object_grasped
        # The Robotiq TCP is the fingertip midpoint. Runtime calibration shows
        # useful pad contact at tcp_to_obj=0.040--0.054 m, so the historical
        # synthetic-gripper 0.02 m gate can never activate. Gate the existing
        # shaping bonus on the physical event it was intended to represent.
        if contact:
            reward += 1.0 + reward + 5.0 * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place
