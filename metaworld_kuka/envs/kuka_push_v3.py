"""KUKA push task with MetaWorld 3 push semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils
from scipy.spatial.transform import Rotation

from metaworld_kuka.kuka_xyz_env import KukaXYZEnv

_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_push_v3.xml"


class KukaPushEnvV3(KukaXYZEnv):
    """Push a puck to a sampled planar goal with the KUKA end effector."""

    TARGET_RADIUS: float = 0.05
    OBJECT_RADIUS: float = 0.03
    PAD_SUCCESS_THRESHOLD: float = 0.055
    NEAR_OBJECT_RADIUS: float = 0.055

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        if reward_function_version != "v2":
            raise ValueError("KUKA Push supports reward_function_version='v2' only")
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
        if self.reward_function_version in (
            "v3", "v3_1", "v3_2_retention", "v3_3_contact_push_progress",
            "v4_simple_approach_progress",
            "v5_contact_start_progress",
            "v6_rear_contact_push_progress",
            "v1_initial_to_contact",
            "v1_1_initial_to_contact_aperture",
            "sawyer_push_v2_adapted",
        ):
            info.update(self._last_reward_components)
        return reward, info

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        return np.array(
            [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]
        )

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
        self._target_pos = np.concatenate([goal_pos[-3:-1], [self.obj_init_pos[-1]]])
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])
        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - self._target_pos
            )
            + self.heightTarget
        )
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA Push supports reward_function_version='v2' only")
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
        object_grasped = self._robotiq_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=self.OBJECT_RADIUS,
            pad_success_thresh=self.PAD_SUCCESS_THRESHOLD,
            xz_thresh=0.005,
            high_density=True,
        )
        reward = 2 * object_grasped
        if tcp_to_obj < self.NEAR_OBJECT_RADIUS and tcp_opened > 0:
            reward += 1.0 + reward + 5.0 * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place

    def _robotiq_caging_reward(
        self, action, obj_pos, obj_radius, pad_success_thresh,
        object_reach_radius, xz_thresh, *, high_density=False
    ) -> float:
        """Sawyer caging reward mapped to the 2F-85 opening geometry.

        The fixed KUKA wrist places the Robotiq opening axis on world Y, as in
        Sawyer's task. All tolerance and Hamacher-product semantics are kept.
        """
        del object_reach_radius
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        pad_y = np.array([left_pad[1], right_pad[1]])
        pad_to_obj = np.abs(pad_y - obj_pos[1])
        pad_to_initial = np.abs(pad_y - self.obj_init_pos[1])
        margins = np.abs(pad_to_initial - pad_success_thresh)
        cage_sides = [
            reward_utils.tolerance(
                pad_to_obj[index], bounds=(obj_radius, pad_success_thresh),
                margin=margins[index], sigmoid="long_tail"
            )
            for index in range(2)
        ]
        cage_y = reward_utils.hamacher_product(*cage_sides)
        xz = [0, 2]
        center_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        center_margin -= xz_thresh
        cage_xz = reward_utils.tolerance(
            np.linalg.norm(self.tcp_center[xz] - obj_pos[xz]),
            bounds=(0, xz_thresh), margin=center_margin, sigmoid="long_tail"
        )
        caging = reward_utils.hamacher_product(cage_y, float(cage_xz))
        gripping = min(max(0.0, float(action[-1])), 1.0) if caging > 0.97 else 0.0
        result = reward_utils.hamacher_product(caging, gripping)
        if high_density:
            result = (result + caging) / 2
        return float(result)
