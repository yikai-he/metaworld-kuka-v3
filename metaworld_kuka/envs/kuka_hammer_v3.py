"""KUKA hammer task with MetaWorld 3 semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import HammerInitConfigDict
from metaworld.utils import reward_utils

from metaworld_kuka.kuka_xyz_env import KukaXYZEnv

_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_hammer_v3.xml"


class KukaHammerEnvV3(KukaXYZEnv):
    """Pick up a hammer and drive the nail along its positive-Y slide."""

    HAMMER_HANDLE_LENGTH = 0.14

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        hammer_reward_variant: str = "original",
        hammer_nail_progress_weight: float = 0.0,
        height: int = 480,
        width: int = 480,
    ) -> None:
        # ROBOTIQ_GEOMETRY_CALIBRATION: the 2F-85 needs a lower TCP than the
        # synthetic fingers to overlap the hammer handle's vertical center.
        hand_low = (-0.25, 0.42, 0.03)
        hand_high = (0.25, 0.75, 0.50)
        obj_low = (-0.10, 0.42, 0.03)
        obj_high = (0.10, 0.50, 0.03)
        goal_low = (0.2399, 0.7399, 0.109)
        goal_high = (0.2401, 0.7401, 0.111)
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
        if hammer_reward_variant not in ("original", "nail_progress"):
            raise ValueError(
                "hammer_reward_variant must be 'original' or 'nail_progress'"
            )
        if hammer_nail_progress_weight < 0:
            raise ValueError("hammer_nail_progress_weight must be nonnegative")
        if hammer_reward_variant == "original" and hammer_nail_progress_weight != 0:
            raise ValueError(
                "original Hammer reward requires hammer_nail_progress_weight=0"
            )
        self.hammer_reward_variant = hammer_reward_variant
        self.hammer_nail_progress_weight = float(hammer_nail_progress_weight)
        self.init_config: HammerInitConfigDict = {
            "hammer_init_pos": np.array([0.0, 0.45, 0.03]),
            "hand_init_pos": np.array([0.0, 0.42, 0.20]),
        }
        self.goal = self.init_config["hammer_init_pos"]
        self.hammer_init_pos = self.init_config["hammer_init_pos"]
        self.obj_init_pos = self.hammer_init_pos.copy()
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.nail_init_pos: npt.NDArray[Any] | None = None
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, grab, ready, in_place, success = self.compute_reward(action, obs)
        return reward, {
            "success": float(success),
            "near_object": ready,
            "grasp_success": grab >= 0.5,
            "grasp_reward": grab,
            "in_place_reward": in_place,
            "obj_to_target": 0.0,
            "unscaled_reward": reward,
        }

    def _get_id_main_object(self) -> int:
        return self.model.geom("HammerHandleCollision").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (self.get_body_com("hammer").copy(), self.get_body_com("nail_link").copy())
        )

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (self.data.body("hammer").xquat, self.data.body("nail_link").xquat)
        )

    def _set_hammer_xyz(self, pos: npt.NDArray[Any]) -> None:
        joint = self.data.joint("hammerjoint")
        joint.qpos[:3] = pos
        joint.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.model.body("box").pos = np.array([0.24, 0.85, 0.0])
        self._target_pos = self._get_site_pos("goal")
        self.hammer_init_pos = self._get_state_rand_vec()
        self.nail_init_pos = self._get_site_pos("nailHead")
        self.obj_init_pos = self.hammer_init_pos.copy()
        self._set_hammer_xyz(self.hammer_init_pos)
        self.liftThresh = 0.09
        self.hammerHeight = self.get_body_com("hammer")[2]
        self.heightTarget = self.hammerHeight + self.liftThresh
        self.maxHammerDist = (
            np.linalg.norm(
                np.array(
                    [
                        self.hammer_init_pos[0],
                        self.hammer_init_pos[1],
                        self.heightTarget,
                    ]
                )
                - self.obj_init_pos
            )
            + self.heightTarget
            + abs(self.obj_init_pos[1] - self._target_pos[1])
        )
        self.pickCompleted = False
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs: npt.NDArray[np.float64]) -> float:
        error = float(np.linalg.norm(obs[7:11] - np.array([1.0, 0.0, 0.0, 0.0])))
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(
        hammer_head: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> float:
        lifted = hammer_head[2] > 0.02
        return 0.1 * float(lifted) + 0.9 * reward_utils.tolerance(
            np.linalg.norm(target_pos - hammer_head),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

    @staticmethod
    def _nail_progress(nail_qpos: float) -> float:
        return float(np.clip(nail_qpos / 0.09, 0.0, 1.0))

    @staticmethod
    def _is_success(nail_qpos: float) -> bool:
        return bool(nail_qpos > 0.09)

    def _shape_nail_progress_reward(
        self, raw_reward: float, nail_qpos: float
    ) -> float:
        if self.hammer_reward_variant == "original":
            return raw_reward
        return raw_reward + self.hammer_nail_progress_weight * self._nail_progress(
            nail_qpos
        )

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, bool]:
        assert self._target_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA V3 tasks support reward_function_version='v2' only")
        hand, hammer = obs[:3], obs[4:7]
        hammer_head = hammer + np.array([0.16, 0.06, 0.0])
        hammer_for_caging = hammer.copy()
        if abs(hammer[0] - hand[0]) < self.HAMMER_HANDLE_LENGTH / 2.0:
            hammer_for_caging[0] = hand[0]
        reward_quat = self._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions,
            hammer_for_caging,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )
        reward_in_place = self._reward_pos(hammer_head, self._target_pos)
        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        nail_qpos = float(self.data.joint("NailSlideJoint").qpos[0])
        reward = self._shape_nail_progress_reward(reward, nail_qpos)
        success = self._is_success(nail_qpos)
        if success and reward > 5.0:
            reward = 10.0
        return reward, reward_grab, reward_quat, reward_in_place, success
