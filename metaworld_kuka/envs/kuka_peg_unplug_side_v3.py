"""Side unplug task for KUKA with original MetaWorld V3 semantics."""

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
    Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_peg_unplug_side_v3.xml"
)


class KukaPegUnplugSideEnvV3(KukaXYZEnv):
    """Grasp an inserted plug and extract it along positive X."""

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
        peg_unplug_reward_variant: str = "original",
        peg_unplug_progress_weight: float = 0.0,
        peg_unplug_progress_gamma: float = 0.99,
    ) -> None:
        hand_low, hand_high = (-0.25, 0.40, 0.05), (0.25, 0.78, 0.50)
        obj_low, obj_high = (-0.22, 0.52, -0.001), (-0.16, 0.68, 0.001)
        goal_low = np.array(obj_low) + np.array([0.194, 0.0, 0.131])
        goal_high = np.array(obj_high) + np.array([0.194, 0.0, 0.131])
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
        if peg_unplug_reward_variant not in ("original", "peg_unplug_progress_v1"):
            raise ValueError(
                "peg_unplug_reward_variant must be 'original' or "
                "'peg_unplug_progress_v1'"
            )
        if peg_unplug_progress_weight < 0:
            raise ValueError("peg_unplug_progress_weight must be nonnegative")
        if not 0 < peg_unplug_progress_gamma <= 1:
            raise ValueError("peg_unplug_progress_gamma must be in (0, 1]")
        if peg_unplug_reward_variant == "original" and peg_unplug_progress_weight != 0:
            raise ValueError(
                "original Peg Unplug reward requires peg_unplug_progress_weight=0"
            )
        if (
            peg_unplug_reward_variant == "peg_unplug_progress_v1"
            and peg_unplug_progress_weight <= 0
        ):
            raise ValueError(
                "peg_unplug_progress_v1 requires a positive progress weight"
            )
        self.peg_unplug_reward_variant = peg_unplug_reward_variant
        self.peg_unplug_progress_weight = float(peg_unplug_progress_weight)
        self.peg_unplug_progress_gamma = float(peg_unplug_progress_gamma)
        self._previous_extraction_potential = 0.0
        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([-0.19, 0.60, 0.0]),
            "hand_init_pos": np.array([0.05, 0.60, 0.25]),
        }
        self.goal = np.array([0.004, 0.60, 0.131])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(goal_low, goal_high, dtype=np.float64)

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place,
            grasp_success,
        ) = self.compute_reward(action, obs)
        progress_potential = self._extraction_progress_potential(obs[4:7])
        shaping_reward = self._progress_shaping_reward(
            self._previous_extraction_potential, progress_potential
        )
        self._previous_extraction_potential = progress_potential
        reward += shaping_reward
        return reward, {
            "success": float(obj_to_target <= 0.07),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "extraction_progress_potential": progress_potential,
            "progress_shaping_reward": shaping_reward,
        }

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("pegEnd")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("plug1").xquat.copy()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        joint = self.data.joint("objjoint")
        joint.qpos[:3] = pos
        joint.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        joint.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self) -> npt.NDArray[np.float64]:
        actuator_id = self.model.actuator("fingers_actuator").id
        self.model.actuator_forcerange[actuator_id] = (-5.0, 5.0)
        self._reset_hand()
        # The side pull must overcome socket friction. Keep this adjustment
        # local to the task, after the standard open-gripper reset, rather
        # than changing the shared Robotiq model used by existing tasks.
        self.model.actuator_forcerange[actuator_id] = (-60.0, 60.0)
        pos_box = self._get_state_rand_vec()
        self.model.body("box").pos = pos_box
        pos_plug = pos_box + np.array([0.044, 0.0, 0.131])
        self._set_obj_xyz(pos_plug)
        self.obj_init_pos = self._get_pos_objects().copy()
        self._target_pos = pos_plug + np.array([0.15, 0.0, 0.0])
        self.model.site("goal").pos = self._target_pos
        self.maxPlacingDist = np.linalg.norm(self._target_pos - self.obj_init_pos)
        self._previous_extraction_potential = self._extraction_progress_potential(
            self._get_pos_objects()
        )
        return self._get_obs()

    def _extraction_progress_potential(self, peg_position: npt.ArrayLike) -> float:
        """Return bounded +X extraction progress for the active task instance."""
        assert self._target_pos is not None and self.obj_init_pos is not None
        denominator = float(self._target_pos[0] - self.obj_init_pos[0])
        if abs(denominator) <= np.finfo(float).eps:
            return 0.0
        progress = (
            float(np.asarray(peg_position)[0]) - self.obj_init_pos[0]
        ) / denominator
        return float(np.clip(progress, 0.0, 1.0))

    def _progress_shaping_reward(self, previous: float, current: float) -> float:
        if self.peg_unplug_reward_variant == "original":
            return 0.0
        return self.peg_unplug_progress_weight * (
            self.peg_unplug_progress_gamma * current - previous
        )

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version != "v2":
            raise ValueError("KUKA V3 tasks support reward_function_version='v2' only")
        obj, tcp_opened = obs[4:7], float(obs[3])
        tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
        obj_to_target = float(np.linalg.norm(obj - self._target_pos))
        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.025,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            desired_gripper_effort=0.8,
            high_density=True,
        )
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.05),
            margin=float(np.linalg.norm(self.obj_init_pos - self._target_pos)),
            sigmoid="long_tail",
        )
        grasp_success = tcp_opened > 0.5 and obj[0] - self.obj_init_pos[0] > 0.015
        reward = 2 * object_grasped
        if grasp_success and tcp_to_obj < 0.035:
            reward = 1 + 2 * object_grasped + 5 * in_place
        if obj_to_target <= 0.05:
            reward = 10.0
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
            float(grasp_success),
        )
