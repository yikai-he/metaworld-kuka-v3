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


class KukaWindowCloseEnvV3(KukaXYZEnv):
    """KUKA embodiment of the current Sawyer-v3 window-close task."""

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
        # REUSED_VALIDATED_KUKA_V3 (also HISTORICAL_KUKA): workspace,
        # fixture randomization, and initial poses.
        hand_low = (-0.25, 0.42, 0.05)
        hand_high = (0.25, 0.75, 0.5)
        obj_low = (-0.1, 0.70, 0.202)
        obj_high = (0.1, 0.85, 0.202)
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
            "obj_init_pos": np.array([-0.1, 0.735, 0.202], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.45, 0.3], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(obj_low), np.array(obj_high), dtype=np.float64)
        self.maxPullDist = 0.2
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

    @property
    def model_name(self) -> str:
        # REUSED_VALIDATED_KUKA_V3: exact Window Open model and mechanism.
        return str(
            Path(__file__).parents[1]
            / "assets"
            / "kuka_xyz"
            / "kuka_window_open_v3.xml"
        )

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_obj, _, target_to_obj, object_grasped, in_place = (
            self.compute_reward(action, obs)
        )
        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        # CURRENT_SAWYER: close-task observation site.
        return self._get_site_pos("handleCloseStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        self.obj_init_pos = self._get_state_rand_vec()
        # CURRENT_SAWYER: closed target at the randomized fixture origin and
        # an open initial slide position of 0.2 m.
        self._target_pos = self.obj_init_pos.copy()
        self.model.body("window").pos = self.obj_init_pos
        self.window_handle_pos_init = self._get_pos_objects() + np.array(
            [0.2, 0.0, 0.0]
        )
        self.data.joint("window_slide").qpos[0] = 0.2
        self.data.joint("window_slide").qvel[0] = 0.0
        self.model.site("goal").pos = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _reset_hand(self, steps: int = 50) -> None:
        super()._reset_hand(steps=steps)
        self.init_tcp = self.tcp_center

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None
        if self.reward_function_version == "v2":
            del actions, obs
            # CURRENT_SAWYER: unmodified Window Close v2 reward semantics.
            obj = self._get_pos_objects()
            tcp = self.tcp_center
            target_to_obj = float(abs(obj[0] - self._target_pos[0]))
            target_to_obj_init = float(
                abs(self.window_handle_pos_init[0] - self._target_pos[0])
            )
            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )
            handle_radius = 0.02
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(
                np.linalg.norm(self.window_handle_pos_init - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="gaussian",
            )
            reward = 10 * reward_utils.hamacher_product(reach, in_place)
            return reward, tcp_to_obj, 0.0, target_to_obj, reach, in_place

        del actions
        obj_pos = obs[4:7]
        finger_com = (
            self._get_site_pos("rightEndEffector")
            + self._get_site_pos("leftEndEffector")
        ) / 2
        pull_dist = float(abs(obj_pos[0] - self._target_pos[0]))
        reach_dist = float(np.linalg.norm(obj_pos - finger_com))
        self.reachCompleted = reach_dist < 0.05
        pull_reward = 0.0
        if self.reachCompleted:
            pull_reward = 1000 * (self.maxPullDist - pull_dist) + 1000 * (
                np.exp(-(pull_dist**2) / 0.01) + np.exp(-(pull_dist**2) / 0.001)
            )
        return -reach_dist + pull_reward, 0.0, 0.0, pull_dist, 0.0, 0.0
