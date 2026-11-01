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


class KukaFaucetCloseEnvV3(KukaXYZEnv):
    """KUKA embodiment of the current Sawyer-v3 faucet-close task."""

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
        # fixture randomization, handle length, and initial poses.
        hand_low = (-0.25, 0.42, 0.05)
        hand_high = (0.25, 0.75, 0.5)
        obj_low = (-0.1, 0.65, 0.05)
        obj_high = (0.1, 0.70, 0.05)
        self._handle_length = 0.175
        # CURRENT_SAWYER: reward and success radius.
        self._target_radius: float = 0.07

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
            "obj_init_pos": np.array([0, 0.65, 0.05]),
            "hand_init_pos": np.array([0.0, 0.45, 0.2]),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        target_offset = np.array([-self._handle_length, 0.0, 0.125])
        self.goal_space = Box(
            np.array(obj_low) + target_offset,
            np.array(obj_high) + target_offset,
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        # REUSED_VALIDATED_KUKA_V3: exact Faucet Open model and mechanism.
        return str(
            Path(__file__).parents[1]
            / "assets"
            / "kuka_xyz"
            / "kuka_faucet_open_v3.xml"
        )

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_obj, _, target_to_obj, object_grasped, in_place = (
            self.compute_reward(action, obs)
        )
        info = {
            "success": float(target_to_obj <= 0.07),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }
        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert self._target_pos is not None
        return [
            ("goal_close", self._target_pos),
            ("goal_open", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("faucetBase").xquat

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        # CURRENT_SAWYER: close-task observation site.
        return self._get_site_pos("handleStartClose") + np.array([0.0, 0.0, -0.01])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("faucetBase").pos = self.obj_init_pos
        # CURRENT_SAWYER/HISTORICAL_KUKA: qpos=0 is the close task's open start.
        self.data.joint("knob_Joint_1").qpos[0] = 0.0
        self.data.joint("knob_Joint_1").qvel[0] = 0.0
        self._target_pos = self.obj_init_pos + np.array(
            [-self._handle_length, 0.0, 0.125]
        )
        mujoco.mj_forward(self.model, self.data)
        self.model.site("goal_close").pos = self._target_pos
        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)
        return self._get_obs()

    def _reset_hand(self, steps: int = 50) -> None:
        super()._reset_hand(steps=steps)
        self.reachCompleted = False

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            del action
            # CURRENT_SAWYER: unmodified Faucet Close v2 reward semantics.
            obj = obs[4:7]
            tcp = self.tcp_center
            target_to_obj = float(np.linalg.norm(obj - self._target_pos))
            target_to_obj_init = float(
                np.linalg.norm(self.obj_init_pos - self._target_pos)
            )
            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self._target_radius),
                margin=abs(target_to_obj_init - self._target_radius),
                sigmoid="long_tail",
            )
            faucet_reach_radius = 0.01
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(np.linalg.norm(self.obj_init_pos - self.init_tcp))
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, faucet_reach_radius),
                margin=abs(tcp_to_obj_init - faucet_reach_radius),
                sigmoid="gaussian",
            )
            reward = 2 * reach + 3 * in_place
            reward *= 2
            reward = 10 if target_to_obj <= self._target_radius else reward
            return reward, tcp_to_obj, 0.0, target_to_obj, reach, in_place

        del action
        obj_pos = obs[4:7]
        finger_com = (
            self._get_site_pos("rightEndEffector")
            + self._get_site_pos("leftEndEffector")
        ) / 2
        pull_dist = float(np.linalg.norm(obj_pos - self._target_pos))
        reach_dist = float(np.linalg.norm(obj_pos - finger_com))
        reach_reward = -reach_dist
        self.reachCompleted = reach_dist < 0.05
        pull_reward = 0.0
        if self.reachCompleted:
            pull_reward = 1000 * (self.maxPullDist - pull_dist) + 1000 * (
                np.exp(-(pull_dist**2) / 0.01) + np.exp(-(pull_dist**2) / 0.001)
            )
            pull_reward = max(pull_reward, 0.0)
        return reach_reward + pull_reward, 0.0, 0.0, pull_dist, 0.0, 0.0
