"""Sawyer-inspired KUKA Push task using a 40 mm cube."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from metaworld.utils import reward_utils

from metaworld_kuka.envs.kuka_push_v3 import KukaPushEnvV3

_MODEL = Path(__file__).parents[1] / "assets" / "kuka_xyz" / "kuka_push_cube_v3.xml"


class KukaPushCubeEnvV3(KukaPushEnvV3):
    """Push a stable 40 mm cube to a sampled planar goal.

    The cube half-extent is 20 mm. Sawyer's scalar caging radius is represented
    by the cube's center-to-corner radius, while the X-Z tolerance includes the
    20 mm center-height offset between a table-supported cube and the Robotiq
    fingertip midpoint.
    """

    CUBE_HALF_EXTENT = 0.02
    OBJECT_RADIUS = float(np.sqrt(3.0) * CUBE_HALF_EXTENT)
    PAD_SUCCESS_THRESHOLD = 0.055
    XZ_ALIGNMENT_THRESHOLD = 0.025
    NEAR_OBJECT_RADIUS = 0.055

    @property
    def model_name(self) -> str:
        return str(_MODEL)

    def reset_model(self):
        observation = super().reset_model()
        # The baseline uses one documented upright orientation, not a hidden
        # curriculum or orientation randomization. Translation remains sampled.
        self.data.joint("objjoint").qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.joint("objjoint").qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_function_version != "v2":
            raise ValueError("KUKA Push-Cube supports reward_function_version='v2' only")
        obj = obs[4:7]
        tcp_opened = obs[3]
        tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
        target_to_obj = float(np.linalg.norm(obj - self._target_pos))
        initial_distance = float(np.linalg.norm(self.obj_init_pos - self._target_pos))
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=initial_distance,
            sigmoid="long_tail",
        )
        caging = self._robotiq_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=self.OBJECT_RADIUS,
            pad_success_thresh=self.PAD_SUCCESS_THRESHOLD,
            xz_thresh=self.XZ_ALIGNMENT_THRESHOLD,
            high_density=True,
        )
        reward = 2 * caging
        if tcp_to_obj < self.NEAR_OBJECT_RADIUS and tcp_opened > 0:
            reward += 1.0 + reward + 5.0 * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return reward, tcp_to_obj, tcp_opened, target_to_obj, caging, in_place
