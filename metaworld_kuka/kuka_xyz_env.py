"""KUKA-specific robot behavior on MetaWorld 3's XYZ environment base."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from metaworld.sawyer_xyz_env import SawyerXYZEnv


class KukaXYZEnv(SawyerXYZEnv):
    """MetaWorld XYZ control with the KUKA IIWA's fixed wrist orientation.

    Gymnasium lifecycle, stepping, observations, rendering, task handling, and
    reward dispatch intentionally remain inherited from ``SawyerXYZEnv``.
    """

    MOCAP_QUATERNION = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    ROBOTIQ_OPEN_CONTROL = 0.0
    ROBOTIQ_CLOSED_CONTROL = 255.0
    ROBOTIQ_PAD_GEOMS = (
        "right_pad1",
        "right_pad2",
        "left_pad1",
        "left_pad2",
    )

    @staticmethod
    def robotiq_control(gripper_action: float) -> float:
        """Map policy effort continuously: -1=open, +1=closed."""
        return 127.5 * (float(np.clip(gripper_action, -1.0, 1.0)) + 1.0)

    def do_simulation(self, ctrl: npt.ArrayLike, n_frames: int) -> None:
        """Translate MetaWorld's two-finger command to the one 2F-85 actuator.

        The inherited step/reset paths supply ``[a, -a]`` for the synthetic
        Sawyer-style fingers. Only the first value is the policy command. The
        Robotiq model instead receives one continuous control in ``[0, 255]``.
        """
        command = np.asarray(ctrl, dtype=np.float64).reshape(-1)
        if command.size == 2 or command.size == 1 and -1.0 <= command[0] <= 1.0:
            command = np.array([self.robotiq_control(command[0])])
        super().do_simulation(command, n_frames)

    def get_endeff_pos(self) -> npt.NDArray[np.float64]:
        """Use the physical fingertip midpoint as the policy EE position."""
        return self.tcp_center.copy()

    @property
    def gripper_aperture(self) -> float:
        """Distance between the Robotiq inner fingertip contact surfaces."""
        return float(
            np.linalg.norm(
                self._get_site_pos("rightEndEffector")
                - self._get_site_pos("leftEndEffector")
            )
        )

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:
        """Build MetaWorld's unchanged 18D current-state block for 2F-85."""
        pos_hand = self.get_endeff_pos()
        # The physical 2F-85 inner-pad aperture is nominally 0--85 mm. Sites
        # are located on those surfaces, so this is physical rather than a
        # driver-angle proxy. Clip tolerates compliant/contact overtravel.
        normalized_opening = np.clip(self.gripper_aperture / 0.085, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return np.hstack((pos_hand, normalized_opening, obs_obj_padded))

    def get_body_com(self, body_name: str) -> npt.NDArray[np.float64]:
        """Keep caging rewards compatible while using Robotiq body names."""
        body_name = {"leftpad": "left_pad", "rightpad": "right_pad"}.get(
            body_name, body_name
        )
        return super().get_body_com(body_name)

    def touching_object(self, object_geom_id: int) -> bool:
        """Return true when both Robotiq pad sides exert force on an object."""
        right_ids = {self.data.geom(name).id for name in self.ROBOTIQ_PAD_GEOMS[:2]}
        left_ids = {self.data.geom(name).id for name in self.ROBOTIQ_PAD_GEOMS[2:]}
        side_force = {"right": 0.0, "left": 0.0}
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if object_geom_id not in pair or contact.efc_address < 0:
                continue
            force = float(self.data.efc_force[contact.efc_address])
            if pair & right_ids:
                side_force["right"] += force
            if pair & left_ids:
                side_force["left"] += force
        return side_force["right"] > 0.0 and side_force["left"] > 0.0

    def pad_contacting_object(self, object_geom_id: int) -> bool:
        """Return whether any Robotiq inner pad physically contacts an object.

        Bilateral ``touching_object`` remains the grasp/contact helper. Push
        rewards use this single-pad predicate because one finger face is the
        expected physical pusher for the closed 2F-85 geometry.
        """
        pad_ids = {self.data.geom(name).id for name in self.ROBOTIQ_PAD_GEOMS}
        return any(
            object_geom_id in (contact.geom1, contact.geom2)
            and bool({contact.geom1, contact.geom2} & pad_ids)
            for contact in self.data.contact
        )

    def robot_object_contact_diagnostics(
        self, object_geom_id: int
    ) -> tuple[bool, bool, bool]:
        """Classify post-step contacts between an object and the Robotiq gripper.

        The physical predicate includes every collidable geom attached to the
        Robotiq ``base`` body or one of its descendants, rather than only the four
        specially named pads. World/table contacts are therefore excluded.
        """
        gripper_root_id = self.model.body("base").id
        gripper_body_ids = {gripper_root_id}
        for body_id in range(self.model.nbody):
            parent = body_id
            while parent > 0:
                parent = int(self.model.body_parentid[parent])
                if parent == gripper_root_id:
                    gripper_body_ids.add(body_id)
                    break
        gripper_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in gripper_body_ids
            and int(self.model.geom_contype[geom_id]) != 0
        }
        pad_ids = {self.model.geom(name).id for name in self.ROBOTIQ_PAD_GEOMS}
        follower_body_ids = {
            self.model.body("left_follower").id,
            self.model.body("right_follower").id,
        }
        follower_geom_ids = {
            geom_id
            for geom_id in gripper_geom_ids
            if int(self.model.geom_bodyid[geom_id]) in follower_body_ids
        }
        physical = named_pad = follower = False
        for contact in self.data.contact:
            if object_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == object_geom_id else contact.geom1
            physical |= other in gripper_geom_ids
            named_pad |= other in pad_ids
            follower |= other in follower_geom_ids
        return bool(physical), bool(named_pad), bool(follower)

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Apply a clipped Cartesian delta while preserving KUKA orientation."""
        action = np.clip(action, -1.0, 1.0)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :], self.mocap_low, self.mocap_high
        )
        self.data.mocap_pos = new_mocap_pos
        self.data.mocap_quat = self.MOCAP_QUATERNION

    def _reset_hand(self, steps: int = 50) -> None:
        """Settle the KUKA hand at its task-defined initial Cartesian pose."""
        assert self.hand_init_pos is not None
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = self.MOCAP_QUATERNION
            # Explicit reset state: fully open Robotiq aperture.
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Move the standard task object's named MuJoCo free joint."""
        obj_joint = self.data.joint("objjoint")
        if obj_joint.qpos.shape != (7,) or obj_joint.qvel.shape != (6,):
            raise RuntimeError("objjoint must be a MuJoCo free joint")
        obj_joint.qpos[:3] = pos
        obj_joint.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
