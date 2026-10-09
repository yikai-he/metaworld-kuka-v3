"""Keyboard teleoperation for registered KUKA-v3 + Robotiq 2F-85 tasks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Make ``python scripts/manual_control_kuka.py`` work from a source checkout
# without requiring an editable install or a caller-supplied PYTHONPATH.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import glfw
import gymnasium as gym
import mujoco
import numpy as np

import metaworld_kuka  # noqa: F401  Registers the KUKA Gymnasium entry point.
from metaworld.wrappers import RandomTaskSelectWrapper
from metaworld_kuka.registration import KUKA_ENVIRONMENTS, KUKA_MT1_ID


CONTROLS = """
Keyboard controls (focus the MuJoCo window):
  W / S       +Y / -Y
  A / D       -X / +X
  R / F       +Z / -Z
  C / O       close / open gripper (persistent)
  Space       hold Cartesian position
  Backspace   reset episode
  + / -       increase / decrease movement speed
  H           show / hide MuJoCo help overlay
  I           show / hide custom HUD
  V           toggle verbose terminal diagnostics
  G           show / hide push-object marker
  1           default camera view
  2           top-down task view
  3           side view
  4           front view
  5           close-up end-effector/task view
  X or Q      quit
Mouse controls provided by the MuJoCo viewer remain available for the camera.
""".strip()

MANUAL_KEYS = {
    glfw.KEY_W,
    glfw.KEY_S,
    glfw.KEY_A,
    glfw.KEY_D,
    glfw.KEY_R,
    glfw.KEY_F,
    glfw.KEY_C,
    glfw.KEY_O,
    glfw.KEY_SPACE,
    glfw.KEY_BACKSPACE,
    glfw.KEY_EQUAL,
    glfw.KEY_KP_ADD,
    glfw.KEY_MINUS,
    glfw.KEY_KP_SUBTRACT,
    glfw.KEY_Q,
    glfw.KEY_X,
    glfw.KEY_H,
    glfw.KEY_I,
    glfw.KEY_V,
    glfw.KEY_G,
    glfw.KEY_1,
    glfw.KEY_2,
    glfw.KEY_3,
    glfw.KEY_4,
    glfw.KEY_5,
}

HUD_CONTROLS_LEFT = "W/S\nA/D\nR/F\nC\nO\nSpace\n1\n2\n3\n4\n5\nG\nH\nI\nV\nQ"
HUD_CONTROLS_RIGHT = (
    "+Y / -Y\n-X / +X\n+Z / -Z\nClose\nOpen\nHold\nDefault view\n"
    "Top-down view\nSide view\nFront view\nClose-up view\n"
    "Object marker\nMuJoCo menu\nHUD\nVerbose\nQuit"
)

CAMERA_PRESET_NAMES = {
    glfw.KEY_1: "Default",
    glfw.KEY_2: "Top-down task",
    glfw.KEY_3: "Side",
    glfw.KEY_4: "Front",
    glfw.KEY_5: "Close-up task/TCP",
}


def find_task_wrapper(env: gym.Env) -> RandomTaskSelectWrapper | None:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, RandomTaskSelectWrapper):
            return current
        current = current.env
    return None


def configure_fixed_task(env: gym.Env, task_index: int | None) -> None:
    if task_index is None:
        return
    wrapper = find_task_wrapper(env)
    if wrapper is None:
        raise RuntimeError("Registered environment has no task-bank wrapper")
    if not 0 <= task_index < len(wrapper.tasks):
        raise ValueError(
            f"task index must be in [0, {len(wrapper.tasks) - 1}], got {task_index}"
        )
    wrapper.unwrapped.set_task(wrapper.tasks[task_index])
    wrapper.toggle_sample_tasks_on_reset(False)


def make_env(args: argparse.Namespace, *, interactive: bool) -> gym.Env:
    env = gym.make(
        KUKA_MT1_ID,
        env_name=args.task,
        seed=args.task_set_seed,
        reward_function_version="v2",
        render_mode="human" if interactive else None,
        camera_name=args.camera_name if interactive else None,
        terminate_on_success=False,
        disable_env_checker=True,
    )
    configure_fixed_task(env, args.task_index)
    return env


def object_position(base: Any) -> np.ndarray | None:
    try:
        value = np.asarray(base._get_pos_objects(), dtype=float).reshape(-1)
        return value[:3].copy() if value.size >= 3 else None
    except (AttributeError, KeyError, NotImplementedError, ValueError):
        return None


def camera_focus(base: Any, *, close_up: bool = False) -> np.ndarray:
    """Return a live task-space point suitable for a free-camera look-at."""
    tcp = np.asarray(base.tcp_center, dtype=float)[:3]
    goal = np.asarray(base._target_pos, dtype=float).reshape(-1)[:3]
    obj = object_position(base)
    points = [goal, tcp] if obj is None else [obj, goal, tcp]
    if close_up:
        return np.mean(points, axis=0)
    stacked = np.vstack(points)
    return (stacked.min(axis=0) + stacked.max(axis=0)) / 2.0


def apply_camera_preset(
    viewer: Any, base: Any, key: int, default_camera: dict[str, Any]
) -> str:
    """Apply a preset through the existing viewer camera controls."""
    cam = viewer.cam
    name = CAMERA_PRESET_NAMES[key]
    if key == glfw.KEY_1:
        cam.type = default_camera["type"]
        cam.fixedcamid = default_camera["fixedcamid"]
        cam.trackbodyid = default_camera["trackbodyid"]
        cam.lookat[:] = default_camera["lookat"]
        cam.distance = default_camera["distance"]
        cam.azimuth = default_camera["azimuth"]
        cam.elevation = default_camera["elevation"]
        return name

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = camera_focus(base, close_up=key == glfw.KEY_5)
    if key == glfw.KEY_2:
        cam.distance, cam.azimuth, cam.elevation = 1.15, 90.0, -90.0
    elif key == glfw.KEY_3:
        cam.distance, cam.azimuth, cam.elevation = 1.10, 0.0, -18.0
    elif key == glfw.KEY_4:
        cam.distance, cam.azimuth, cam.elevation = 1.10, 90.0, -18.0
    else:
        cam.distance, cam.azimuth, cam.elevation = 0.48, 135.0, -28.0
    return name


def print_reset(
    base: Any, obs: np.ndarray, task_index: int | None, *, verbose: bool
) -> None:
    suffix = f" | task_index={task_index}" if task_index is not None else ""
    print(f"RESET | tcp={np.array2string(base.tcp_center, precision=3)} "
          f"| aperture={obs[3]:.3f}{suffix}")
    if not verbose:
        return
    rand_vec = getattr(base, "_last_rand_vec", None)
    if rand_vec is not None:
        print(f"  task configuration: {np.array2string(np.asarray(rand_vec), precision=4)}")
    obj = object_position(base)
    if obj is not None:
        print(f"  object: {np.array2string(obj, precision=4)}")
    print(f"  goal:   {np.array2string(np.asarray(base._target_pos), precision=4)}")


def validate_step(obs: np.ndarray, reward: float, action: np.ndarray) -> None:
    if obs.shape != (39,):
        raise RuntimeError(f"Expected 39D observation, got {obs.shape}")
    if action.shape != (4,):
        raise RuntimeError(f"Expected 4D action, got {action.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(reward):
        raise RuntimeError("Environment produced NaN or Inf")


def rollover_manual_bookkeeping(env: gym.Env) -> bool:
    """Roll over only MetaWorld's path counter, returning whether it changed."""
    base = env.unwrapped
    path_length = getattr(base, "curr_path_length", None)
    max_path_length = getattr(base, "max_path_length", None)
    if path_length is None or max_path_length is None:
        raise RuntimeError("MetaWorld path-length bookkeeping is unavailable")
    if path_length >= max_path_length:
        base.curr_path_length = 0
        return True
    return False


def step_manual(
    env: gym.Env, action: np.ndarray
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Step without resetting physics when MetaWorld's internal horizon expires.

    Registered wrappers and their formal truncation signals remain intact. The
    base environment refuses the *next* step after ``curr_path_length`` reaches
    ``max_path_length``; rolling over only that counter keeps the diagnostic
    session alive without touching MuJoCo state, task state, or reward logic.
    """
    rollover_manual_bookkeeping(env)
    obs, reward, terminated, truncated, info = env.step(action)
    return np.asarray(obs), float(reward), terminated, truncated, info


def print_diagnostics(
    task: str,
    base: Any,
    timestep: int,
    reward: float,
    info: dict[str, Any],
    obs: np.ndarray,
    action: np.ndarray,
    initial_object_position: np.ndarray | None,
    verbose: bool,
) -> None:
    obj = object_position(base)
    fields = [
        f"t={timestep}",
        f"reward={reward:.3f}",
        f"success={int(bool(info.get('success', False)))}",
        f"tcp={np.array2string(base.tcp_center, precision=3)}",
        f"aperture={obs[3]:.2f}",
    ]
    if task == "kuka-push-v3" and obj is not None:
        contact = base.pad_contacting_object(base._get_id_main_object())
        contact_gate = contact and bool(obs[3] > 0.0)
        displacement = (
            float(np.linalg.norm(obj - initial_object_position))
            if initial_object_position is not None
            else float("nan")
        )
        fields.extend(
            (
                f"contact={int(contact)}",
                f"goal_dist={np.linalg.norm(obj - base._target_pos):.3f}",
                f"displacement={displacement:.3f}",
            )
        )
        if verbose:
            fields.extend(
                (
                    f"object={np.array2string(obj, precision=3)}",
                    f"goal={np.array2string(np.asarray(base._target_pos), precision=3)}",
                    f"tcp_to_obj={np.linalg.norm(base.tcp_center - obj):.4f}",
                    f"contact_gate={int(contact_gate)}",
                )
            )
    elif verbose:
        if obj is not None:
            fields.append(f"object={np.array2string(obj, precision=3)}")
        fields.append(f"goal={np.array2string(np.asarray(base._target_pos), precision=3)}")
    if verbose:
        fields.extend(
            (
                f"grip_cmd={action[3]:+.1f}",
                f"action={np.array2string(action, precision=2)}",
            )
        )
    print(" | ".join(fields))


def hud_status(
    task: str,
    base: Any,
    timestep: int,
    reward: float,
    success: bool,
    obs: np.ndarray,
    action: np.ndarray,
    initial_object_position: np.ndarray | None,
    speed: float,
    manual_horizon: int,
    control_hz: float,
    training_horizon: int | None,
    camera_preset: str,
    verbose: bool,
) -> tuple[str, str]:
    """Build the two compact text columns consumed by MuJoCo's overlay API."""
    horizon = str(manual_horizon) if manual_horizon else "unlimited"
    labels = [
        "Task",
        "Step",
        "Reward",
        "Success",
        "TCP",
        "Aperture",
        "Speed",
        "Control Hz",
        "Camera",
    ]
    values = [
        task,
        f"{timestep} / {horizon}",
        f"{reward:.3f}",
        "YES" if success else "NO",
        np.array2string(np.asarray(base.tcp_center), precision=3),
        f"{obs[3]:.2f}",
        f"{speed:.2f}",
        f"{control_hz:g}",
        camera_preset,
    ]
    if training_horizon is not None and timestep >= training_horizon:
        labels.append("Training horizon")
        values.append("PASSED")
    obj = object_position(base)
    if task == "kuka-push-v3" and obj is not None:
        contact = base.pad_contacting_object(base._get_id_main_object())
        gate_active = contact and bool(obs[3] > 0.0)
        displacement = (
            float(np.linalg.norm(obj - initial_object_position))
            if initial_object_position is not None
            else float("nan")
        )
        labels.extend(("Pad contact", "Goal distance", "Displacement", "Contact gate"))
        values.extend(
            (
                "YES" if contact else "NO",
                f"{np.linalg.norm(obj - base._target_pos):.3f}",
                f"{displacement:.3f}",
                "ACTIVE" if gate_active else "INACTIVE",
            )
        )
        labels.extend(("Object", "Goal"))
        values.extend(("red puck", "green marker"))
        if verbose:
            object_goal = np.asarray(base._target_pos) - obj
            labels.extend(
                (
                    "Action",
                    "Grip command",
                    "TCP to object",
                    "Object pos",
                    "Goal pos",
                    "Object -> goal",
                )
            )
            values.extend(
                (
                    np.array2string(action, precision=2),
                    f"{action[3]:+.1f}",
                    f"{np.linalg.norm(base.tcp_center - obj):.4f}",
                    np.array2string(obj, precision=3),
                    np.array2string(np.asarray(base._target_pos), precision=3),
                    np.array2string(object_goal, precision=3),
                )
            )
    elif verbose:
        labels.extend(("Action", "Grip command"))
        values.extend((np.array2string(action, precision=2), f"{action[3]:+.1f}"))
        if obj is not None:
            labels.append("Object")
            values.append(np.array2string(obj, precision=3))
        labels.append("Goal")
        values.append(np.array2string(np.asarray(base._target_pos), precision=3))
    return "\n".join(labels), "\n".join(values)


def reset(env: gym.Env, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    obs, info = env.reset()
    obs = np.asarray(obs)
    validate_step(obs, 0.0, np.zeros(4, dtype=np.float32))
    print_reset(env.unwrapped, obs, args.task_index, verbose=args.verbose)
    return obs, info


def smoke_test(args: argparse.Namespace) -> None:
    env = make_env(args, interactive=False)
    try:
        obs, _ = reset(env, args)
        action = np.array([0.05, -0.05, 0.0, -1.0], dtype=np.float32)
        for step in range(1, args.smoke_test + 1):
            obs, reward, _, _, _ = step_manual(env, action)
            validate_step(np.asarray(obs), float(reward), action)
            if args.manual_horizon and step % args.manual_horizon == 0:
                obs, _ = reset(env, args)
        print(
            f"SMOKE TEST PASSED: task={args.task}, steps={args.smoke_test}, "
            f"observation={np.asarray(obs).shape}, action={action.shape}"
        )
    finally:
        env.close()


def pressed(window: Any, key: int) -> bool:
    return glfw.get_key(window, key) == glfw.PRESS


def interactive(args: argparse.Namespace) -> None:
    env = make_env(args, interactive=True)
    try:
        obs, _ = reset(env, args)
        print(f"\nTask: {args.task} | task-bank seed: {args.task_set_seed}")
        print(CONTROLS)
        env.render()  # Construct the existing Gymnasium MuJoCo WindowViewer.
        viewer = env.unwrapped.mujoco_renderer.viewer
        if viewer is None or not hasattr(viewer, "window") or viewer.window is None:
            raise RuntimeError("MuJoCo human viewer did not expose a GLFW window")
        window = viewer.window
        default_camera = {
            "type": viewer.cam.type,
            "fixedcamid": viewer.cam.fixedcamid,
            "trackbodyid": viewer.cam.trackbodyid,
            "lookat": np.asarray(viewer.cam.lookat, dtype=float).copy(),
            "distance": float(viewer.cam.distance),
            "azimuth": float(viewer.cam.azimuth),
            "elevation": float(viewer.cam.elevation),
        }

        if not all(
            hasattr(viewer, attribute)
            for attribute in ("_hide_menu", "_create_overlay", "add_overlay")
        ):
            raise RuntimeError("MuJoCo human viewer does not support custom overlays")

        speed = args.speed
        gripper = -1.0
        verbose = args.verbose
        timestep = 0
        reward = 0.0
        success = False
        action = np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32)
        initial_object_position = object_position(env.unwrapped)
        training_horizon = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        show_default_overlay = False
        hud_visible = True
        camera_preset = CAMERA_PRESET_NAMES[glfw.KEY_1]
        object_marker_visible = args.task == "kuka-push-v3"

        def render_manual() -> None:
            """Render with a visual-only copy of the hidden physical push puck."""
            if object_marker_visible and args.task == "kuka-push-v3":
                geom = env.unwrapped.data.geom("objGeom")
                geom_id = geom.id
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=env.unwrapped.model.geom_size[geom_id].copy(),
                    pos=geom.xpos.copy(),
                    mat=geom.xmat.copy(),
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                )
            env.render()

        # The renderer's _hide_menu flag suppresses every overlay, including
        # custom entries. Keep its drawing pass enabled and decide here whether
        # to populate the original menu independently from the manual HUD.
        viewer._hide_menu = False
        create_default_overlay = viewer._create_overlay

        def create_manual_overlay() -> None:
            if show_default_overlay:
                create_default_overlay()
            if not hud_visible:
                return
            status_left, status_right = hud_status(
                args.task,
                env.unwrapped,
                timestep,
                reward,
                success,
                obs,
                action,
                initial_object_position,
                speed,
                args.manual_horizon,
                args.control_hz,
                training_horizon,
                camera_preset,
                verbose,
            )
            viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPLEFT, status_left, status_right
            )
            viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                HUD_CONTROLS_LEFT,
                HUD_CONTROLS_RIGHT,
            )

        viewer._create_overlay = create_manual_overlay

        # Prevent the viewer's C/R/S/F/D/Space shortcuts from changing display or
        # playback state while those keys are being used for teleoperation. All
        # non-manual viewer shortcuts continue through the viewer callback.
        viewer_key_callback = viewer._key_callback

        def filtered_viewer_key_callback(
            callback_window: Any,
            key: int,
            scancode: int,
            key_action: int,
            mods: int,
        ) -> None:
            if key not in MANUAL_KEYS:
                viewer_key_callback(
                    callback_window, key, scancode, key_action, mods
                )

        glfw.set_key_callback(window, filtered_viewer_key_callback)
        glfw.maximize_window(window)

        previous: dict[int, bool] = {}
        success_seen = False
        training_horizon_reported = False
        termination_reported = False
        control_period = 1.0 / args.control_hz
        next_step_at = time.monotonic()
        while window is not None and not glfw.window_should_close(window):
            keys = {
                key: pressed(window, key)
                for key in (
                    glfw.KEY_C,
                    glfw.KEY_O,
                    glfw.KEY_BACKSPACE,
                    glfw.KEY_EQUAL,
                    glfw.KEY_KP_ADD,
                    glfw.KEY_MINUS,
                    glfw.KEY_KP_SUBTRACT,
                    glfw.KEY_Q,
                    glfw.KEY_X,
                    glfw.KEY_H,
                    glfw.KEY_I,
                    glfw.KEY_V,
                    glfw.KEY_G,
                    glfw.KEY_1,
                    glfw.KEY_2,
                    glfw.KEY_3,
                    glfw.KEY_4,
                    glfw.KEY_5,
                )
            }
            rising = lambda key: keys[key] and not previous.get(key, False)
            if rising(glfw.KEY_Q) or rising(glfw.KEY_X):
                break
            if rising(glfw.KEY_C):
                gripper = 1.0
                print("Gripper command: CLOSE (+1)")
            if rising(glfw.KEY_O):
                gripper = -1.0
                print("Gripper command: OPEN (-1)")
            if rising(glfw.KEY_H):
                show_default_overlay = not show_default_overlay
                print(
                    f"MuJoCo debug overlay: "
                    f"{'ON' if show_default_overlay else 'OFF'}"
                )
            if rising(glfw.KEY_I):
                hud_visible = not hud_visible
                print(f"Manual-control HUD: {'ON' if hud_visible else 'OFF'}")
            if rising(glfw.KEY_V):
                verbose = not verbose
                args.verbose = verbose
                print(f"Verbose diagnostics: {'ON' if verbose else 'OFF'}")
            if rising(glfw.KEY_G) and args.task == "kuka-push-v3":
                object_marker_visible = not object_marker_visible
                print(
                    f"Push-object marker: "
                    f"{'ON' if object_marker_visible else 'OFF'}"
                )
            for camera_key in CAMERA_PRESET_NAMES:
                if rising(camera_key):
                    camera_preset = apply_camera_preset(
                        viewer, env.unwrapped, camera_key, default_camera
                    )
                    print(f"Camera view: {camera_preset}")
            if rising(glfw.KEY_EQUAL) or rising(glfw.KEY_KP_ADD):
                speed = min(1.0, speed + 0.05)
                print(f"Movement speed: {speed:.2f}")
            if rising(glfw.KEY_MINUS) or rising(glfw.KEY_KP_SUBTRACT):
                speed = max(0.05, speed - 0.05)
                print(f"Movement speed: {speed:.2f}")
            if rising(glfw.KEY_BACKSPACE):
                obs, _ = reset(env, args)
                timestep = 0
                reward = 0.0
                success = False
                action = np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32)
                initial_object_position = object_position(env.unwrapped)
                success_seen = False
                training_horizon_reported = False
                termination_reported = False
                render_manual()
                previous = keys
                next_step_at += control_period
                delay = next_step_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_step_at = time.monotonic()
                continue

            cartesian = np.zeros(3, dtype=np.float32)
            if not pressed(window, glfw.KEY_SPACE):
                cartesian[0] = float(pressed(window, glfw.KEY_D)) - float(
                    pressed(window, glfw.KEY_A)
                )
                cartesian[1] = float(pressed(window, glfw.KEY_W)) - float(
                    pressed(window, glfw.KEY_S)
                )
                cartesian[2] = float(pressed(window, glfw.KEY_R)) - float(
                    pressed(window, glfw.KEY_F)
                )
            action = np.r_[cartesian * speed, gripper].astype(np.float32)
            obs, reward, terminated, truncated, info = step_manual(env, action)
            validate_step(obs, reward, action)
            timestep += 1
            success = bool(info.get("success", False))
            just_succeeded = success and not success_seen
            if just_succeeded:
                print("\n*** SUCCESS ***\n")
            success_seen = success
            if timestep % args.print_every == 0 or just_succeeded:
                print_diagnostics(
                    args.task,
                    env.unwrapped,
                    timestep,
                    reward,
                    info,
                    obs,
                    action,
                    initial_object_position,
                    verbose,
                )
            render_manual()
            if truncated and not training_horizon_reported:
                print("Training horizon passed; continuing manual control.")
                training_horizon_reported = True
            if terminated and not termination_reported:
                print("Environment reported termination; continuing manual inspection.")
                termination_reported = True
            if args.manual_horizon and timestep >= args.manual_horizon:
                print(f"Manual horizon ({args.manual_horizon}) reached; resetting.")
                obs, _ = reset(env, args)
                timestep = 0
                reward = 0.0
                success = False
                action = np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32)
                initial_object_position = object_position(env.unwrapped)
                success_seen = False
                training_horizon_reported = False
                termination_reported = False
                render_manual()
            previous = keys
            next_step_at += control_period
            delay = next_step_at - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_step_at = time.monotonic()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(KUKA_ENVIRONMENTS), default="kuka-push-v3")
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="manual action frequency in Hz (default: 20)",
    )
    parser.add_argument(
        "--manual-horizon",
        type=int,
        default=2000,
        metavar="STEPS",
        help="reset after STEPS manual actions; 0 means unlimited (default: 2000)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        metavar="STEPS",
        help="print status every STEPS environment steps (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include full action, object, goal, and push-gate diagnostics",
    )
    parser.add_argument("--task-set-seed", type=int, default=10000)
    parser.add_argument("--task-index", type=int, choices=range(50))
    parser.add_argument("--camera-name", default="corner")
    parser.add_argument(
        "--smoke-test",
        type=int,
        metavar="STEPS",
        help="run STEPS without a viewer, validate the API, and exit",
    )
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be in (0, 1]")
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.manual_horizon < 0:
        parser.error("--manual-horizon must be non-negative")
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    if args.smoke_test is not None and args.smoke_test < 1:
        parser.error("--smoke-test must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke_test is not None:
        smoke_test(args)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
