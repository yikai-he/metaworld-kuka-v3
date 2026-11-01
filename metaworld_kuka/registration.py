"""Isolated Gymnasium registration for KUKA MetaWorld tasks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from metaworld import _init_each_env, _make_tasks

from metaworld_kuka.envs import (
    KukaButtonPressEnvV3,
    KukaButtonPressTopdownEnvV3,
    KukaDoorOpenEnvV3,
    KukaDrawerCloseEnvV3,
    KukaFaucetCloseEnvV3,
    KukaFaucetOpenEnvV3,
    KukaHammerEnvV3,
    KukaHandlePressSideEnvV3,
    KukaPegUnplugSideEnvV3,
    KukaPickPlaceEnvV3,
    KukaPushEnvV3,
    KukaPushCubeEnvV3,
    KukaReachEnvV3,
    KukaWindowCloseEnvV3,
    KukaWindowOpenEnvV3,
)

KUKA_MT1_ID = "Meta-World-KUKA/MT1"
KUKA_ENVIRONMENTS = OrderedDict(
    (
        ("kuka-reach-v3", KukaReachEnvV3),
        ("kuka-push-v3", KukaPushEnvV3),
        ("kuka-push-cube-v3", KukaPushCubeEnvV3),
        ("kuka-handle-press-side-v3", KukaHandlePressSideEnvV3),
        ("kuka-button-press-v3", KukaButtonPressEnvV3),
        ("kuka-button-press-topdown-v3", KukaButtonPressTopdownEnvV3),
        ("kuka-peg-unplug-side-v3", KukaPegUnplugSideEnvV3),
        ("kuka-hammer-v3", KukaHammerEnvV3),
        ("kuka-drawer-close-v3", KukaDrawerCloseEnvV3),
        ("kuka-faucet-open-v3", KukaFaucetOpenEnvV3),
        ("kuka-faucet-close-v3", KukaFaucetCloseEnvV3),
        ("kuka-window-open-v3", KukaWindowOpenEnvV3),
        ("kuka-window-close-v3", KukaWindowCloseEnvV3),
        ("kuka-door-open-v3", KukaDoorOpenEnvV3),
        ("kuka-pick-place-v3", KukaPickPlaceEnvV3),
    )
)


class Float32ObservationWrapper(gym.ObservationWrapper):
    """Expose MetaWorld's physical observation using the thesis float32 contract."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or space.shape != (39,):
            raise ValueError("KUKA MetaWorld environments must expose a 39D Box")
        self.observation_space = gym.spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


def make_kuka_mt1(
    env_name: str,
    seed: int | None = None,
    use_one_hot: bool = False,
    terminate_on_success: bool = False,
    reward_function_version: Literal["v1", "v2"] = "v2",
    hammer_reward_variant: Literal["original", "nail_progress"] = "original",
    hammer_nail_progress_weight: float = 0.0,
    peg_unplug_reward_variant: Literal[
        "original", "peg_unplug_progress_v1"
    ] = "original",
    peg_unplug_progress_weight: float = 0.0,
    peg_unplug_progress_gamma: float = 0.99,
    task_select: Literal["random", "pseudorandom"] = "random",
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
    camera_name: str | None = None,
    camera_id: int | None = None,
    width: int = 480,
    height: int = 480,
    **kwargs: Any,
) -> gym.Env:
    """Construct one KUKA task with MetaWorld's MT1 task and wrapper lifecycle."""
    if env_name not in KUKA_ENVIRONMENTS:
        raise ValueError(
            f"Unknown KUKA V3 task {env_name!r}; expected one of "
            f"{tuple(KUKA_ENVIRONMENTS)}"
        )
    if kwargs:
        raise TypeError(f"Unsupported KUKA MT1 options: {tuple(sorted(kwargs))}")
    if reward_function_version not in ("v1", "v2"):
        raise ValueError(
            f"Unsupported reward_function_version={reward_function_version!r}"
        )

    env_cls = KUKA_ENVIRONMENTS[env_name]
    args_kwargs = {env_name: {"args": [], "kwargs": {"task_id": 0}}}
    tasks = _make_tasks(
        KUKA_ENVIRONMENTS,
        args_kwargs,
        {"partially_observable": False},
        seed=seed,
    )
    if env_name != "kuka-hammer-v3" and (
        hammer_reward_variant != "original" or hammer_nail_progress_weight != 0.0
    ):
        raise ValueError("Hammer reward options are valid only for kuka-hammer-v3")
    if env_name != "kuka-peg-unplug-side-v3" and (
        peg_unplug_reward_variant != "original" or peg_unplug_progress_weight != 0.0
    ):
        raise ValueError(
            "Peg Unplug reward options are valid only for kuka-peg-unplug-side-v3"
        )
    if peg_unplug_reward_variant not in ("original", "peg_unplug_progress_v1"):
        raise ValueError(
            "peg_unplug_reward_variant must be 'original' or 'peg_unplug_progress_v1'"
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
        raise ValueError("peg_unplug_progress_v1 requires a positive progress weight")
    env = _init_each_env(
        env_cls=env_cls,
        tasks=tasks,
        seed=seed,
        terminate_on_success=terminate_on_success,
        use_one_hot=use_one_hot,
        env_id=0,
        num_tasks=1,
        task_select=task_select,
        reward_function_version=reward_function_version,
        render_mode=render_mode,
        camera_name=camera_name,
        camera_id=camera_id,
        width=width,
        height=height,
    )
    # Build the public observation space from the goal-observable task contract.
    # Without an initial task, Float32ObservationWrapper would cache the
    # pre-task, partially-observable zero goal bounds permanently.
    env.unwrapped.set_task(tasks[0])
    env.unwrapped.observation_space = env.unwrapped.sawyer_observation_space
    if env_name == "kuka-hammer-v3":
        raw_env = env.unwrapped
        raw_env.hammer_reward_variant = hammer_reward_variant
        raw_env.hammer_nail_progress_weight = float(hammer_nail_progress_weight)
    if env_name == "kuka-peg-unplug-side-v3":
        raw_env = env.unwrapped
        raw_env.peg_unplug_reward_variant = peg_unplug_reward_variant
        raw_env.peg_unplug_progress_weight = float(peg_unplug_progress_weight)
        raw_env.peg_unplug_progress_gamma = float(peg_unplug_progress_gamma)
    return Float32ObservationWrapper(env)


def register_kuka_envs() -> None:
    """Register the isolated KUKA MT1 entry point idempotently."""
    if KUKA_MT1_ID not in gym.envs.registry:
        gym.register(
            id=KUKA_MT1_ID,
            entry_point=make_kuka_mt1,
            max_episode_steps=200,
        )
