"""KUKA robot extensions for MetaWorld 3."""

from metaworld_kuka.kuka_xyz_env import KukaXYZEnv
from metaworld_kuka.envs import (
    KukaButtonPressEnvV3,
    KukaHandlePressSideEnvV3,
    KukaHammerEnvV3,
    KukaPushEnvV3,
    KukaReachEnvV3,
)
from metaworld_kuka.registration import KUKA_MT1_ID, make_kuka_mt1, register_kuka_envs

register_kuka_envs()

__all__ = [
    "KUKA_MT1_ID",
    "KukaButtonPressEnvV3",
    "KukaHandlePressSideEnvV3",
    "KukaHammerEnvV3",
    "KukaPushEnvV3",
    "KukaReachEnvV3",
    "KukaXYZEnv",
    "make_kuka_mt1",
    "register_kuka_envs",
]
