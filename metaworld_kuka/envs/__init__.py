"""KUKA task environments for MetaWorld 3."""

from metaworld_kuka.envs.kuka_handle_press_side_v3 import KukaHandlePressSideEnvV3
from metaworld_kuka.envs.kuka_drawer_close_v3 import KukaDrawerCloseEnvV3
from metaworld_kuka.envs.kuka_faucet_open_v3 import KukaFaucetOpenEnvV3
from metaworld_kuka.envs.kuka_faucet_close_v3 import KukaFaucetCloseEnvV3
from metaworld_kuka.envs.kuka_window_open_v3 import KukaWindowOpenEnvV3
from metaworld_kuka.envs.kuka_window_close_v3 import KukaWindowCloseEnvV3
from metaworld_kuka.envs.kuka_door_open_v3 import KukaDoorOpenEnvV3
from metaworld_kuka.envs.kuka_pick_place_v3 import KukaPickPlaceEnvV3
from metaworld_kuka.envs.kuka_hammer_v3 import KukaHammerEnvV3
from metaworld_kuka.envs.kuka_reach_v3 import KukaReachEnvV3
from metaworld_kuka.envs.kuka_push_v3 import KukaPushEnvV3
from metaworld_kuka.envs.kuka_push_cube_v3 import KukaPushCubeEnvV3
from metaworld_kuka.envs.kuka_button_press_v3 import KukaButtonPressEnvV3
from metaworld_kuka.envs.kuka_button_press_topdown_v3 import KukaButtonPressTopdownEnvV3
from metaworld_kuka.envs.kuka_peg_unplug_side_v3 import KukaPegUnplugSideEnvV3

__all__ = [
    "KukaButtonPressEnvV3",
    "KukaButtonPressTopdownEnvV3",
    "KukaPegUnplugSideEnvV3",
    "KukaDrawerCloseEnvV3",
    "KukaFaucetOpenEnvV3",
    "KukaFaucetCloseEnvV3",
    "KukaWindowOpenEnvV3",
    "KukaWindowCloseEnvV3",
    "KukaDoorOpenEnvV3",
    "KukaPickPlaceEnvV3",
    "KukaHandlePressSideEnvV3",
    "KukaHammerEnvV3",
    "KukaPushEnvV3",
    "KukaPushCubeEnvV3",
    "KukaReachEnvV3",
]
