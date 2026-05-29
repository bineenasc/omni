"""
Shared configuration for all robot controllers and the RL environment.

The supervisor reads robot type from this dict to build the observation space
and to encode the robot-type scalar that is fed into the neural network.

All physical constants must match the values declared in the corresponding
.proto files and in soccer.wbt.
"""

import os
import sys

# ── Robot configurations ──────────────────────────────────────────────────────
# Keys must match the Webots robot name (robot.getName() in a controller, or
# the 'name' field set in the world file).

ROBOT_CONFIGS = {
    "viper": {
        "type_id":        0,           # scalar fed into the observation
        "wheel_radius":   0.030,       # m  — matches Viper.proto HingeJoint endPoint Cylinder radius
        "wheel_distance": 0.138,       # m  — centre-of-robot to wheel anchor
        "max_velocity":   15.0,        # rad/s — safe clamp for RL actions
        "body_radius":    0.120,       # m  — used for collision / display
        "lidar_rays":     1440,
        "lidar_min_range": 0.02,       # m
        "lidar_max_range": 3.50,       # m
    },
    "titan": {
        "type_id":        1,
        # Bigger and slower than Viper — same device names (wheel1/2/3,
        # lidar, IMU, GPS) but different physical parameters.
        "wheel_radius":   0.035,       # m  — larger wheels
        "wheel_distance": 0.170,       # m  — wider stance (larger body)
        "max_velocity":   9.0,         # rad/s — ~0.315 m/s top speed (vs ~0.45 m/s Viper)
        "body_radius":    0.165,       # m
        "lidar_rays":     1440,
        "lidar_min_range": 0.02,
        "lidar_max_range": 3.50,
    },
}

# ── Field geometry (NUE, metres) — must match soccer.wbt ─────────────────────
# Coordinate system: X = width, Y = up, Z = length
# Ground plane is XZ.  Goals are on the ±Z ends.

FIELD = {
    "half_width":      3.70,    # ±X boundary of the green pitch
    "half_length":     5.20,    # ±Z boundary of the green pitch
    "goal_z_attack":   4.55,    # Z coordinate of the goal the agent attacks (+Z)
    "goal_z_own":     -4.55,    # Z coordinate of the agent's own goal (−Z)
    "goal_half_width": 0.75,    # ±X extent of a goal opening
    "goal_depth":      0.20,    # how far inside the goal counts as 'scored'
}

# ── Ball ──────────────────────────────────────────────────────────────────────
BALL = {
    "radius": 0.025,   # m — matches BOLA Sphere radius in soccer.wbt
    "mass":   0.055,   # kg
}

# ── Simulation timing ─────────────────────────────────────────────────────────
SIM = {
    "basic_time_step": 8,       # ms — must match WorldInfo.basicTimeStep in soccer.wbt
    "steps_per_action": 5,      # physics steps executed per RL action (= 40 ms / action)
    "max_episode_steps": 1000,  # episode timeout (1000 × 40 ms = 40 s)
    # ↑ Reduced from 2000: at MAX_LINEAR=0.5 m/s the robot can cross the full
    # 10.4 m field in ≈520 steps.  1000 steps gives a generous margin while
    # doubling the number of resets per epoch (≥30 episodes vs ≤15).
}


# ── IPC protocol (Emitter/Receiver between robot and supervisor) ──────────────
# Channel 0: supervisor → robot   (action packet: [vx, vz, omega])
# Channel 1: robot → supervisor   (sensor packet: [robot_id, lidar × 1440])
#
# Both sides must use the same format strings.  Defined once here to prevent
# desync bugs between robot_controller.py and soccer_supervisor.py.

import struct as _struct

IPC = {
    "action_channel":  0,
    "sensor_channel":  1,
    "action_fmt":      "3f",
    "action_bytes":    _struct.calcsize("3f"),         # 12 bytes
    "n_lidar":         1440,
    "sensor_fmt":      f"i{1440}f",
    "sensor_bytes":    _struct.calcsize(f"i{1440}f"), # 5764 bytes
}


def get_robot_config(name: str) -> dict:
    """Return config dict for a robot, falling back gracefully on unknown names."""
    if name not in ROBOT_CONFIGS:
        raise KeyError(
            f"Unknown robot name '{name}'. "
            f"Valid names: {list(ROBOT_CONFIGS.keys())}"
        )
    return ROBOT_CONFIGS[name]


def add_controllers_to_path() -> None:
    """
    Add the controllers/ directory to sys.path so that any controller can
    import other modules that live there (e.g. future shared utilities).

    IMPORTANT — bootstrap order:
    This function lives inside shared_configs, so it cannot be used to import
    shared_configs itself.  Each controller must do the path manipulation
    BEFORE the first `import shared_configs` line:

        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from shared_configs import ROBOT_CONFIGS, FIELD, BALL, SIM, get_robot_config

    After that one-time bootstrap, call add_controllers_to_path() only if you
    need to import additional modules from controllers/.
    """
    controllers_dir = os.path.dirname(os.path.abspath(__file__))
    if controllers_dir not in sys.path:
        sys.path.insert(0, controllers_dir)
