"""
Robot controller — works for both Viper and Titan.

Responsibilities each timestep:
  1. Read the Lidar scan.
  2. Pack and send [robot_id, lidar × 360] to the supervisor (channel 1).
  3. Receive the latest [vx, vz, omega] action from the supervisor (channel 0).
  4. Apply omniwheel inverse kinematics and set motor velocities.

The controller holds the last received action so the robot keeps moving
between supervisor commands (the supervisor sends one action per N sim steps).

IK (body frame: front = +X, lateral = +Z, NUE / Z-up internally):

  [w1]   1  [  1       0      L ] [vx  ]
  [w2] = -- [ -sin30   cos30  L ] [vz  ]
  [w3]   R  [ -sin30  -cos30  L ] [omega]
"""

import base64
import math
import os
import struct
import sys

# Bootstrap shared_configs path  (controllers/ is one level up from here)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared_configs import get_robot_config, IPC

from controller import Robot

# ── IK constants ──────────────────────────────────────────────────────────────
_S30 = math.sin(math.radians(30))  # 0.5
_C30 = math.cos(math.radians(30))  # ~0.866


def _ik(vx: float, vz: float, omega: float,
        R: float, L: float) -> tuple:
    """Return (w1, w2, w3) wheel speeds in rad/s."""
    return (
        ( 1.0  * vx + 0.0  * vz + L * omega) / R,
        (-_S30 * vx + _C30 * vz + L * omega) / R,
        (-_S30 * vx - _C30 * vz + L * omega) / R,
    )


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


# ── Initialise robot ──────────────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

cfg      = get_robot_config(robot.getName())
R        = cfg["wheel_radius"]
L        = cfg["wheel_distance"]
MAX_W    = cfg["max_velocity"]
N_LIDAR  = IPC["n_lidar"]          # 360
robot_id = cfg["type_id"]          # 0 = viper, 1 = titan

_SENSOR_FMT   = IPC["sensor_fmt"]    # "i360f"
_SENSOR_BYTES = IPC["sensor_bytes"]  # 1444
_ACTION_FMT   = IPC["action_fmt"]    # "3f"
_ACTION_BYTES = IPC["action_bytes"]  # 12

# ── Devices ───────────────────────────────────────────────────────────────────
m1 = robot.getDevice("wheel1")
m2 = robot.getDevice("wheel2")
m3 = robot.getDevice("wheel3")
for _m in (m1, m2, m3):
    _m.setPosition(float("inf"))
    _m.setVelocity(0.0)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

emitter  = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

# ── Persistent action state (hold last command between supervisor packets) ─────
_vx    = 0.0
_vz    = 0.0
_omega = 0.0

# ── Main control loop ─────────────────────────────────────────────────────────
while robot.step(timestep) != -1:

    # 1. Read lidar (360 range values)
    scan = lidar.getRangeImage() or []
    scan = list(scan)
    # Pad or truncate to exactly N_LIDAR readings
    scan = (scan + [cfg["lidar_max_range"]] * N_LIDAR)[:N_LIDAR]

    # 2. Send sensor packet to supervisor (base64-encoded so Webots R2025a
    #    Receiver.getString() can decode the payload without UTF-8 errors)
    try:
        payload = base64.b64encode(
            struct.pack(_SENSOR_FMT, robot_id, *scan)
        ).decode("ascii")
        emitter.send(payload)
    except Exception:
        pass  # non-fatal — supervisor will use last known lidar

    # 3. Receive latest action (drain queue, keep most recent)
    while receiver.getQueueLength() > 0:
        try:
            raw = base64.b64decode(receiver.getString())
            if len(raw) == _ACTION_BYTES:
                _vx, _vz, _omega = struct.unpack(_ACTION_FMT, raw)
        except Exception:
            pass
        receiver.nextPacket()

    # 4. Apply IK and drive motors
    w1, w2, w3 = _ik(_vx, _vz, _omega, R, L)
    m1.setVelocity(_clamp(w1, MAX_W))
    m2.setVelocity(_clamp(w2, MAX_W))
    m3.setVelocity(_clamp(w3, MAX_W))
