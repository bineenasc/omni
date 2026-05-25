"""
Soccer Supervisor Controller  +  Gymnasium Training Environment  (1×0).

Dual role:
  • Webots Supervisor controller  — drives the simulation step-by-step.
  • gymnasium.Env                 — exposes the RL interface used by SB3.

Architecture
────────────
                    channel 0 (action)
  Supervisor ──────────────────────────► Robot controller
             ◄────────────────────────── Robot controller
                    channel 1 (lidar + robot_id)

The Supervisor is OMNISCIENT for positions: it reads robot and ball
positions directly via the Webots node API (no IPC needed for that).
Only lidar data travels over the wire because the Supervisor cannot
read sensor values that belong to another controller process.

Observation space (gymnasium.spaces.Dict)
──────────────────────────────────────────
  "vector"  Box(7,)   [type_id, dist_ball_n, dir_bx, dir_bz,
                        dist_goal_n, dir_gx, dir_gz]
  "lidar"   Box(360,) normalised range image [0, 1]

Action space
────────────
  Box(3,) in [-1, 1]   →   scaled to [vx, vz, omega]

Reward (dense)
──────────────
  +10    goal scored
  -8     own goal
  ±3·Δd  progress toward ball
  ±5·Δd  progress of ball toward attack goal
  +0.05  ball aligned between robot and goal
  +0..0.5 ball velocity component toward goal
  -0.05  robot stationary for > 30 consecutive steps
  -0.001 per-step time penalty

Usage
─────
  Webots opens worlds/soccer.wbt.
  This file is the supervisor controller → model.learn() drives the loop.
  The simulation must be RUNNING (not paused) while training.
"""

from __future__ import annotations

import base64
import math
import os
import struct
import sys

import numpy as np

# ── Bootstrap shared_configs (controllers/ is one level up) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared_configs import BALL, FIELD, IPC, ROBOT_CONFIGS, SIM, get_robot_config

import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# ── Module-level constants ────────────────────────────────────────────────────
FIELD_DIAG = math.sqrt(FIELD["half_width"] ** 2 + FIELD["half_length"] ** 2)

# VRML snippets used by swap_robot() to insert each robot.
# Both protos are DEF'd as "VIPER" so self._robot_node = getFromDef("VIPER")
# remains valid after the swap, and the reset logic never needs updating.
# Titan uses name "titan" so robot.getName() returns the correct config key.
_ROBOT_VRML: dict[str, str] = {
    "viper": (
        'DEF VIPER Viper {\n'
        '  translation -0.35 0 0.055\n'
        '  rotation 1 0 0 -1.5707953071795862\n'
        '  name "viper"\n'
        '  controller "robot_controller"\n'
        '}'
    ),
    "titan": (
        'DEF VIPER Titan {\n'
        '  translation 0.35 0 0.055\n'
        '  rotation 1 0 0 -1.5707953071795862\n'
        '  name "titan"\n'
        '  controller "robot_controller"\n'
        '}'
    ),
}

MAX_LINEAR  = 0.5   # m/s  — scale factor for normalised vx / vz action
MAX_ANGULAR = 3.0   # rad/s — scale factor for normalised omega action

_N_LIDAR      = IPC["n_lidar"]         # 360
_SENSOR_FMT   = IPC["sensor_fmt"]      # "i360f"
_SENSOR_BYTES = IPC["sensor_bytes"]    # 1444
_ACTION_FMT   = IPC["action_fmt"]      # "3f"
_ACTION_BYTES = IPC["action_bytes"]    # 12


# ══════════════════════════════════════════════════════════════════════════════
class SoccerEnv(Supervisor, gym.Env):
    """1×0 soccer RL environment backed by a Webots Supervisor."""

    metadata = {"render_modes": []}

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        Supervisor.__init__(self)
        gym.Env.__init__(self)

        # Run as fast as the CPU allows — no real-time synchronisation.
        # This is safe in training because we drive every step from Python.
        self.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self._timestep      = int(self.getBasicTimeStep())   # 8 ms
        self._steps_per_act = SIM["steps_per_action"]        # 5  → 40 ms/step
        self._max_steps     = SIM["max_episode_steps"]       # 2000

        # ── Webots node handles ────────────────────────────────────────────
        self._ball_node  = self.getFromDef("BOLA")
        self._robot_node = self.getFromDef("VIPER")

        # ── IPC devices (declared in the Supervisor node in soccer.wbt) ───
        self._emitter  = self.getDevice("supervisor_emitter")
        self._receiver = self.getDevice("supervisor_receiver")
        self._receiver.enable(self._timestep)

        # ── Gymnasium spaces ───────────────────────────────────────────────
        # "vector" bounds are per-component:
        #   index 0 : type_id             ∈ {0, 1}    → [0, 1]
        #   index 1 : dist_ball_norm      ∈ [0, 1]    → [0, 1]
        #   index 2 : dir_ball_x          ∈ [-1, 1]
        #   index 3 : dir_ball_z          ∈ [-1, 1]
        #   index 4 : dist_goal_norm      ∈ [0, 1]    → [0, 1]
        #   index 5 : dir_goal_x          ∈ [-1, 1]
        #   index 6 : dir_goal_z          ∈ [-1, 1]
        _vec_low  = np.array([0., 0., -1., -1., 0., -1., -1.], dtype=np.float32)
        _vec_high = np.array([1., 1.,  1.,  1., 1.,  1.,  1.], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=_vec_low, high=_vec_high, dtype=np.float32),
            "lidar":  spaces.Box(low=0.0, high=1.0,
                                 shape=(_N_LIDAR,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Episode state ──────────────────────────────────────────────────
        self._active_robot        : str   = "viper"
        self._step_count          : int   = 0
        self._still_steps         : int   = 0
        self._last_lidar          = np.ones(_N_LIDAR, dtype=np.float32)
        self._prev_dist_ball      : float = FIELD_DIAG
        self._prev_dist_ball_goal : float = FIELD_DIAG
        self._prev_robot_pos      : tuple = (0.0, 0.0)
        self._rng                         = np.random.default_rng()

    # ══════════════════════════════════════════════════════════════════════════
    # Gymnasium core
    # ══════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count  = 0
        self._still_steps = 0

        # ── Randomise ball (centre region of the field) ────────────────────
        bx = self._rng.uniform(
            -FIELD["half_width"]  * 0.60,
             FIELD["half_width"]  * 0.60,
        )
        bz = self._rng.uniform(
            -FIELD["half_length"] * 0.40,
             FIELD["half_length"] * 0.40,
        )
        self._ball_node.getField("translation").setSFVec3f(
            [float(bx), BALL["radius"], float(bz)]
        )
        self._ball_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._ball_node.resetPhysics()

        # ── Randomise robot (own half: z < 0) ─────────────────────────────
        rx = self._rng.uniform(
            -FIELD["half_width"]  * 0.70,
             FIELD["half_width"]  * 0.70,
        )
        rz = self._rng.uniform(
            -FIELD["half_length"] * 0.85,
            -0.5,
        )
        self._robot_node.getField("translation").setSFVec3f(
            [float(rx), 0.0, float(rz)]
        )
        # Restore the standard -90° X rotation that orients the proto in NUE
        self._robot_node.getField("rotation").setSFRotation(
            [1.0, 0.0, 0.0, -math.pi / 2]
        )
        self._robot_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._robot_node.resetPhysics()

        # ── Settle physics (send zero action for a few steps) ──────────────
        for _ in range(8):
            self._send_action(0.0, 0.0, 0.0)
            self._sim_step()

        self._drain_receiver()  # discard lidar from settle steps

        # ── Compute initial distances ──────────────────────────────────────
        ball_pos = _flat(self._ball_node)
        obs      = self._get_obs()

        self._prev_dist_ball      = float(obs["vector"][1]) * FIELD_DIAG
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"]
        )
        self._prev_robot_pos = _flat(self._robot_node)

        return obs, {}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> tuple:
        self._step_count += 1

        # ── Scale and send action to robot ─────────────────────────────────
        vx    = float(np.clip(action[0], -1.0, 1.0)) * MAX_LINEAR
        vz    = float(np.clip(action[1], -1.0, 1.0)) * MAX_LINEAR
        omega = float(np.clip(action[2], -1.0, 1.0)) * MAX_ANGULAR
        self._send_action(vx, vz, omega)

        # ── Advance simulation (N physics steps per RL step) ───────────────
        for _ in range(self._steps_per_act):
            self._sim_step()

        # ── Collect latest lidar ───────────────────────────────────────────
        self._drain_receiver()

        # ── Build observation ──────────────────────────────────────────────
        obs       = self._get_obs()
        ball_pos  = _flat(self._ball_node)
        robot_pos = _flat(self._robot_node)
        dist_ball = float(obs["vector"][1]) * FIELD_DIAG

        # ── Events & terminal flags ────────────────────────────────────────
        events     = self._check_events(ball_pos)
        terminated = events["goal_scored"] or events["own_goal"] or events["ball_out"]
        truncated  = self._step_count >= self._max_steps

        # ── Reward (uses OLD _prev_* values) ──────────────────────────────
        reward = self._compute_reward(
            dist_ball, ball_pos, robot_pos, events
        )

        # ── Update prev state for next step ───────────────────────────────
        self._prev_dist_ball      = dist_ball
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"]
        )
        self._prev_robot_pos = robot_pos

        info = {**events, "step": self._step_count}
        return obs, float(reward), terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # Observation
    # ══════════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> dict:
        cfg       = get_robot_config(self._active_robot)
        robot_pos = _flat(self._robot_node)
        ball_pos  = _flat(self._ball_node)
        goal_pos  = (0.0, float(FIELD["goal_z_attack"]))

        dist_ball, dir_ball = _vec2d(robot_pos, ball_pos)
        dist_goal, dir_goal = _vec2d(robot_pos, goal_pos)

        vector = np.array(
            [
                float(cfg["type_id"]),       # 0 (viper) or 1 (titan)
                dist_ball / FIELD_DIAG,      # normalised robot→ball distance
                dir_ball[0],                 # unit-vector X component
                dir_ball[1],                 # unit-vector Z component
                dist_goal / FIELD_DIAG,      # normalised robot→goal distance
                dir_goal[0],
                dir_goal[1],
            ],
            dtype=np.float32,
        )
        return {"vector": vector, "lidar": self._last_lidar.copy()}

    # ══════════════════════════════════════════════════════════════════════════
    # Reward
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        dist_ball : float,
        ball_pos  : tuple,
        robot_pos : tuple,
        events    : dict,
    ) -> float:

        # 1. Terminal events — dominant signal, early return
        if events["goal_scored"]:
            return 10.0
        if events["own_goal"]:
            return -8.0
        if events["ball_out"]:
            return -2.0     # ball left the field without scoring

        reward = 0.0

        # 2. Progress of robot toward ball
        reward += (self._prev_dist_ball - dist_ball) * 3.0

        # 3. Progress of ball toward attack goal
        dist_ball_to_goal = math.hypot(
            ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"]
        )
        reward += (self._prev_dist_ball_goal - dist_ball_to_goal) * 5.0

        # 4. Ball aligned on the robot→goal line (+0.05 bonus)
        reward += _alignment_bonus(
            ball_pos, robot_pos, FIELD["goal_z_attack"]
        )

        # 5. Ball velocity component directed toward goal
        reward += _ball_toward_goal_bonus(
            self._ball_node, ball_pos, FIELD["goal_z_attack"]
        )

        # 6. Stillness penalty — encourages the robot to keep moving
        moved = math.hypot(
            robot_pos[0] - self._prev_robot_pos[0],
            robot_pos[1] - self._prev_robot_pos[1],
        )
        if moved < 0.003:
            self._still_steps += 1
            if self._still_steps > 30:
                reward -= 0.05
        else:
            self._still_steps = 0

        # 7. Per-step time penalty
        reward -= 0.001

        return float(reward)

    # ══════════════════════════════════════════════════════════════════════════
    # Terminal conditions
    # ══════════════════════════════════════════════════════════════════════════

    def _check_events(self, ball_pos: tuple) -> dict:
        bx, bz = ball_pos
        hw = FIELD["goal_half_width"]

        # Ball inside goal mouth (crosses the goal-line within the post width)
        goal_scored = bz >= FIELD["goal_z_attack"] and abs(bx) <= hw
        own_goal    = bz <= FIELD["goal_z_own"]    and abs(bx) <= hw

        # Ball outside the pitch (side lines or end lines that are NOT goals)
        ball_out = (
            abs(bx) > FIELD["half_width"]  + 0.05               # side out
            or (bz >=  FIELD["goal_z_attack"] and not goal_scored)  # end line, missed goal
            or (bz <= FIELD["goal_z_own"]    and not own_goal)       # own end, missed
        )

        return {
            "goal_scored": goal_scored,
            "own_goal":    own_goal,
            "ball_out":    ball_out,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # IPC helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _drain_receiver(self) -> None:
        """Consume all pending IPC packets; keep only the most recent lidar.

        Webots R2025a changed Receiver.getData() to decode bytes as UTF-8,
        which breaks for binary struct data.  We base64-encode every packet
        before sending so the wire payload is always valid ASCII.  Here we
        call getString() (the new API) and base64-decode before unpacking.
        """
        latest: bytes | None = None
        while self._receiver.getQueueLength() > 0:
            try:
                raw = base64.b64decode(self._receiver.getString())
                if len(raw) == _SENSOR_BYTES:
                    latest = raw
            except Exception:
                pass
            self._receiver.nextPacket()

        if latest is not None:
            vals  = struct.unpack(_SENSOR_FMT, latest)
            arr   = np.array(vals[1:], dtype=np.float32)
            max_r = get_robot_config(self._active_robot)["lidar_max_range"]
            self._last_lidar = np.clip(arr / max_r, 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Robot hot-swap (used by train.py for epoch alternation — Step 8)
    # ══════════════════════════════════════════════════════════════════════════

    def set_robot_type(self, robot_name: str) -> None:
        """Update the active robot label (type_id in obs) without a physical swap."""
        if robot_name not in ROBOT_CONFIGS:
            raise ValueError(
                f"Unknown robot '{robot_name}'. "
                f"Valid names: {list(ROBOT_CONFIGS)}"
            )
        self._active_robot = robot_name

    def swap_robot(self, robot_name: str) -> None:
        """
        Replace the physical robot node in the simulation.

        Removes the node currently DEF'd as "VIPER" from the scene, then
        inserts the requested proto (Viper or Titan) under the same DEF name
        so all existing references remain valid.

        Both Viper and Titan protos must be declared as EXTERNPROTO in the
        world file (soccer.wbt).  After insertion, 10 settle steps are run
        so the new controller can initialise before training resumes.
        """
        if robot_name not in ROBOT_CONFIGS:
            raise ValueError(
                f"Unknown robot '{robot_name}'. "
                f"Valid names: {list(ROBOT_CONFIGS)}"
            )

        # Remove current robot node
        current = self.getFromDef("VIPER")
        if current is not None:
            current.remove()

        # Insert requested proto, keeping DEF name "VIPER"
        self.getRoot().getField("children").importMFNodeFromString(
            -1, _ROBOT_VRML[robot_name]
        )

        # Refresh node reference
        self._robot_node  = self.getFromDef("VIPER")
        self._active_robot = robot_name

        # Let the new controller initialise and lidar warm up
        for _ in range(10):
            self._send_action(0.0, 0.0, 0.0)
            self._sim_step()

        # Reset lidar buffer for the fresh robot
        self._last_lidar[:] = 1.0
        print(f"[SoccerEnv] Swapped to '{robot_name}'.")

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation stepping
    # ══════════════════════════════════════════════════════════════════════════

    def _send_action(self, vx: float, vz: float, omega: float) -> None:
        """Base64-encode a 3-float action packet and send it to the robot.

        Required because Webots R2025a Receiver.getData() decodes bytes as
        UTF-8, breaking raw struct payloads.  Encoding as base64 (pure ASCII)
        keeps every packet decodable on both ends.
        """
        payload = base64.b64encode(
            struct.pack(_ACTION_FMT, vx, vz, omega)
        ).decode("ascii")
        self._emitter.send(payload)

    def _sim_step(self) -> None:
        """Advance Webots by one basic timestep (avoids name clash with gym.step)."""
        Supervisor.step(self, self._timestep)

    # ══════════════════════════════════════════════════════════════════════════
    # Required gym.Env stubs
    # ══════════════════════════════════════════════════════════════════════════

    def render(self) -> None:
        pass   # Webots provides its own 3-D visualisation window

    def close(self) -> None:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# Module-level helpers (pure functions — no self dependency)
# ════════════════════════════════════════════════════════════════════════════════

def _flat(node) -> tuple[float, float]:
    """Return (X, Z) ground-plane position from a Webots node (NUE: Y is up)."""
    p = node.getPosition()
    return (p[0], p[2])


def _vec2d(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    """
    Return (distance, unit_vector) from 2-D point a to point b.
    unit_vector is (0, 0) when distance < 1e-6.
    """
    dx, dz = b[0] - a[0], b[1] - a[1]
    dist   = math.hypot(dx, dz)
    if dist > 1e-6:
        return dist, (dx / dist, dz / dist)
    return 0.0, (0.0, 0.0)


def _alignment_bonus(
    ball_pos  : tuple[float, float],
    robot_pos : tuple[float, float],
    goal_z    : float,
) -> float:
    """
    Returns +0.05 when the ball lies within 30 cm of the straight line
    from the robot to the attack goal centre — i.e. the robot is positioned
    behind the ball and facing goal.
    """
    rx, rz = robot_pos
    bx, bz = ball_pos

    # Ball must be between robot and goal along the Z axis
    if not (min(rz, goal_z) < bz < max(rz, goal_z)):
        return 0.0

    line_dx, line_dz = 0.0 - rx, goal_z - rz
    line_len = math.hypot(line_dx, line_dz)
    if line_len < 1e-6:
        return 0.0

    # Perpendicular distance of the ball from the robot→goal line
    db_dx, db_dz = bx - rx, bz - rz
    perp = abs(db_dx * line_dz - db_dz * line_dx) / line_len
    return 0.05 if perp < 0.30 else 0.0


def _ball_toward_goal_bonus(
    ball_node : object,
    ball_pos  : tuple[float, float],
    goal_z    : float,
) -> float:
    """
    Proportional reward for the component of ball velocity directed toward
    the attack goal.  Capped at +0.5 per step.
    """
    try:
        vel = ball_node.getVelocity()   # [vx, vy, vz, wx, wy, wz] in NUE
    except Exception:
        return 0.0

    bx, bz    = ball_pos
    to_x, to_z = 0.0 - bx, goal_z - bz
    dist = math.hypot(to_x, to_z)
    if dist < 1e-6:
        return 0.0

    # Dot product of ball velocity with unit direction toward goal
    dot = vel[0] * (to_x / dist) + vel[2] * (to_z / dist)
    return float(np.clip(dot * 0.5, 0.0, 0.5))


# ════════════════════════════════════════════════════════════════════════════════
# Webots entry point — delegates to train.py
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # train.py lives in the same directory and takes the raw env as argument,
    # avoiding any circular-import issue (train.py never imports this module).
    from train import train  # noqa: E402
    train(SoccerEnv())
