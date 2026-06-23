"""
Soccer Supervisor 1v1  —  Gymnasium Environment  +  Webots Supervisor.

Layout
──────
  Viper  attacks  +Z goal  (goal_z_attack = +4.55)
  Titan  attacks  −Z goal  (goal_z_own    = −4.55)
  Both run the same robot_controller.py via Webots IPC.

IPC channels
────────────
  ch 0: supervisor_emitter        → Viper receiver        (action)
  ch 1: Viper emitter             → supervisor_receiver   (lidar)
  ch 2: supervisor_emitter_titan  → Titan receiver        (action)
  ch 3: Titan emitter             → supervisor_receiver_titan (lidar)

Observation (30D, always from the ACTIVE robot's local frame)
─────────────────────────────────────────────────────────────
  [0]      type_id             — 0=Viper, 1=Titan
  [1]      dist_ball_norm
  [2-3]    dir_ball  (local x, z)
  [4]      dist_goal_norm      — active robot's ATTACK goal
  [5-6]    dir_goal  (local x, z)
  [7-18]   4 posts × (dist_norm, dir_x, dir_z) LOCAL
  [19-21]  robot velocity (vx, vz, ω) LOCAL, normalised
  [22-23]  ball velocity  (vx, vz)    LOCAL, normalised
  [24]     dist_opp_norm
  [25-26]  dir_opp   (local x, z)
  [27-28]  opp velocity (vx, vz)      LOCAL, normalised
  [29]     opp_between_agent_goal      — 1.0 if opponent is in the
            corridor between active robot and active robot's attack goal

Action (3D, active robot's LOCAL frame, clipped to [-1, 1])
────────────────────────────────────────────────────────────
  [0] vx   → scaled to  MAX_LINEAR  m/s
  [1] vz   → scaled to  MAX_LINEAR  m/s
  [2] ω    → scaled to  MAX_ANGULAR rad/s

Curriculum phases
─────────────────
  "warmup"  — 60 k steps, only Viper learns, Titan static at fixed position.
               Viper adapts its 1v0 weights to the extended 30-D obs space.
  "phase1"  — Viper only learns.  Titan static in its own half.
               Advances when Viper goal_rate ≥ 60 % over last 10 episodes.
  "phase2"  — Alternating fine-tune.  When Viper learns: Titan uses scripted
               rule (go-to-ball → shoot at −Z).  When Titan learns: Viper uses
               its latest frozen checkpoint.
               Advances when BOTH robots have reached the goal_rate threshold.
  "phase3"  — Alternating.  Opponent always uses its latest saved snapshot
               (updated once per epoch).

Entry point (Webots)
────────────────────
  Webots runs this file as "soccer_supervisor_1v1" controller.
  At module level (bottom of file): SoccerEnv1v1() is created; depending on
  MODE in shared_configs, train_1v1() or a future eval function is called.
"""

from __future__ import annotations

import base64
import math
import os
import struct
import sys
from collections import deque

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared_configs import (
    BALL, FIELD, IPC_1V1, MODE, ROBOT_CONFIGS, SIM,
    get_robot_config,
)

import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# ── Module-level constants ────────────────────────────────────────────────────
FIELD_DIAG    = math.sqrt(FIELD["half_width"] ** 2 + FIELD["half_length"] ** 2)
MAX_LINEAR    = 0.5    # m/s
MAX_ANGULAR   = 3.0    # rad/s
MAX_BALL_SPD  = 4.0    # m/s  (for normalising ball vel in obs)
MAX_OPP_SPD   = 0.5    # m/s  (for normalising opponent vel in obs)

_N_LIDAR      = IPC_1V1["n_lidar"]
_SENSOR_FMT   = IPC_1V1["sensor_fmt"]
_SENSOR_BYTES = IPC_1V1["sensor_bytes"]
_ACTION_FMT   = IPC_1V1["action_fmt"]
_ACTION_BYTES = IPC_1V1["action_bytes"]

# Minimum inter-robot separation at spawn (m).
_MIN_ROBOT_SEP = 0.50

# Titan's fixed spawn position during warmup/phase1 (static obstacle).
_TITAN_STATIC_X = 0.0
_TITAN_STATIC_Z = -3.0   # its own half, well away from the ball


# ══════════════════════════════════════════════════════════════════════════════
class SoccerEnv1v1(Supervisor, gym.Env):
    """1v1 soccer RL environment backed by a Webots Supervisor.

    Call ``set_active_robot(name)`` to switch which robot is the learning agent.
    Call ``set_phase(phase)``      to advance the curriculum.
    Call ``set_opp_policy(model)`` to install a frozen opponent policy (phase 3).
    """

    metadata = {"render_modes": []}

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        Supervisor.__init__(self)
        gym.Env.__init__(self)

        self.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self._timestep      = int(self.getBasicTimeStep())   # 8 ms
        self._steps_per_act = SIM["steps_per_action"]        # 5 → 40 ms/step
        self._max_steps     = SIM["max_episode_steps"]       # 1000

        # ── Webots node handles ────────────────────────────────────────────
        self._ball_node  = self.getFromDef("BOLA")
        self._viper_node = self.getFromDef("VIPER")
        self._titan_node = self.getFromDef("TITAN")

        # ── IPC devices ───────────────────────────────────────────────────
        # Viper IPC
        self._emitter_viper   = self.getDevice("supervisor_emitter")
        self._receiver_viper  = self.getDevice("supervisor_receiver")
        self._receiver_viper.enable(self._timestep)
        # Titan IPC
        self._emitter_titan   = self.getDevice("supervisor_emitter_titan")
        self._receiver_titan  = self.getDevice("supervisor_receiver_titan")
        self._receiver_titan.enable(self._timestep)

        # ── Gymnasium spaces (30-D observation) ────────────────────────────
        #  [0]      type_id                  [0, 1]
        #  [1]      dist_ball_norm           [0, 1]
        #  [2-3]    dir_ball  LOCAL          [-1,1]²
        #  [4]      dist_goal_norm           [0, 1]
        #  [5-6]    dir_goal  LOCAL          [-1,1]²
        #  [7-18]   4 posts × (dist, dx, dz) LOCAL
        #  [19-21]  robot vel (vx,vz,ω)      [-1,1]³
        #  [22-23]  ball vel  (vx,vz)        [-1,1]²
        #  [24]     dist_opp_norm            [0, 1]
        #  [25-26]  dir_opp   LOCAL          [-1,1]²
        #  [27-28]  opp vel   (vx,vz)        [-1,1]²
        #  [29]     opp_between_flag         {0, 1}
        _post_low  = [0., -1., -1.] * 4
        _post_high = [1.,  1.,  1.] * 4
        _obs_low  = np.array(
            [0., 0., -1., -1., 0., -1., -1.]
            + _post_low
            + [-1., -1., -1., -1., -1.]    # robot+ball vel
            + [0., -1., -1., -1., -1., 0.] # opp dist/dir/vel/flag
            , dtype=np.float32
        )
        _obs_high = np.array(
            [1., 1.,  1.,  1., 1.,  1.,  1.]
            + _post_high
            + [1.,  1.,  1.,  1.,  1.]
            + [1.,  1.,  1.,  1.,  1., 1.]
            , dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=_obs_low, high=_obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Episode state ──────────────────────────────────────────────────
        self._active_robot        : str   = "viper"
        self._phase               : str   = "warmup"
        self._opp_policy                  = None   # frozen opponent model (phase3)

        self._step_count          : int   = 0
        self._still_steps         : int   = 0
        self._post_stuck_steps    : int   = 0
        self._midfield_bonus_given: bool  = False
        self._last_touch_step     : int   = -999
        self._prev_dist_ball      : float = FIELD_DIAG
        self._prev_dist_ball_goal : float = FIELD_DIAG
        self._prev_robot_pos      : tuple = (0.0, 0.0)
        self._prev_ball_z         : float = 0.0

        self._last_lidar_viper = np.ones(_N_LIDAR, dtype=np.float32)
        self._last_lidar_titan = np.ones(_N_LIDAR, dtype=np.float32)

        self._rng = np.random.default_rng()

        # ── Curriculum step counter (never reset between episodes) ─────────
        self._curriculum_step  : int   = 0
        self._CURRICULUM_WINDOW: int   = 10
        self._CURRICULUM_THRESH: float = 0.60
        self._curriculum_outcomes: deque = deque(maxlen=10)

        # ── No-progress history ────────────────────────────────────────────
        _WIN = 150
        self._pos_history = deque(maxlen=_WIN)

    # ══════════════════════════════════════════════════════════════════════════
    # Public API — called by train_1v1.py
    # ══════════════════════════════════════════════════════════════════════════

    def set_active_robot(self, robot_name: str) -> None:
        """Switch which robot is the learning agent for the next episode."""
        if robot_name not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot '{robot_name}'.")
        self._active_robot = robot_name

    def set_phase(self, phase: str) -> None:
        """Advance the curriculum phase.

        Valid values: "warmup", "phase1", "phase2", "phase3".
        Resets the adaptive curriculum counter.
        """
        valid = {"warmup", "phase1", "phase2", "phase3"}
        if phase not in valid:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {valid}")
        self._phase = phase
        self._curriculum_outcomes.clear()
        self._curriculum_step = 0
        print(f"[SoccerEnv1v1] Phase → {phase}")

    def set_opp_policy(self, policy) -> None:
        """Install a frozen opponent policy (used in phase3).

        ``policy`` is a loaded SB3 PPO model whose ``predict(obs)`` method
        returns normalised actions in [-1, 1]³.  Pass ``None`` to revert to
        the scripted opponent.
        """
        self._opp_policy = policy

    # ══════════════════════════════════════════════════════════════════════════
    # Gymnasium core
    # ══════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count          = 0
        self._still_steps         = 0
        self._post_stuck_steps    = 0
        self._midfield_bonus_given= False
        self._last_touch_step     = -999
        self._pos_history.clear()

        agent_node = self._agent_node()
        opp_node   = self._opp_node()
        attack_z   = self._agent_attack_z()
        own_z      = self._agent_own_z()

        # ── Spawn positions based on phase ────────────────────────────────
        if self._phase in ("warmup", "phase1"):
            bx, bz, rx, rz = self._spawn_1v0_style(attack_z)
            ox, oz          = _TITAN_STATIC_X, _TITAN_STATIC_Z
            # Mirror static position if Titan is the active agent
            if self._active_robot == "titan":
                oz = -_TITAN_STATIC_Z

        elif self._phase == "phase2":
            bx, bz, rx, rz, ox, oz = self._spawn_phase2(attack_z, own_z)

        else:  # phase3 — fully random with collision avoidance
            bx, bz, rx, rz, ox, oz = self._spawn_phase3(attack_z, own_z)

        # ── Place the ball ────────────────────────────────────────────────
        self._ball_node.getField("translation").setSFVec3f(
            [float(bx), BALL["radius"], float(bz)]
        )
        self._ball_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._ball_node.resetPhysics()

        # ── Place active robot ────────────────────────────────────────────
        agent_node.getField("translation").setSFVec3f(
            [float(rx), 0.0, float(rz)]
        )
        agent_rot = self._spawn_rotation_for(attack_z)
        agent_node.getField("rotation").setSFRotation(agent_rot)
        agent_node.setVelocity([0, 0, 0, 0, 0, 0])
        agent_node.resetPhysics()

        # ── Place opponent ────────────────────────────────────────────────
        opp_node.getField("translation").setSFVec3f(
            [float(ox), 0.0, float(oz)]
        )
        opp_attack_z = self._opp_attack_z()
        opp_rot = self._spawn_rotation_for(opp_attack_z)
        if self._phase in ("warmup", "phase1"):
            # Static opponent — use default upright rotation
            opp_rot = [1.0, 0.0, 0.0, -math.pi / 2]
        opp_node.getField("rotation").setSFRotation(opp_rot)
        opp_node.setVelocity([0, 0, 0, 0, 0, 0])
        opp_node.resetPhysics()

        # ── Settle physics ────────────────────────────────────────────────
        for _ in range(15):
            self._send_agent_action(0.0, 0.0, 0.0)
            self._send_opp_action(0.0, 0.0, 0.0)
            self._sim_step()

        # Guard against physics glitches (sinking / tipping)
        for node, tx, tz in [(agent_node, rx, rz), (opp_node, ox, oz)]:
            m = node.getOrientation()
            if node.getPosition()[1] < -0.02 or m[4] < 0.7:
                node.getField("translation").setSFVec3f(
                    [float(tx), 0.0, float(tz)]
                )
                node.getField("rotation").setSFRotation(
                    [1.0, 0.0, 0.0, -math.pi / 2]
                )
                node.setVelocity([0, 0, 0, 0, 0, 0])
                node.resetPhysics()
                for _ in range(5):
                    self._send_agent_action(0.0, 0.0, 0.0)
                    self._send_opp_action(0.0, 0.0, 0.0)
                    self._sim_step()

        self._drain_receivers()

        ball_pos = _flat(self._ball_node)
        obs      = self._get_obs()

        self._prev_dist_ball      = float(obs[1]) * FIELD_DIAG
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0], ball_pos[1] - attack_z
        )
        self._prev_robot_pos = _flat(agent_node)
        self._prev_ball_z    = ball_pos[1]

        return obs, {}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> tuple:
        self._step_count += 1

        # ── Scale and send active robot's action ──────────────────────────
        vx    = float(np.clip(action[0], -1.0, 1.0)) * MAX_LINEAR
        vz    = float(np.clip(action[1], -1.0, 1.0)) * MAX_LINEAR
        omega = float(np.clip(action[2], -1.0, 1.0)) * MAX_ANGULAR
        self._send_agent_action(vx, vz, omega)

        # ── Compute and send opponent's action ────────────────────────────
        opp_vx, opp_vz, opp_omega = self._compute_opp_action()
        self._send_opp_action(opp_vx, opp_vz, opp_omega)

        # ── Advance simulation ────────────────────────────────────────────
        for _ in range(self._steps_per_act):
            self._sim_step()

        self._drain_receivers()

        # ── Build observation and collect state ───────────────────────────
        obs       = self._get_obs()
        ball_pos  = _flat(self._ball_node)
        robot_pos = _flat(self._agent_node())
        dist_ball = float(obs[1]) * FIELD_DIAG

        moved = math.hypot(
            robot_pos[0] - self._prev_robot_pos[0],
            robot_pos[1] - self._prev_robot_pos[1],
        )

        # Goal-post stuck detection
        if self._is_near_post(robot_pos) and moved < 0.003:
            self._post_stuck_steps += 1
        else:
            self._post_stuck_steps = max(0, self._post_stuck_steps - 2)

        # ── Events ────────────────────────────────────────────────────────
        events     = self._check_events(ball_pos)
        terminated = events["agent_goal"] or events["opp_goal"] or events["ball_out"]
        truncated  = (
            self._step_count >= self._max_steps
            or self._post_stuck_steps > 100
        )

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._compute_reward(dist_ball, ball_pos, robot_pos, moved, events)

        # ── Update state ──────────────────────────────────────────────────
        self._prev_dist_ball      = dist_ball
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0], ball_pos[1] - self._agent_attack_z()
        )
        self._prev_robot_pos  = robot_pos
        self._prev_ball_z     = ball_pos[1]
        self._curriculum_step += 1

        # Adaptive curriculum advancement (same mechanism as 1v0)
        if terminated or truncated:
            self._curriculum_outcomes.append(1 if events["agent_goal"] else 0)
            if len(self._curriculum_outcomes) == self._CURRICULUM_WINDOW:
                goal_rate = sum(self._curriculum_outcomes) / self._CURRICULUM_WINDOW
                if goal_rate >= self._CURRICULUM_THRESH:
                    print(
                        f"[Curriculum] {self._phase} ready to advance — "
                        f"goal_rate={goal_rate:.0%} over last {self._CURRICULUM_WINDOW} eps "
                        f"(active={self._active_robot})"
                    )

        info = {**events, "step": self._step_count}
        return obs, float(reward), terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # Observation
    # ══════════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        """Compute 30-D observation from the active robot's perspective."""
        return self._get_obs_internal(
            agent_node  = self._agent_node(),
            opp_node    = self._opp_node(),
            attack_z    = self._agent_attack_z(),
            own_z       = self._agent_own_z(),
            agent_cfg   = get_robot_config(self._active_robot),
        )

    def _get_obs_internal(
        self,
        agent_node,
        opp_node,
        attack_z: float,
        own_z: float,
        agent_cfg: dict,
    ) -> np.ndarray:
        """
        Unified 30-D observation builder.
        Can be called for any agent/opponent pair, enabling the frozen opponent
        policy to receive its own local-frame observation during phase 3.
        """
        robot_pos = _flat(agent_node)
        ball_pos  = _flat(self._ball_node)
        goal_pos  = (0.0, float(attack_z))

        heading  = _get_heading_from_node(agent_node)
        cos_h    = math.cos(heading)
        sin_h    = math.sin(heading)

        def to_local(dx: float, dz: float) -> tuple[float, float]:
            return (cos_h * dx + sin_h * dz, -sin_h * dx + cos_h * dz)

        dist_ball, dir_ball_w = _vec2d(robot_pos, ball_pos)
        dist_goal, dir_goal_w = _vec2d(robot_pos, goal_pos)
        dir_ball = to_local(dir_ball_w[0], dir_ball_w[1])
        dir_goal = to_local(dir_goal_w[0], dir_goal_w[1])

        # 4 goal posts in local frame
        hw = FIELD["goal_half_width"]
        _posts = [
            ( hw, attack_z),   # agent attack right
            (-hw, attack_z),   # agent attack left
            ( hw, own_z),      # agent own right
            (-hw, own_z),      # agent own left
        ]
        post_feats: list[float] = []
        for px, pz in _posts:
            d, uv_w = _vec2d(robot_pos, (px, pz))
            uv_l = to_local(uv_w[0], uv_w[1])
            post_feats += [d / FIELD_DIAG, uv_l[0], uv_l[1]]

        # Robot velocity (local)
        try:
            rv = agent_node.getVelocity()
            rvx_l, rvz_l = to_local(rv[0], rv[2])
            omega_l = float(rv[4])
        except Exception:
            rvx_l = rvz_l = omega_l = 0.0
        rvx_n   = float(np.clip(rvx_l  / MAX_LINEAR,   -1.0, 1.0))
        rvz_n   = float(np.clip(rvz_l  / MAX_LINEAR,   -1.0, 1.0))
        omega_n = float(np.clip(omega_l / MAX_ANGULAR,  -1.0, 1.0))

        # Ball velocity (local)
        try:
            bv = self._ball_node.getVelocity()
            bvx_l, bvz_l = to_local(bv[0], bv[2])
        except Exception:
            bvx_l = bvz_l = 0.0
        bvx_n = float(np.clip(bvx_l / MAX_BALL_SPD, -1.0, 1.0))
        bvz_n = float(np.clip(bvz_l / MAX_BALL_SPD, -1.0, 1.0))

        # Opponent position and velocity (local)
        opp_pos = _flat(opp_node)
        dist_opp, dir_opp_w = _vec2d(robot_pos, opp_pos)
        dir_opp = to_local(dir_opp_w[0], dir_opp_w[1])
        try:
            ov = opp_node.getVelocity()
            ovx_l, ovz_l = to_local(ov[0], ov[2])
        except Exception:
            ovx_l = ovz_l = 0.0
        ovx_n = float(np.clip(ovx_l / MAX_OPP_SPD, -1.0, 1.0))
        ovz_n = float(np.clip(ovz_l / MAX_OPP_SPD, -1.0, 1.0))

        # Binary flag: is opponent in the corridor between agent and attack goal?
        opp_between = _opp_between_agent_goal(robot_pos, opp_pos, goal_pos)

        obs = np.array(
            [
                float(agent_cfg["type_id"]),     # [0]
                dist_ball / FIELD_DIAG,          # [1]
                dir_ball[0],                     # [2]
                dir_ball[1],                     # [3]
                dist_goal / FIELD_DIAG,          # [4]
                dir_goal[0],                     # [5]
                dir_goal[1],                     # [6]
            ]
            + post_feats                         # [7-18]
            + [
                rvx_n, rvz_n, omega_n,           # [19-21]
                bvx_n, bvz_n,                    # [22-23]
                dist_opp / FIELD_DIAG,           # [24]
                dir_opp[0], dir_opp[1],          # [25-26]
                ovx_n, ovz_n,                    # [27-28]
                float(opp_between),              # [29]
            ],
            dtype=np.float32,
        )
        return obs

    # ══════════════════════════════════════════════════════════════════════════
    # Opponent control
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_opp_action(self) -> tuple[float, float, float]:
        """Return (vx, vz, omega) in opponent's LOCAL frame."""
        if self._phase in ("warmup", "phase1"):
            return (0.0, 0.0, 0.0)   # static

        if self._phase == "phase2":
            return self._scripted_opp_action()

        # phase3 — use frozen policy if available, else fallback to scripted
        if self._opp_policy is not None:
            return self._frozen_policy_action()
        return self._scripted_opp_action()

    def _scripted_opp_action(self) -> tuple[float, float, float]:
        """Go to ball → shoot at opponent's attack goal.  Converted to LOCAL."""
        opp_pos      = _flat(self._opp_node())
        ball_pos     = _flat(self._ball_node)
        opp_attack_z = self._opp_attack_z()

        dist_ob, dir_ob = _vec2d(opp_pos, ball_pos)

        if dist_ob > 0.25:
            world_vx = dir_ob[0] * 0.35
            world_vz = dir_ob[1] * 0.35
        else:
            _, dir_g = _vec2d(ball_pos, (0.0, opp_attack_z))
            world_vx = dir_g[0] * 0.45
            world_vz = dir_g[1] * 0.45

        heading = _get_heading_from_node(self._opp_node())
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        local_vx =  cos_h * world_vx + sin_h * world_vz
        local_vz = -sin_h * world_vx + cos_h * world_vz
        return local_vx, local_vz, 0.0

    def _frozen_policy_action(self) -> tuple[float, float, float]:
        """Query the frozen opponent policy using the opponent's local-frame obs."""
        # Determine opponent's role parameters
        if self._active_robot == "viper":
            opp_name     = "titan"
            opp_attack_z = self._opp_attack_z()
            opp_own_z    = self._agent_attack_z()
        else:
            opp_name     = "viper"
            opp_attack_z = self._opp_attack_z()
            opp_own_z    = self._agent_attack_z()

        opp_obs = self._get_obs_internal(
            agent_node = self._opp_node(),
            opp_node   = self._agent_node(),
            attack_z   = opp_attack_z,
            own_z      = opp_own_z,
            agent_cfg  = get_robot_config(opp_name),
        )

        action, _ = self._opp_policy.predict(
            opp_obs.reshape(1, -1), deterministic=True
        )
        action = action.flatten()

        vx    = float(np.clip(action[0], -1.0, 1.0)) * MAX_LINEAR
        vz    = float(np.clip(action[1], -1.0, 1.0)) * MAX_LINEAR
        omega = float(np.clip(action[2], -1.0, 1.0)) * MAX_ANGULAR
        return vx, vz, omega

    # ══════════════════════════════════════════════════════════════════════════
    # Reward
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        dist_ball: float,
        ball_pos: tuple,
        robot_pos: tuple,
        moved: float,
        events: dict,
    ) -> float:
        """Dense reward from active robot's perspective (mirrors 1v0 logic)."""

        # ── 1. Terminal events ────────────────────────────────────────────────
        if events["agent_goal"]:
            return +300.0
        if events["opp_goal"]:
            return -250.0
        if events["ball_out"]:
            return -25.0

        bx, bz   = ball_pos
        rx, rz   = robot_pos
        GOAL_Z   = self._agent_attack_z()
        TOUCH_TH = 0.16
        reward   = 0.0

        # ── 2. Time penalty ───────────────────────────────────────────────────
        reward -= 0.003

        # ── 3. Midfield bonus (once per episode, phase3 only) ─────────────────
        # The sign flip correctly handles Titan attacking -Z.
        _sign = 1 if self._active_robot == "viper" else -1
        if (not self._midfield_bonus_given
                and self._phase == "phase3"
                and self._prev_ball_z * _sign < 0.0
                and bz * _sign >= 0.0):
            self._midfield_bonus_given = True
            reward += 1.0

        # ── 4. Robot → ball progress ──────────────────────────────────────────
        # ×4.0: strong gradient to navigate to the ball (matches bine-safecode).
        reward += (self._prev_dist_ball - dist_ball) * 4.0

        # ── 5. Ball → goal progress ───────────────────────────────────────────
        # max(0, Δ): only reward ball advancing; backward movement never punishes.
        # Without this lock the agent learns to flee the ball (backward punished).
        _hw = FIELD["goal_half_width"]
        _target_x = float(np.clip(bx, -_hw, _hw))
        dist_ball_goal = math.hypot(bx - _target_x, bz - GOAL_Z)
        reward += max(0.0, self._prev_dist_ball_goal - dist_ball_goal) * 15.0

        # ── 6. Ball velocity toward goal ──────────────────────────────────────
        try:
            vel = self._ball_node.getVelocity()
            bvx, bvz = vel[0], vel[2]
            tgx, tgz = -bx, GOAL_Z - bz
            nm = math.hypot(tgx, tgz)
            if nm > 1e-6:
                reward += float(np.clip((bvx*tgx/nm + bvz*tgz/nm) * 0.8, -1.0, 1.0))
        except Exception:
            pass

        # ── 7. Robot–ball–goal alignment ─────────────────────────────────────
        rb_x, rb_z = bx - rx, bz - rz
        rb_norm = math.hypot(rb_x, rb_z)
        if rb_norm > 1e-6:
            rb_x /= rb_norm; rb_z /= rb_norm
            bg_x, bg_z = -bx, GOAL_Z - bz
            bg_norm = math.hypot(bg_x, bg_z)
            if bg_norm > 1e-6:
                bg_x /= bg_norm; bg_z /= bg_norm
                reward += (rb_x * bg_x + rb_z * bg_z) * 0.3

        # ── 8. Contact bonus ──────────────────────────────────────────────────
        if dist_ball < TOUCH_TH:
            if (self._step_count - self._last_touch_step) > 8:
                try:
                    bvel = self._ball_node.getVelocity()
                    tgz  = GOAL_Z - bz
                    tgx  = _target_x - bx
                    ng   = math.hypot(tgx, tgz)
                    if ng > 1e-6:
                        dot = (bvel[0] * tgx + bvel[2] * tgz) / ng
                        reward += 0.25 if dot > 0.1 else 0.05
                    else:
                        reward += 0.05
                except Exception:
                    reward += 0.05
                self._last_touch_step = self._step_count

        # ── 9. Ball-hover penalty ─────────────────────────────────────────────
        try:
            ball_speed = math.hypot(*self._ball_node.getVelocity()[:3:2])
            if dist_ball < 0.25 and ball_speed < 0.03:
                reward -= 0.05
        except Exception:
            pass

        # ── 10. Opponent proximity penalty (crowding / blocking) ──────────────
        opp_pos = _flat(self._opp_node())
        dist_opp = math.hypot(robot_pos[0] - opp_pos[0], robot_pos[1] - opp_pos[1])
        if dist_opp < 0.30:
            reward -= 0.08

        # ── 11. Stillness penalty ─────────────────────────────────────────────
        if moved < 0.003:
            self._still_steps += 1
            if self._still_steps > 20:
                reward -= 0.10
        else:
            self._still_steps = 0

        # ── 12. Goal-post stuck penalty ───────────────────────────────────────
        if self._post_stuck_steps > 20:
            reward -= 0.30

        # ── 13. No-progress history (kept for compatibility) ──────────────────
        self._pos_history.append(robot_pos)

        return reward

    # ══════════════════════════════════════════════════════════════════════════
    # Terminal conditions
    # ══════════════════════════════════════════════════════════════════════════

    def _check_events(self, ball_pos: tuple) -> dict:
        bx, bz = ball_pos
        hw = FIELD["goal_half_width"]

        plus_z_goal  = bz >= FIELD["goal_z_attack"] and abs(bx) <= hw
        minus_z_goal = bz <= FIELD["goal_z_own"]    and abs(bx) <= hw

        # From active robot's perspective
        if self._active_robot == "viper":
            agent_goal = plus_z_goal
            opp_goal   = minus_z_goal
        else:
            agent_goal = minus_z_goal
            opp_goal   = plus_z_goal

        ball_out = (
            abs(bx) > FIELD["half_width"] + 0.05
            or (bz >= FIELD["goal_z_attack"] and not plus_z_goal)
            or (bz <= FIELD["goal_z_own"]    and not minus_z_goal)
        )

        return {
            "agent_goal": agent_goal,
            "opp_goal":   opp_goal,
            "ball_out":   ball_out,
            # Keep aliases for backward compat with train callback
            "goal_scored": agent_goal,
            "own_goal":    opp_goal,
        }

    def _is_near_post(self, robot_pos: tuple) -> bool:
        hw = FIELD["goal_half_width"]
        for px, pz in (
            ( hw, FIELD["goal_z_attack"]),
            (-hw, FIELD["goal_z_attack"]),
            ( hw, FIELD["goal_z_own"]),
            (-hw, FIELD["goal_z_own"]),
        ):
            if math.hypot(robot_pos[0] - px, robot_pos[1] - pz) < 0.30:
                return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # Spawn helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _spawn_1v0_style(
        self, attack_z: float
    ) -> tuple[float, float, float, float]:
        """Ball and agent spawn using the 1v0 progressive curriculum.

        Returns (bx, bz, rx, rz).  The sign of attack_z flips the curriculum
        so it works identically for Viper (+Z) and Titan (−Z).
        """
        cs   = self._curriculum_step
        gz   = attack_z
        ghw  = FIELD["goal_half_width"]
        sign = 1.0 if attack_z > 0 else -1.0

        if cs < 400_000:
            # Phase 1a — ball just in front of goal
            bz_abs = self._rng.uniform(abs(gz) - 0.20, abs(gz) - 0.05)
            bx     = self._rng.uniform(-ghw * 0.70, ghw * 0.70)
            rz_abs = bz_abs - self._rng.uniform(0.15, 0.40)
            rx     = float(bx) + self._rng.uniform(-0.15, 0.15)
            bz     = sign * bz_abs
            rz     = sign * rz_abs

        elif cs < 520_000:
            bz_abs = self._rng.uniform(abs(gz) - 1.50, abs(gz) - 0.50)
            bx     = self._rng.uniform(-ghw, ghw)
            rz_abs = max(bz_abs - 1.20, 0.0)
            rz_abs = self._rng.uniform(rz_abs, bz_abs - 0.20)
            rx     = float(bx) + self._rng.uniform(-0.30, 0.30)
            bz     = sign * bz_abs
            rz     = sign * rz_abs

        elif cs < 640_000:
            fl = FIELD["half_length"]
            fw = FIELD["half_width"]
            bz = sign * self._rng.uniform(0.0, fl * 0.80)
            bx = self._rng.uniform(-fw * 0.60, fw * 0.60)
            rz = sign * self._rng.uniform(0.0, abs(float(bz)) - 0.20) if abs(float(bz)) > 0.20 else 0.0
            rx = self._rng.uniform(-fw * 0.70, fw * 0.70)

        else:
            fl = FIELD["half_length"]
            fw = FIELD["half_width"]
            bx = self._rng.uniform(-fw * 0.60, fw * 0.60)
            bz = sign * self._rng.uniform(0.0, fl * 0.40)
            rx = self._rng.uniform(-fw * 0.70, fw * 0.70)
            rz = sign * self._rng.uniform(0.50, fl * 0.85)

        return float(bx), float(bz), float(rx), float(rz)

    def _spawn_phase2(
        self, attack_z: float, own_z: float
    ) -> tuple[float, float, float, float, float, float]:
        """Phase 2 spawns.  Agent in own half, ball in attack half, opponent in its half."""
        fl = FIELD["half_length"]
        fw = FIELD["half_width"]
        sign = 1.0 if attack_z > 0 else -1.0

        bx = self._rng.uniform(-fw * 0.60, fw * 0.60)
        bz = sign * self._rng.uniform(0.20, fl * 0.70)
        rx = self._rng.uniform(-fw * 0.70, fw * 0.70)
        rz = -sign * self._rng.uniform(0.30, fl * 0.85)
        # Opponent (scripted) in its own half, somewhat away from ball
        ox = self._rng.uniform(-fw * 0.60, fw * 0.60)
        oz = -sign * self._rng.uniform(0.30, fl * 0.80)

        # Ensure no robot–robot overlap
        for _ in range(20):
            if math.hypot(rx - ox, rz - oz) >= _MIN_ROBOT_SEP:
                break
            ox = self._rng.uniform(-fw * 0.60, fw * 0.60)
            oz = -sign * self._rng.uniform(0.30, fl * 0.80)

        return float(bx), float(bz), float(rx), float(rz), float(ox), float(oz)

    def _spawn_phase3(
        self, attack_z: float, own_z: float
    ) -> tuple[float, float, float, float, float, float]:
        """Phase 3 — fully random with collision avoidance."""
        fl = FIELD["half_length"]
        fw = FIELD["half_width"]
        sign = 1.0 if attack_z > 0 else -1.0

        bx = self._rng.uniform(-fw * 0.60, fw * 0.60)
        bz = self._rng.uniform(-fl * 0.80, fl * 0.80)
        rx = self._rng.uniform(-fw * 0.70, fw * 0.70)
        rz = self._rng.uniform(-fl * 0.85, fl * 0.85)

        for _ in range(30):
            ox = self._rng.uniform(-fw * 0.70, fw * 0.70)
            oz = self._rng.uniform(-fl * 0.85, fl * 0.85)
            if (math.hypot(rx - ox, rz - oz) >= _MIN_ROBOT_SEP
                    and math.hypot(bx - ox, bz - oz) >= 0.30):
                break

        return float(bx), float(bz), float(rx), float(rz), float(ox), float(oz)

    def _spawn_rotation_for(self, attack_z: float) -> list[float]:
        """Rotation that makes the robot face its attack goal direction.

        Phase 1a curriculum (first 400k steps): explicit facing.
        Later phases: default upright rotation (agent learns to rotate freely).
        """
        if self._curriculum_step < 400_000:
            # Face +Z if attack_z > 0 (Viper), face −Z if attack_z < 0 (Titan)
            if attack_z > 0:
                _INV_SQRT3 = 1.0 / math.sqrt(3)
                return [-_INV_SQRT3, _INV_SQRT3, _INV_SQRT3, 2.0 * math.pi / 3.0]
            else:
                # Ry(π)·Rx(−90°): face −Z
                return [0.0, 0.0, 1.0, math.pi]
        return [1.0, 0.0, 0.0, -math.pi / 2]

    # ══════════════════════════════════════════════════════════════════════════
    # IPC helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _send_agent_action(self, vx: float, vz: float, omega: float) -> None:
        if self._active_robot == "viper":
            self._send_to(self._emitter_viper, vx, vz, omega)
        else:
            self._send_to(self._emitter_titan, vx, vz, omega)

    def _send_opp_action(self, vx: float, vz: float, omega: float) -> None:
        if self._active_robot == "viper":
            self._send_to(self._emitter_titan, vx, vz, omega)
        else:
            self._send_to(self._emitter_viper, vx, vz, omega)

    def _send_to(self, emitter, vx: float, vz: float, omega: float) -> None:
        payload = base64.b64encode(
            struct.pack(_ACTION_FMT, vx, vz, omega)
        ).decode("ascii")
        emitter.send(payload)

    def _drain_receivers(self) -> None:
        """Drain both lidar queues to prevent overflow. Data is stored but unused."""
        for receiver, buf_attr in (
            (self._receiver_viper, "_last_lidar_viper"),
            (self._receiver_titan, "_last_lidar_titan"),
        ):
            latest: bytes | None = None
            while receiver.getQueueLength() > 0:
                try:
                    raw = base64.b64decode(receiver.getString())
                    if len(raw) == _SENSOR_BYTES:
                        latest = raw
                except Exception:
                    pass
                receiver.nextPacket()
            if latest is not None:
                vals = struct.unpack(_SENSOR_FMT, latest)
                arr  = np.array(vals[1:], dtype=np.float32)
                # Determine which robot this packet belongs to from the robot_id field
                robot_id = int(vals[0])
                max_r    = ROBOT_CONFIGS["viper" if robot_id == 0 else "titan"]["lidar_max_range"]
                setattr(self, buf_attr, np.clip(arr / max_r, 0.0, 1.0))

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _sim_step(self) -> None:
        Supervisor.step(self, self._timestep)

    # ── Active/opponent node selectors ────────────────────────────────────────

    def _agent_node(self):
        return self._viper_node if self._active_robot == "viper" else self._titan_node

    def _opp_node(self):
        return self._titan_node if self._active_robot == "viper" else self._viper_node

    def _agent_attack_z(self) -> float:
        return float(FIELD["goal_z_attack"] if self._active_robot == "viper"
                     else FIELD["goal_z_own"])

    def _agent_own_z(self) -> float:
        return float(FIELD["goal_z_own"] if self._active_robot == "viper"
                     else FIELD["goal_z_attack"])

    def _opp_attack_z(self) -> float:
        return float(FIELD["goal_z_own"] if self._active_robot == "viper"
                     else FIELD["goal_z_attack"])

    # Required gym stubs
    def render(self) -> None: pass
    def close(self) -> None:  pass


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _flat(node) -> tuple[float, float]:
    """(X, Z) ground-plane position from a Webots node (NUE: Y is up)."""
    p = node.getPosition()
    return (p[0], p[2])


def _vec2d(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    dx, dz = b[0] - a[0], b[1] - a[1]
    dist   = math.hypot(dx, dz)
    if dist > 1e-6:
        return dist, (dx / dist, dz / dist)
    return 0.0, (0.0, 0.0)


def _get_heading_from_node(node) -> float:
    """Yaw in radians from a Webots node orientation matrix (NUE, Rx−90° convention)."""
    m = node.getOrientation()
    return math.atan2(-m[1], m[0])


def _opp_between_agent_goal(
    agent_pos: tuple[float, float],
    opp_pos:   tuple[float, float],
    goal_pos:  tuple[float, float],
    corridor_half_width: float = 0.50,
) -> float:
    """Return 1.0 if the opponent lies in the agent→goal corridor, else 0.0.

    The corridor is defined as the set of points whose projection onto the
    agent→goal segment falls between 10% and 90% of the segment length AND
    whose perpendicular distance from the segment is < corridor_half_width.
    """
    ax, az = agent_pos
    gx, gz = goal_pos
    ox, oz = opp_pos
    dx, dz = gx - ax, gz - az
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return 0.0
    t = ((ox - ax) * dx + (oz - az) * dz) / (length * length)
    if not (0.10 < t < 0.90):
        return 0.0
    perp = abs((ox - ax) * dz - (oz - az) * dx) / length
    return 1.0 if perp < corridor_half_width else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Webots entry point
# ══════════════════════════════════════════════════════════════════════════════

env = SoccerEnv1v1()

if MODE == "train":
    from train_1v1 import train_1v1
    train_1v1(env)
elif MODE == "eval":
    from eval_1v1 import eval_1v1
    eval_1v1(env)
