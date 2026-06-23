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

Observation space (gymnasium.spaces.Box)
────────────────────────────────────────
  Box(19,)  [type_id,
              dist_ball_n, dir_bx, dir_bz,     ← LOCAL frame
              dist_goal_n, dir_gx, dir_gz,     ← LOCAL frame
              dist_ar_n, dir_ar_x, dir_ar_z,   ← attack right post  (LOCAL)
              dist_al_n, dir_al_x, dir_al_z,   ← attack left post   (LOCAL)
              dist_or_n, dir_or_x, dir_or_z,   ← own right post     (LOCAL)
              dist_ol_n, dir_ol_x, dir_ol_z]   ← own left post      (LOCAL)

  Todas as direções são rotacionadas para o frame LOCAL do robô antes de
  entrar no obs. Assim dir_bz > 0 significa "bola à frente" e a rede
  mapeia diretamente observação → ação sem aprender rotação implícita.
  (LiDAR é transmitido por IPC mas não entra na política.)

  (LiDAR is bumped to 1440 rays and still transmitted over IPC so the
   robot controller can detect obstacles, but goal-post detection is done
   analytically from GPS in the supervisor instead of through CNN.)

Action space
────────────
  Box(3,)  in [-1, 1]  →   scaled to [vx, vz, omega]

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
from collections import deque


print("Python executable:", sys.executable)
#print("Python version:", sys.version)

import numpy as np

# ── Bootstrap shared_configs (controllers/ is one level up) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared_configs import BALL, FIELD, IPC, MODE, ROBOT_CONFIGS, SIM, get_robot_config

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

MAX_LINEAR    = 0.5   # m/s    — scale factor for normalised vx / vz action
MAX_ANGULAR   = 3.0   # rad/s  — scale factor for normalised omega action
MAX_BALL_SPEED = 4.0  # m/s    — clamp for normalising ball velocity in obs

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
        self._max_steps     = 1000  # 40 s / 0.04 s per step

        # ── Webots node handles ────────────────────────────────────────────
        self._ball_node  = self.getFromDef("BOLA")
        self._robot_node = self.getFromDef("VIPER")

        # ── IPC devices (declared in the Supervisor node in soccer.wbt) ───
        self._emitter  = self.getDevice("supervisor_emitter")
        self._receiver = self.getDevice("supervisor_receiver")
        self._receiver.enable(self._timestep)

        # ── Gymnasium spaces ───────────────────────────────────────────────
        # "vector" bounds are per-component (19 dimensions, LOCAL frame):
        #   [0]       type_id             ∈ {0, 1}    → [0, 1]
        #   [1]       dist_ball_norm      ∈ [0, 1]
        #   [2–3]     dir_ball (local)    ∈ [-1, 1] each
        #   [4]       dist_goal_norm      ∈ [0, 1]
        #   [5–6]     dir_goal (local)    ∈ [-1, 1] each
        #   [7–9]     attack-right post   dist_norm, dir_x, dir_z  (local)
        #   [10–12]   attack-left  post   dist_norm, dir_x, dir_z  (local)
        #   [13–15]   own-right    post   dist_norm, dir_x, dir_z  (local)
        #   [16–18]   own-left     post   dist_norm, dir_x, dir_z  (local)
        # Pattern per post: [0,1], [-1,1], [-1,1]
        _post_low  = [0., -1., -1.] * 4   # 12 values for 4 posts
        _post_high = [1.,  1.,  1.] * 4
        # [19-21] robot vel (vx, vz, omega) LOCAL, normalised to [-1, 1]
        # [22-23] ball vel (vbx, vbz) LOCAL, normalised to [-1, 1]
        _vel_low  = [-1., -1., -1., -1., -1.]
        _vel_high = [ 1.,  1.,  1.,  1.,  1.]
        _obs_low  = np.array(
            [0., 0., -1., -1., 0., -1., -1.] + _post_low  + _vel_low,  dtype=np.float32
        )
        _obs_high = np.array(
            [1., 1.,  1.,  1., 1.,  1.,  1.] + _post_high + _vel_high, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=_obs_low, high=_obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Episode state ──────────────────────────────────────────────────
        self._active_robot        : str   = "viper"
        self._step_count          : int   = 0
        self._still_steps         : int   = 0
        self._post_stuck_steps    : int   = 0
        self._last_lidar          = np.ones(_N_LIDAR, dtype=np.float32)
        self._prev_dist_ball      : float = FIELD_DIAG
        self._prev_dist_ball_goal : float = FIELD_DIAG
        self._prev_robot_pos      : tuple = (0.0, 0.0)
        self._prev_ball_z         : float = 0.0
        self._rng                         = np.random.default_rng()

        # ── Curriculum por competência (8 fases) ───────────────────────────
        # Fase 0: bola 0.05–0.20 m do gol   Fase 1: 0.20–0.50 m
        # Fase 2: 0.50–1.00 m               Fase 3: 1.00–1.75 m
        # Fase 4: 1.75–2.75 m               Fase 5: 2.75–4.00 m
        # Fase 6: campo de ataque (z>0)      Fase 7: completamente aleatório
        self._N_PHASES           : int   = 8
        self._phase              : int   = 0
        self._MAX_STEPS_BY_PHASE : list  = [350, 500, 650, 800, 950, 1100, 1200, 1250]
        self._GOAL_WINDOW        : int   = 40   # episódios na janela de avaliação
        self._PROMOTE_THRESH     : float = 0.50 # taxa de gols p/ promover
        self._goal_history       : deque = deque(maxlen=40)

        # Contador mantido p/ compatibilidade e diagnóstico em train.py
        self._curriculum_step : int = 0

        # ── Bônus de meio campo (uma vez por episódio) ─────────────────────
        self._midfield_bonus_given : bool = False

        # ── Histórico de posições ──────────────────────────────────────────
        self._pos_history : deque = deque(maxlen=150)

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

        self._step_count          = 0
        self._still_steps         = 0
        self._post_stuck_steps    = 0
        self._midfield_bonus_given = False
        self._pos_history.clear()

        # ── Curriculum por competência: spawn baseado na fase atual ────────
        ph   = self._phase
        _gz  = FIELD["goal_z_attack"]    # 4.55
        _ghw = FIELD["goal_half_width"]  # 0.75
        _HW  = FIELD["half_width"]       # 3.70
        _HL  = FIELD["half_length"]      # 5.20

        self._max_steps = self._MAX_STEPS_BY_PHASE[ph]

        # Helper: place robot just behind the ball (natural pushing posture)
        def _behind(bx_, bz_, lo, hi, jit):
            return (float(bx_) + self._rng.uniform(-jit, jit),
                    float(bz_) - self._rng.uniform(lo, hi))

        if ph == 0:                                       # bola 0.05–0.20 m do gol
            bz = self._rng.uniform(_gz - 0.20, _gz - 0.05)
            bx = self._rng.uniform(-_ghw * 0.70, _ghw * 0.70)
            rx, rz = _behind(bx, bz, 0.15, 0.40, 0.15)
        elif ph == 1:                                     # 0.20–0.50 m
            bz = self._rng.uniform(_gz - 0.50, _gz - 0.20)
            bx = self._rng.uniform(-_ghw, _ghw)
            rx, rz = _behind(bx, bz, 0.15, 0.45, 0.20)
        elif ph == 2:                                     # 0.50–1.00 m
            bz = self._rng.uniform(_gz - 1.00, _gz - 0.50)
            bx = self._rng.uniform(-_ghw, _ghw)
            rx, rz = _behind(bx, bz, 0.20, 0.70, 0.30)
        elif ph == 3:                                     # 1.00–1.75 m
            bz = self._rng.uniform(_gz - 1.75, _gz - 1.00)
            bx = self._rng.uniform(-_HW * 0.60, _HW * 0.60)
            rx, rz = _behind(bx, bz, 0.25, 0.90, 0.35)
        elif ph == 4:                                     # 1.75–2.75 m
            bz = self._rng.uniform(_gz - 2.75, _gz - 1.75)
            bx = self._rng.uniform(-_HW * 0.70, _HW * 0.70)
            rx, rz = _behind(bx, bz, 0.30, 1.20, 0.45)
        elif ph == 5:                                     # 2.75–4.00 m
            bz = self._rng.uniform(_gz - 4.00, _gz - 2.75)
            bx = self._rng.uniform(-_HW * 0.70, _HW * 0.70)
            rx, rz = _behind(bx, bz, 0.40, 1.60, 0.60)
            rz = max(rz, -_HL * 0.80)
        elif ph == 6:                                     # campo de ataque (z > 0)
            bz = self._rng.uniform(0.0, _HL * 0.80)
            bx = self._rng.uniform(-_HW * 0.60, _HW * 0.60)
            rz_max = max(float(bz) - 0.20, -0.20)
            rz = self._rng.uniform(-_HL * 0.85, rz_max)
            rx = self._rng.uniform(-_HW * 0.70, _HW * 0.70)
        else:                                             # fase 7 — completamente aleatório
            bx = self._rng.uniform(-_HW * 0.60, _HW * 0.60)
            bz = self._rng.uniform(-_HL * 0.40, _HL * 0.40)
            rx = self._rng.uniform(-_HW * 0.70, _HW * 0.70)
            rz = self._rng.uniform(-_HL * 0.85, -0.5)

        self._ball_node.getField("translation").setSFVec3f(
            [float(bx), BALL["radius"], float(bz)]
        )
        self._ball_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._ball_node.resetPhysics()

        # ── Place robot ────────────────────────────────────────────────────
        self._robot_node.getField("translation").setSFVec3f(
            [float(rx), 0.0, float(rz)]
        )
        # Phases 0–1 (trivial): face the attack goal to bootstrap learning.
        # Ry(π/2)·Rx(−90°) = axis-angle [−1/√3, 1/√3, 1/√3, 2π/3] → faces +Z.
        _INV_SQRT3 = 1.0 / math.sqrt(3)
        if ph <= 1:
            self._robot_node.getField("rotation").setSFRotation(
                [-_INV_SQRT3, _INV_SQRT3, _INV_SQRT3, 2.0 * math.pi / 3.0]
            )
        else:
            self._robot_node.getField("rotation").setSFRotation(
                [1.0, 0.0, 0.0, -math.pi / 2]
            )
        self._robot_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._robot_node.resetPhysics()

        # ── Settle physics (send zero action for a few steps) ──────────────
        for _ in range(15):
            self._send_action(0.0, 0.0, 0.0)
            self._sim_step()

        # Guard against two physics glitches that can occur during settle steps:
        #   1. Robot sinks below ground  (Y < -0.02 m).
        #   2. Robot tips over / spawns perpendicular to the floor.
        #      getOrientation() returns a 3×3 row-major rotation matrix.
        #      m[4] is the Y-Y component: ≈ 1.0 when perfectly upright, → 0
        #      when tipped ≥ 90°.  Require m[4] > 0.7 (tilt < ~45°).
        _m      = self._robot_node.getOrientation()
        _sunk   = self._robot_node.getPosition()[1] < -0.02
        _tipped = _m[4] < 0.7
        if _sunk or _tipped:
            self._robot_node.getField("translation").setSFVec3f(
                [float(rx), 0.0, float(rz)]
            )
            self._robot_node.getField("rotation").setSFRotation(
                [1.0, 0.0, 0.0, -math.pi / 2]
            )
            self._robot_node.setVelocity([0, 0, 0, 0, 0, 0])
            self._robot_node.resetPhysics()
            for _ in range(5):
                self._send_action(0.0, 0.0, 0.0)
                self._sim_step()

        self._drain_receiver()  # discard lidar from settle steps

        # ── Compute initial distances ──────────────────────────────────────
        ball_pos = _flat(self._ball_node)
        obs      = self._get_obs()

        self._prev_dist_ball      = float(obs[1]) * FIELD_DIAG
        _hw = FIELD["goal_half_width"]
        _tx = float(np.clip(ball_pos[0], -_hw, _hw))
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0] - _tx, ball_pos[1] - FIELD["goal_z_attack"]
        )
        self._prev_robot_pos = _flat(self._robot_node)
        self._prev_ball_z    = ball_pos[1]

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
        dist_ball = float(obs[1]) * FIELD_DIAG

        # ── Robot displacement since last step ─────────────────────────────
        moved = math.hypot(
            robot_pos[0] - self._prev_robot_pos[0],
            robot_pos[1] - self._prev_robot_pos[1],
        )

        # ── Goal-post stuck detection ──────────────────────────────────────
        if self._is_near_post(robot_pos) and moved < 0.003:
            self._post_stuck_steps += 1
        else:
            self._post_stuck_steps = max(0, self._post_stuck_steps - 2)

        # ── Events & terminal flags ────────────────────────────────────────
        events     = self._check_events(ball_pos)
        terminated = events["goal_scored"] or events["own_goal"] or events["ball_out"]
        truncated  = (self._step_count >= self._max_steps) or (self._post_stuck_steps > 100)

        # ── Reward (uses OLD _prev_* values) ──────────────────────────────
        reward = self._compute_reward_2(
            dist_ball, ball_pos, robot_pos, moved, events
        )

        # ── Update prev state for next step ───────────────────────────────
        self._prev_dist_ball      = dist_ball
        _hw2 = FIELD["goal_half_width"]
        _tx2 = float(np.clip(ball_pos[0], -_hw2, _hw2))
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0] - _tx2, ball_pos[1] - FIELD["goal_z_attack"]
        )
        self._prev_robot_pos  = robot_pos
        self._prev_ball_z     = ball_pos[1]
        self._curriculum_step += 1

        # ── Curriculum por competência: promoção de fase ───────────────────
        if terminated or truncated:
            self._goal_history.append(1 if events["goal_scored"] else 0)
            if (len(self._goal_history) >= self._GOAL_WINDOW
                    and self._phase < self._N_PHASES - 1):
                goal_rate = sum(self._goal_history) / len(self._goal_history)
                if goal_rate >= self._PROMOTE_THRESH:
                    self._phase += 1
                    self._goal_history.clear()
                    print(f"[Curriculum] >>> PROMOVIDO para a fase "
                          f"{self._phase}/{self._N_PHASES - 1} "
                          f"(goal_rate={goal_rate:.0%}, robot={self._active_robot})")

        info = {**events, "step": self._step_count, "phase": self._phase}
        return obs, float(reward), terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # Observation
    # ══════════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        cfg       = get_robot_config(self._active_robot)
        robot_pos = _flat(self._robot_node)
        ball_pos  = _flat(self._ball_node)
        goal_pos  = (0.0, float(FIELD["goal_z_attack"]))

        # ── Heading do robô para rotacionar para frame local ───────────────
        heading    = self._get_heading()
        cos_h      = math.cos(heading)
        sin_h      = math.sin(heading)

        dist_ball, dir_ball_w = _vec2d(robot_pos, ball_pos)
        dist_goal, dir_goal_w = _vec2d(robot_pos, goal_pos)

        # Rotaciona direção do frame mundo → frame local do robô (Ry(−θ)).
        # local_x =  cos θ·dx + sin θ·dz   → dir_bz > 0 significa bola à frente.
        # local_z = −sin θ·dx + cos θ·dz
        def to_local(dx: float, dz: float) -> tuple[float, float]:
            return ( cos_h * dx + sin_h * dz,
                    -sin_h * dx + cos_h * dz)

        dir_ball = to_local(dir_ball_w[0], dir_ball_w[1])
        dir_goal = to_local(dir_goal_w[0], dir_goal_w[1])

        # ── 4 postes do gol em frame local ────────────────────────────────
        hw = FIELD["goal_half_width"]
        _posts = [
            ( hw, FIELD["goal_z_attack"]),   # attack right
            (-hw, FIELD["goal_z_attack"]),   # attack left
            ( hw, FIELD["goal_z_own"]),      # own right
            (-hw, FIELD["goal_z_own"]),      # own left
        ]
        post_feats: list[float] = []
        for px, pz in _posts:
            d, uv_w = _vec2d(robot_pos, (px, pz))
            uv_l = to_local(uv_w[0], uv_w[1])
            post_feats += [d / FIELD_DIAG, uv_l[0], uv_l[1]]

        # ── Robot velocity in local frame [19-21] ─────────────────────────
        try:
            rv      = self._robot_node.getVelocity()  # [vx,vy,vz,wx,wy,wz] world NUE
            rvx_l, rvz_l = to_local(rv[0], rv[2])
            omega_l = float(rv[4])                    # yaw rate around Y-axis
        except Exception:
            rvx_l = rvz_l = omega_l = 0.0
        rvx_n   = float(np.clip(rvx_l   / MAX_LINEAR,   -1.0, 1.0))
        rvz_n   = float(np.clip(rvz_l   / MAX_LINEAR,   -1.0, 1.0))
        omega_n = float(np.clip(omega_l  / MAX_ANGULAR,  -1.0, 1.0))

        # ── Ball velocity in local frame [22-23] ──────────────────────────
        try:
            bv      = self._ball_node.getVelocity()   # [vx,vy,vz,wx,wy,wz] world NUE
            bvx_l, bvz_l = to_local(bv[0], bv[2])
        except Exception:
            bvx_l = bvz_l = 0.0
        bvx_n = float(np.clip(bvx_l / MAX_BALL_SPEED, -1.0, 1.0))
        bvz_n = float(np.clip(bvz_l / MAX_BALL_SPEED, -1.0, 1.0))

        obs = np.array(
            [
                float(cfg["type_id"]),       # [0]   0=viper / 1=titan
                dist_ball / FIELD_DIAG,      # [1]   dist normalizada robô→bola
                dir_ball[0],                 # [2]   dir_x LOCAL
                dir_ball[1],                 # [3]   dir_z LOCAL  (>0 = bola à frente)
                dist_goal / FIELD_DIAG,      # [4]   dist normalizada robô→gol
                dir_goal[0],                 # [5]   dir_x LOCAL
                dir_goal[1],                 # [6]   dir_z LOCAL  (>0 = gol à frente)
            ] + post_feats +                 # [7-18] 4 postes × (dist, dir_x, dir_z) LOCAL
            [
                rvx_n, rvz_n, omega_n,       # [19-21] velocidade robô (local, normalizada)
                bvx_n, bvz_n,                # [22-23] velocidade bola (local, normalizada)
            ],
            dtype=np.float32,
        )
        return obs

    def _get_heading(self) -> float:
        """
        Extrai o yaw (rotação em torno do eixo Y global) do robô.

        Webots usa NUE: X=leste, Y=cima, Z=sul.
        getOrientation() retorna uma matriz de rotação 3×3 em row-major:
            [m0 m1 m2]
            [m3 m4 m5]
            [m6 m7 m8]

        O proto do robô tem rotação base de -90° em X (para ficar em pé no
        plano XZ). A orientação efectiva no plano horizontal é uma rotação
        Ry(θ) composta com Rx(-90°). Após a álgebra, o yaw θ no plano XZ é:
            yaw = atan2(-m[1], m[0])
        onde m[0] = cos θ  e  m[1] = -sin θ  (colunas da linha 0 de Ry(θ)·Rx(-90°)).

        Retorna yaw em radianos ∈ (-π, π].
        """
        m = self._robot_node.getOrientation()   # lista de 9 floats, row-major
        return math.atan2(-m[1], m[0])

    # ══════════════════════════════════════════════════════════════════════════
    # Reward
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_reward_2(
        self,
        dist_ball: float,
        ball_pos: tuple,
        robot_pos: tuple,
        moved: float,
        events: dict,
    ) -> float:
        """
        Reward hierarchy (dominant first):
          1. Score goal          → dominant terminal signal (+300 / -200 / -100)
          2. Ball toward goal    → main dense shaping
          3. Robot reaches ball  → prerequisite shaping
          4. Positioning         → alignment / behind-ball bonuses
          5. Anti-stall          → stillness, no-progress, post-stuck penalties
        """
        # ── 1. TERMINAL EVENTS ──────────────────────────────────────────────────
        if events["goal_scored"]:
            return +300.0
        if events["own_goal"]:
            return -200.0
        if events["ball_out"]:
            return -25.0

        bx, bz   = ball_pos
        rx, rz   = robot_pos
        GOAL_Z   = FIELD["goal_z_attack"]
        TOUCH_TH = 0.16
        reward   = 0.0

        # ── 2. TIME PENALTY ──────────────────────────────────────────────────────
        reward -= 0.003

        # ── 3. MIDFIELD BONUS (once per episode, só fase 7) ──────────────────────
        # Só faz sentido quando a bola pode começar no campo defensivo (fase 7).
        if (not self._midfield_bonus_given
                and self._phase >= 7
                and self._prev_ball_z < 0.0 and bz >= 0.0):
            self._midfield_bonus_given = True
            reward += 1.0

        # ── 4. ROBOT → BALL PROGRESS ─────────────────────────────────────────────
        # ×4.0 — gradiente forte para aprender a navegar até à bola.
        reward += (self._prev_dist_ball - dist_ball) * 4.0

        # ── 5. BALL → GOAL PROGRESS (ponto aberto mais próximo das traves) ────────
        # max(0, Δ): só recompensa avanço da bola; recuo não pune o agente.
        # Sem esta trava o robô aprende a fugir da bola (recuo punido → evita toque).
        _hw = FIELD["goal_half_width"]
        _target_x = float(np.clip(bx, -_hw, _hw))
        dist_ball_goal = math.hypot(bx - _target_x, bz - GOAL_Z)
        reward += max(0.0, self._prev_dist_ball_goal - dist_ball_goal) * 15.0

        # ── 6. BALL VELOCITY TOWARD GOAL ─────────────────────────────────────────
        try:
            vel = self._ball_node.getVelocity()
            ball_vx, ball_vz = vel[0], vel[2]
            to_goal_x, to_goal_z = -bx, GOAL_Z - bz
            norm = math.hypot(to_goal_x, to_goal_z)
            if norm > 1e-6:
                to_goal_x /= norm
                to_goal_z /= norm
                toward_goal = ball_vx * to_goal_x + ball_vz * to_goal_z
                reward += float(np.clip(toward_goal * 0.8, -1.0, 1.0))
        except Exception:
            pass

        # ── 7. ROBOT BEHIND BALL ─────────────────────────────────────────────────
        # REMOVIDO: o bonus ±0.05 por estar atrás da bola em Z causava o robô
        # a fazer um percurso longo pelo campo para chegar ao lado "certo",
        # em vez de reposicionar rapidamente. O alignment bonus (secção 8)
        # já cobre o posicionamento correto de forma mais precisa.

        # ── 8. ROBOT–BALL–GOAL ALIGNMENT ─────────────────────────────────────────
        rb_x, rb_z = bx - rx, bz - rz
        rb_norm = math.hypot(rb_x, rb_z)
        if rb_norm > 1e-6:
            rb_x /= rb_norm
            rb_z /= rb_norm
            bg_x, bg_z = -bx, GOAL_Z - bz
            bg_norm = math.hypot(bg_x, bg_z)
            if bg_norm > 1e-6:
                bg_x /= bg_norm
                bg_z /= bg_norm
                reward += (rb_x * bg_x + rb_z * bg_z) * 0.3

        # ── 9. CONTACT BONUS (condicionado à direção do toque) ───────────────────
        # +0.25 só se a bola está a ir em direção ao gol após o toque.
        # +0.05 por qualquer toque (incentiva chegar à bola, sem reforçar toques errados).
        if dist_ball < TOUCH_TH:
            if (self._step_count - getattr(self, "_last_touch_step", -999)) > 8:
                try:
                    _bvel = self._ball_node.getVelocity()
                    _to_goal_z = GOAL_Z - bz
                    _to_goal_x = _target_x - bx
                    _ng = math.hypot(_to_goal_x, _to_goal_z)
                    if _ng > 1e-6:
                        _dot = (_bvel[0] * _to_goal_x + _bvel[2] * _to_goal_z) / _ng
                        reward += 0.25 if _dot > 0.1 else 0.05
                    else:
                        reward += 0.05
                except Exception:
                    reward += 0.05
                self._last_touch_step = self._step_count

        # ── 10. BALL-HOVER PENALTY (near ball but ball not moving) ────────────────
        try:
            ball_speed = math.hypot(*self._ball_node.getVelocity()[:3:2])
            if dist_ball < 0.25 and ball_speed < 0.03:
                reward -= 0.05
        except Exception:
            pass

        # ── 11. ROBOT STILLNESS PENALTY ──────────────────────────────────────────
        if moved < 0.003:
            self._still_steps += 1
            if self._still_steps > 20:      # trigger after ~0.8 s motionless
                reward -= 0.10
        else:
            self._still_steps = 0

        # ── 12. LONG-RANGE NO-PROGRESS PENALTY ───────────────────────────────────
        # REMOVIDO: penalizava o robô por ficar perto da bola (Phase 1a: zona
        # de 0.30m). O -0.08/step era 13× maior que o approach reward (×0.3),
        # forçando o robô a afastar-se para evitar a penalidade.
        # O stillness penalty (secção 11) já cobre robôs verdadeiramente parados.
        self._pos_history.append(robot_pos)  # mantido para compatibilidade

        # ── 13. GOAL-POST STUCK PENALTY ──────────────────────────────────────────
        if self._post_stuck_steps > 20:
            reward -= 0.30

        return reward


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

    def _is_near_post(self, robot_pos: tuple) -> bool:
        """Return True if the robot centre is within 0.30 m of any goal post."""
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

    # Required gym.Env stubs

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


# Module-level helpers

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
    from the robot to the attack goal centre.
    """
    rx, rz = robot_pos
    bx, bz = ball_pos
    if not (min(rz, goal_z) < bz < max(rz, goal_z)):
        return 0.0
    # Lateral distance from ball to the line robot→goal_centre
    dz = goal_z - rz
    dx = 0.0 - rx  # goal centre is at x=0
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return 0.0
    lateral = abs((bx - rx) * dz - (bz - rz) * dx) / length
    return 0.05 if lateral < 0.30 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — Webots runs this file as the supervisor controller
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from train import train
    from eval import model_evaluate

    env = SoccerEnv()

    if MODE == "train":
        train(env)
    else:
        model_evaluate(env)