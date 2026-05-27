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


#print("Python executable:", sys.executable)
#print("Python version:", sys.version)

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
        self._max_steps     = 375 #15 seconds/0.04   //#SIM["max_episode_steps"]       # 2000

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
        _obs_low  = np.array(
            [0., 0., -1., -1., 0., -1., -1.] + _post_low,  dtype=np.float32
        )
        _obs_high = np.array(
            [1., 1.,  1.,  1., 1.,  1.,  1.] + _post_high, dtype=np.float32
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
        self._last_lidar          = np.ones(_N_LIDAR, dtype=np.float32)
        self._prev_dist_ball      : float = FIELD_DIAG
        self._prev_dist_ball_goal : float = FIELD_DIAG
        self._prev_robot_pos      : tuple = (0.0, 0.0)
        self._prev_ball_z         : float = 0.0   # z da bola no passo anterior
        self._rng                         = np.random.default_rng()

        # ── Curriculum de posições ─────────────────────────────────────────
        # Contador global de passos — incrementado em step(), nunca resetado.
        # Fase 1 (0–100k)  : bola perto do gol, robô atrás da bola
        # Fase 2 (100k–300k): bola no campo de ataque, robô aleatório
        # Fase 3 (300k+)   : completamente aleatório
        self._curriculum_step : int = 0

        # ── Bônus de meio campo (uma vez por episódio) ─────────────────────
        # Dado quando a bola cruza de z<0 para z>0 (campo de ataque)
        self._midfield_bonus_given : bool = False

        # ── Histórico de posições para penalidade de 10 segundos ──────────
        # 10 s ÷ 40 ms/step = 250 steps
        _WINDOW = 250
        self._pos_history : deque = deque(maxlen=_WINDOW)
        self._NO_PROGRESS_WINDOW  = _WINDOW          # passos na janela
        self._NO_PROGRESS_THRESH  = 0.30             # metros de deslocamento líquido mínimo
        self._NO_PROGRESS_PENALTY = -0.02            # reward por passo sem progresso

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
        self._midfield_bonus_given = False
        self._pos_history.clear()

        # ── Curriculum de posições ─────────────────────────────────────────
        # Fase 1a (0–30k)   : bola a 5–20 cm do gol, dentro das traves → gol quase garantido
        # Fase 1b (30k–100k): bola a 0.5–1.5 m do gol, robô logo atrás
        # Fase 2 (100k–300k): bola no campo de ataque (z>0), robô aleatório
        # Fase 3 (300k+)    : completamente aleatório
        cs = self._curriculum_step
        _gz  = FIELD["goal_z_attack"]    # 4.55
        _ghw = FIELD["goal_half_width"]  # 0.75

        if cs < 30_000:
            # Fase 1a — trivial: bola quase dentro do gol.
            # Qualquer toque no +Z marca gol → PPO descobre o +50 rapidamente.
            bz = self._rng.uniform(_gz - 0.20, _gz - 0.05)   # 4.35 → 4.50
            bx = self._rng.uniform(-_ghw * 0.70, _ghw * 0.70) # dentro das traves
            # Robô alinhado com a bola, 15–40 cm atrás
            rz = float(bz) - self._rng.uniform(0.15, 0.40)
            rx = float(bx) + self._rng.uniform(-0.15, 0.15)

        elif cs < 100_000:
            # Fase 1b — fácil: bola de 0.5 a 1.5 m do gol, robô atrás
            bz = self._rng.uniform(_gz - 1.50, _gz - 0.50)   # 3.05 → 4.05
            bx = self._rng.uniform(-_ghw, _ghw)               # dentro das traves
            rz_max = float(bz) - 0.20
            rz_min = max(float(bz) - 1.20, 0.0)
            rz = self._rng.uniform(rz_min, rz_max)
            rx = float(bx) + self._rng.uniform(-0.30, 0.30)

        elif cs < 300_000:
            # Fase 2 — médio: bola no campo de ataque (z>0)
            bz = self._rng.uniform(0.0, FIELD["half_length"] * 0.80)
            bx = self._rng.uniform(-FIELD["half_width"] * 0.60,
                                    FIELD["half_width"] * 0.60)
            rz_max = max(float(bz) - 0.20, -0.20)
            rz = self._rng.uniform(-FIELD["half_length"] * 0.85, rz_max)
            rx = self._rng.uniform(-FIELD["half_width"] * 0.70,
                                    FIELD["half_width"] * 0.70)

        else:
            # Fase 3 — aleatório completo
            bx = self._rng.uniform(-FIELD["half_width"]  * 0.60,
                                    FIELD["half_width"]  * 0.60)
            bz = self._rng.uniform(-FIELD["half_length"] * 0.40,
                                    FIELD["half_length"] * 0.40)
            rx = self._rng.uniform(-FIELD["half_width"]  * 0.70,
                                    FIELD["half_width"]  * 0.70)
            rz = self._rng.uniform(-FIELD["half_length"] * 0.85, -0.5)

        self._ball_node.getField("translation").setSFVec3f(
            [float(bx), BALL["radius"], float(bz)]
        )
        self._ball_node.setVelocity([0, 0, 0, 0, 0, 0])
        self._ball_node.resetPhysics()

        # ── Randomise robot (own half: z < 0) ─────────────────────────────
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

        self._prev_dist_ball      = float(obs[1]) * FIELD_DIAG
        self._prev_dist_ball_goal = math.hypot(
            ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"]
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
        self._prev_robot_pos  = robot_pos
        self._prev_ball_z     = ball_pos[1]
        self._curriculum_step += 1

        info = {**events, "step": self._step_count}
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

        # Rotaciona direção do frame mundo → frame local do robô.
        # Ry(-θ): local_x =  cos θ·dx + sin θ·dz
        #         local_z = -sin θ·dx + cos θ·dz
        # Com isso: se a bola está à frente, dir_bz > 0 → vz > 0 (direto).
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

        obs = np.array(
            [
                float(cfg["type_id"]),       # [0]  0=viper / 1=titan
                dist_ball / FIELD_DIAG,      # [1]  dist normalizada robô→bola
                dir_ball[0],                 # [2]  dir_x LOCAL
                dir_ball[1],                 # [3]  dir_z LOCAL  (>0 = bola à frente)
                dist_goal / FIELD_DIAG,      # [4]  dist normalizada robô→gol
                dir_goal[0],                 # [5]  dir_x LOCAL
                dir_goal[1],                 # [6]  dir_z LOCAL  (>0 = gol à frente)
            ] + post_feats,                  # [7-18] 4 postes × (dist, dir_x, dir_z) LOCAL
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
        events: dict,
    ) -> float:
        """ 
        1. SCORE GOAL                       | (dominant objective)
        2. MOVE BALL TOWARD GOAL            | (main shaping)
        3. GET INTO GOOD STRIKING POSITION  |
        4. REACH BALL                       |
        5. AVOID USELESS / STALLED MOTION   |
        """
        
        # --------------- #
        # TERMINAL EVENTS #
        # --------------- #

        if events["goal_scored"]:
            return +300.0
        if events["own_goal"]:
            return -201.0
        if events["ball_out"]:
            return -117.0

        # --------- #
        # Over Time #
        # --------- #
        
        # --- calcs --- #
        TOUCH_THRESHOLD = 0.16
        GOAL_Z = FIELD["goal_z_attack"]
        bx, bz = ball_pos
        rx, rz = robot_pos
        reward = 0.0



        # --- time penalty --- #
        reward -= 0.002
        # --- position penalties/rewards --- #
        if (
            not self._midfield_bonus_given
            and self._prev_ball_z < 0.0
            and bz >= 0.0
        ):

            self._midfield_bonus_given = True
            reward += 1.0



        # --- ball progress toward robot --- #
        ball_progress = self._prev_dist_ball - dist_ball
        reward += ball_progress * 2.0



        # --- ball progress toward goal --- #
        dist_ball_goal = math.hypot(
            bx,
            bz - GOAL_Z
        )
        goal_progress = (
            self._prev_dist_ball_goal
            - dist_ball_goal
        )
        reward += goal_progress * 12.0


        # --- ball velocity towords goal [-, +]based direction --- #
        try:
            vel = self._ball_node.getVelocity()

            ball_vx = vel[0]
            ball_vz = vel[2]

            to_goal_x = -bx
            to_goal_z = GOAL_Z - bz

            norm = math.hypot(to_goal_x, to_goal_z)

            if norm > 1e-6:
                to_goal_x /= norm
                to_goal_z /= norm
                # Dot product
                toward_goal_speed = (
                    ball_vx * to_goal_x
                    + ball_vz * to_goal_z
                )
                # Positive if ball moving toward goal
                reward += np.clip(
                    toward_goal_speed * 0.8,
                    -1.0,
                    1.0,
                )
        except Exception:
            pass

        # --- ball between robot and goal --- #
        robot_behind_ball = rz < bz
        if robot_behind_ball:
            reward += 0.05
        else:
            reward -= 0.05

        # --- if alighned with ball and goal --- #
        rb_x = bx - rx
        rb_z = bz - rz

        rb_norm = math.hypot(rb_x, rb_z)

        if rb_norm > 1e-6:

            rb_x /= rb_norm
            rb_z /= rb_norm

            bg_x = -bx
            bg_z = GOAL_Z - bz

            bg_norm = math.hypot(bg_x, bg_z)

            if bg_norm > 1e-6:

                bg_x /= bg_norm
                bg_z /= bg_norm

                alignment = (
                    rb_x * bg_x
                    + rb_z * bg_z
                )
                # [-1, 1]
                reward += alignment * 0.3


        # --- contact bonus & cooldown --- #
        if dist_ball < TOUCH_THRESHOLD:
            if (
                self._step_count
                - getattr(self, "_last_touch_step", -999)
            ) > 8:

                reward += 0.25
                self._last_touch_step = self._step_count


        # --- stillness penalty - ball reached so nothing else to do - --- #
        try:
            vel = self._ball_node.getVelocity()
            ball_speed = math.hypot(
                vel[0],
                vel[2]
            )
            if (
                dist_ball < 0.25
                and ball_speed < 0.03
            ):
                reward -= 0.05
        except Exception:
            pass


        # --- stillness penalty - robot stuck --- #
        try:
            vel = self._ball_node.getVelocity()

            ball_speed = math.hypot(
                vel[0],
                vel[2]
            )

            if (
                dist_ball < 0.25
                and ball_speed < 0.03
            ):
                reward -= 0.05
        except Exception:
            pass

        
        return reward


    def _compute_reward(
        self,
        dist_ball : float,
        ball_pos  : tuple,
        robot_pos : tuple,
        events    : dict,
    ) -> float:

        # 1. Terminal events — dominant signal, early return
        if events["goal_scored"]:
            return 50.0     # aumentado de 10 para 50 — gol deve dominar o sinal
        if events["own_goal"]:
            return -30.0    # aumentado de -8 para -30
        if events["ball_out"]:
            return -2.0     # ball left the field without scoring

        reward = 0.0

        # 2. Progress of robot toward ball
        reward += (self._prev_dist_ball - dist_ball) * 2.0

        # 3. Proximidade contínua à bola — sinal denso para a rede de valor aprender.
        # Decai linearmente: +0.03 quando tocando a bola, → 0 na outra ponta do campo.
        # Máximo acumulado: 1500 × 0.03 = +45/episódio < gol (+50) → hierarquia correta.
        # Substitui o bônus de contato binário (+0.3 fixo) que causava hover.
        reward += 0.03 * (1.0 - min(dist_ball, FIELD_DIAG) / FIELD_DIAG)

        # 4. Progress of ball toward attack goal — escalado pela proximidade do robô.
        # O fator de proximidade força a sequência: ir até a bola → empurrar pro gol.
        # Sem essa escala, o robô pode ganhar reward de "bola avançando" sem estar perto,
        # e aprende a ficar parado esperando a bola rolar sozinha.
        #
        # fator = 1.0 quando tocando a bola, 0.0 quando dist ≥ _GOAL_PROX_SCALE.
        _GOAL_PROX_SCALE = 2.0   # metros — raio de influência do robô sobre a bola
        prox_factor = max(0.0, 1.0 - dist_ball / _GOAL_PROX_SCALE)
        dist_ball_to_goal = math.hypot(
            ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"]
        )
        # Clipa para nunca ser negativo: empurrar a bola para o lado é neutro (0),
        # empurrar para o gol é bom (+). Sem o clip, empurrar na direção errada
        # dava reward negativo, e o robô aprendia a NÃO tocar na bola.
        goal_progress = self._prev_dist_ball_goal - dist_ball_to_goal
        reward += max(0.0, goal_progress) * 10.0 * prox_factor

        # 4. Bônus de meio campo — +0.5 uma vez por episódio quando a bola
        # cruza de z<0 (campo defensivo) para z>0 (campo de ataque).
        # Cria um degrau intermediário: aproximar → cruzar → marcar.
        if (not self._midfield_bonus_given
                and self._prev_ball_z < 0.0
                and ball_pos[1] >= 0.0):
            self._midfield_bonus_given = True
            reward += 0.5

        # 5. Ball velocity component directed toward goal
        reward += _ball_toward_goal_bonus(
            self._ball_node, ball_pos, FIELD["goal_z_attack"]
        )

        # (bônus de alinhamento removido — sinal muito fraco, só adicionava ruído)

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

        # 7. Penalidade por falta de progresso em 10 segundos
        # Guarda posição atual no histórico e compara com a posição de 250 passos atrás.
        # Se o deslocamento líquido for menor que 30cm, o robô está preso/oscilando.
        self._pos_history.append(robot_pos)
        if len(self._pos_history) == self._NO_PROGRESS_WINDOW:
            old_pos = self._pos_history[0]
            net_displacement = math.hypot(
                robot_pos[0] - old_pos[0],
                robot_pos[1] - old_pos[1],
            )
            if net_displacement < self._NO_PROGRESS_THRESH:
                reward += self._NO_PROGRESS_PENALTY

        # 8. Per-step time penalty
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
