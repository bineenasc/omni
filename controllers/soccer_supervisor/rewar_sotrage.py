# just to store all tried - previous rewards for now
# will probably be deleted later --- ignore ---

import math
from controllers.soccer_supervisor.constants import FIELD

def _compute_reward_baseline(
        self,
        dist_ball: float,
        ball_pos: tuple,
        robot_pos: tuple,
        events: dict,
)-> float:
    # this is supose to be a baseline - simplest reward
    #deduct time, proximity to ball & score goal

    # --------------- #
    # TERMINAL EVENTS #
    # --------------- #

    if events["goal_scored"]:
        print("GOAL SCORED!")
        return +100.0
    if events["own_goal"]:
        return -100.0
    if events["ball_out"]:
        return -50.0
    
    # --------- #
    # Over Time #
    # --------- #
    reward = 0.0

    # --- time penalty --- #
    reward -= 0.01

    # --- dist to ball penalty --- #
    reward -= 0.05 * dist_ball

    return reward

def _compute_reward_s1( #s - simple 1 - frist adapt from bseline
        self,
        dist_ball: float,
        ball_pos: tuple,
        robot_pos: tuple,
        events: dict,
)-> float:
    # adds more ball relevance- ponts deducted by how far away it is from goal

    # --------------- #
    # TERMINAL EVENTS #
    # --------------- #

    if events["goal_scored"]:
        print("GOAL SCORED!")
        return +100.0
    if events["own_goal"]:
        return -100.0
    if events["ball_out"]:
        return -50.0
    
    # --------- #
    # Over Time #
    # --------- #
    reward = 0.0

    # --- time penalty --- #
    reward -= 0.01

    # --- dist to ball penalty --- #
    reward -= 0.05 * dist_ball

    # --- dist ball to goal penalty --- #
    dist_ball_goal = math.hypot(ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"])
    reward -= 0.05 * dist_ball_goal

    return reward



def _compute_reward_s2(
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
        print("GOAL SCORED!")
        return +100.0
    if events["own_goal"]:
        return -100.0
    if events["ball_out"]:
        return -50.0

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
    reward -= 0.01

    # --- dist to ball penalty --- #
    reward -= 0.05 * dist_ball

    # --- dist ball to goal penalty --- #
    dist_ball_goal = math.hypot(ball_pos[0], ball_pos[1] - FIELD["goal_z_attack"])
    reward -= 0.05 * dist_ball_goal


    # --- ball direction & speed towords goal area --- #
    # adds based on vector velocity of ball



    # --- ball between robot and goal --- #
    if rz>bz: reward -= 0.1

    
    
    # --- position penalties/rewards --- #
    '''if (bz <= 0.0): 
        reward -= 1.0'''

    # --- if alighned with ball and goal --- #
    '''rb_x, rb_z = bx - rx, bz - rz
    rb_norm    = math.hypot(rb_x, rb_z)

    if rb_norm > 1e-6:
        rb_x /= rb_norm
        rb_z /= rb_norm
        bg_x, bg_z = -bx, GOAL_Z - bz
        bg_norm    = math.hypot(bg_x, bg_z)
        if bg_norm > 1e-6:
            bg_x /= bg_norm
            bg_z /= bg_norm
            alignment = rb_x * bg_x + rb_z * bg_z
            reward   += alignment * 0.2
    '''

    # --- contact bonus & cooldown --- #
    if dist_ball < TOUCH_THRESHOLD:
        reward += 0.5
        '''if (self._step_count - getattr(self, "_last_touch_step", -999)) > 8:
            reward += 0.03
            self._last_touch_step = self._step_count'''



    # --- stillness penalty - ball reached so nothing else to do - --- #
    try:
        robot_vel   = self._robot_node.getVelocity()
        robot_speed = math.hypot(robot_vel[0], robot_vel[2])
        ball_vel    = self._ball_node.getVelocity()
        ball_speed  = math.hypot(ball_vel[0], ball_vel[2])
        if dist_ball < 0.25 and ball_speed < 0.03 and robot_speed < 0.03:
            reward -= 0.18   # robot AND ball both stationary near each other
    except Exception:
        pass
    
        
    return reward

def _compute_reward_s3(
    self,
    dist_ball: float,
    ball_pos: tuple,
    robot_pos: tuple,
    events: dict,
) -> float:

    # ---------------- #
    # TERMINAL REWARD  #
    # ---------------- #

    if events["goal_scored"]:
        print("GOAL SCORED!")
        return 150.0

    if events["own_goal"]:
        return -100.0

    if events["ball_out"]:
        return -50.0

    # ---------------- #
    # CONSTANTS        #
    # ---------------- #

    TOUCH_THRESHOLD = 0.16
    GOAL_Z = FIELD["goal_z_attack"]

    bx, bz = ball_pos
    rx, rz = robot_pos

    reward = 0.0

    # ---------------- #
    # TIME PENALTY     #
    # ---------------- #

    reward -= 0.01

    # ---------------- #
    # APPROACH BALL    #
    # ---------------- #

    # progress reward
    reward += (
        self._prev_dist_ball
        - dist_ball
    ) * 2.0

    # small shaping penalty
    reward -= 0.03 * dist_ball

    # ---------------- #
    # BALL -> GOAL     #
    # ---------------- #

    dist_ball_goal = math.hypot(
        bx,
        bz - GOAL_Z
    )

    ball_progress = (
        self._prev_dist_ball_goal
        - dist_ball_goal
    )

    reward += ball_progress * 8.0

    # ---------------- #
    # ROBOT BEHIND BALL#
    # ---------------- #

    # goal is +Z
    if rz > bz:
        reward -= 0.1

    # ---------------- #
    # BALL VELOCITY    #
    # ---------------- #

    try:
        ball_vel = self._ball_node.getVelocity()

        vx = ball_vel[0]
        vz = ball_vel[2]

        ball_speed = math.hypot(vx, vz)

        # unit vector ball -> goal
        to_goal_x = -bx
        to_goal_z = GOAL_Z - bz

        norm = math.hypot(
            to_goal_x,
            to_goal_z
        )

        if norm > 1e-6:

            to_goal_x /= norm
            to_goal_z /= norm

            toward_goal_speed = (
                vx * to_goal_x +
                vz * to_goal_z
            )

            # reward motion toward goal
            reward += max(
                0.0,
                toward_goal_speed
            ) * 1.5

            # kick bonus only when near ball
            if dist_ball < TOUCH_THRESHOLD:

                reward += max(
                    0.0,
                    toward_goal_speed
                ) * 0.75

    except Exception:
        pass

    # ---------------- #
    # ALIGNMENT BONUS  #
    # ---------------- #

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
                rb_x * bg_x +
                rb_z * bg_z
            )

            reward += alignment * 0.05

    # ---------------- #
    # TOUCH BONUS      #
    # ---------------- #

    if dist_ball < TOUCH_THRESHOLD:
        reward += 0.1

    # ---------------- #
    # STALL PENALTY    #
    # ---------------- #

    try:

        robot_vel = self._robot_node.getVelocity()
        robot_speed = math.hypot(
            robot_vel[0],
            robot_vel[2]
        )

        ball_vel = self._ball_node.getVelocity()
        ball_speed = math.hypot(
            ball_vel[0],
            ball_vel[2]
        )

        if (
            dist_ball < 0.25
            and robot_speed < 0.03
            and ball_speed < 0.03
        ):
            reward -= 0.2

    except Exception:
        pass

    return float(reward)


def _compute_reward_og(
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
