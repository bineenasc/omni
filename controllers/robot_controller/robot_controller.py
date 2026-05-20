"""
Controlador Viper — 3 omniwheels, Webots R2025a, NUE (Y-up)

Cinemática inversa (body frame, frente = +X, lateral = +Z):

  [w1]   1  [  1       0      L ] [vx]
  [w2] = -- [ -sin30°  cos30° L ] [vz]
  [w3]   R  [ -sin30° -cos30° L ] [ω ]

  R = 0.030 m  (raio da roda)
  L = 0.138 m  (distância centro → roda)
"""

import math
from controller import Robot

R = 0.030   # raio da roda (m)
L = 0.138   # distância centro-roda (m)
S = math.sin(math.radians(30))   # 0.5
C = math.cos(math.radians(30))   # 0.866
MAX_W = 15.0

def ik(vx, vz, omega):
    """Retorna (w1, w2, w3) em rad/s"""
    return (
        ( 1.0*vx + 0.0*vz + L*omega) / R,
        (-S  *vx + C  *vz + L*omega) / R,
        (-S  *vx - C  *vz + L*omega) / R,
    )

def clamp(v): return max(-MAX_W, min(MAX_W, v))

# ── Inicialização ──────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

m1 = robot.getDevice("wheel1")
m2 = robot.getDevice("wheel2")
m3 = robot.getDevice("wheel3")

for m in (m1, m2, m3):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# ── Sequência de demo ──────────────────────────────────────────────
# (vx m/s, vz m/s, omega rad/s, duração ms)
DEMO = [
    (0.3,  0.0,  0.0, 2000),   # frente
    (0.0,  0.0,  0.0,  500),
    (-0.3, 0.0,  0.0, 2000),   # trás
    (0.0,  0.0,  0.0,  500),
    (0.0,  0.3,  0.0, 2000),   # lateral direita
    (0.0,  0.0,  0.0,  500),
    (0.0, -0.3,  0.0, 2000),   # lateral esquerda
    (0.0,  0.0,  0.0,  500),
    (0.0,  0.0,  1.5, 2000),   # rotação CCW
    (0.0,  0.0,  0.0,  500),
    (0.0,  0.0, -1.5, 2000),   # rotação CW
    (0.0,  0.0,  0.0, 1000),
]

idx = 0
t0  = None

while robot.step(timestep) != -1:
    now = robot.getTime() * 1000.0

    if idx < len(DEMO):
        vx, vz, omega, dur = DEMO[idx]
        if t0 is None:
            t0 = now
        if now - t0 >= dur:
            idx += 1
            t0 = now
            if idx < len(DEMO):
                vx, vz, omega, dur = DEMO[idx]
    else:
        vx, vz, omega = 0.0, 0.0, 0.0

    w1, w2, w3 = ik(vx, vz, omega)
    m1.setVelocity(clamp(w1))
    m2.setVelocity(clamp(w2))
    m3.setVelocity(clamp(w3))