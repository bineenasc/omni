# Omni Soccer RL 🤖⚽

Treinamento de robôs omnidirecionais para jogar futebol usando **Reinforcement Learning** (PPO) dentro do simulador **Webots R2025a**.

Dois robôs fisicamente diferentes — **Viper** e **Titan** — compartilham um único modelo de política e aprendem simultaneamente a partir de suas diferenças morfológicas.

---

## Visão Geral

```
                     channel 0 (ação)
  Supervisor ──────────────────────────► Robot Controller
             ◄────────────────────────── Robot Controller
                     channel 1 (lidar)
```

O **Supervisor** é onisciente para posições (lê GPS direto da API do Webots) e serve como ambiente Gymnasium. O **Robot Controller** aplica a cinemática inversa das rodas omnidirecionais e transmite os dados do LiDAR via IPC.

---

## Robôs

| Robô  | Raio do corpo | Velocidade máx. | Raio da roda |
|-------|--------------|-----------------|-------------|
| Viper | 12 cm        | 15 rad/s        | 3 cm        |
| Titan | 16.5 cm      | 9 rad/s         | 3.5 cm      |

Ambos possuem LiDAR de 1440 raios, GPS, IMU e 3 rodas omnidirecionais dispostas a 120°.

---

## Arquitetura de RL

| Componente        | Detalhe                                    |
|-------------------|--------------------------------------------|
| Algoritmo         | PPO (Stable Baselines3)                    |
| Política          | MlpPolicy — rede [256, 128]                |
| Espaço de ação    | Box(3,) → [vx, vz, ω] em frame local      |
| Espaço de obs     | Box(19,) — vetores em frame local do robô  |
| Normalização      | VecNormalize (reward only)                 |
| Steps por epoch   | 30 000                                     |
| Epochs            | 20 (alternando Viper/Titan)                |
| Duração máx.      | 1500 passos por episódio (60 s simulados)  |

### Observação (19 dimensões, frame local do robô)

```
[0]      type_id              — 0 = Viper, 1 = Titan
[1]      dist_ball_norm       — distância normalizada até a bola
[2–3]    dir_ball (x, z)      — direção LOCAL à bola
[4]      dist_goal_norm       — distância normalizada até o gol de ataque
[5–6]    dir_goal (x, z)      — direção LOCAL ao gol
[7–9]    poste direito ataque — dist, dir_x, dir_z (LOCAL)
[10–12]  poste esquerdo ataque
[13–15]  poste direito próprio
[16–18]  poste esquerdo próprio
```

> Todas as direções são rotacionadas para o frame local do robô usando `atan2(-m[1], m[0])` a partir da matriz de orientação do Webots. Isso permite que o mapeamento observação → ação seja direto: `dir_bz > 0` significa "bola à frente" → `vz > 0`.

---

## Sistema de Recompensas

| Sinal                        | Valor                                          |
|------------------------------|------------------------------------------------|
| Gol marcado                  | **+50** (encerra episódio)                     |
| Gol contra                   | −30 (encerra episódio)                         |
| Bola fora do campo           | −2 (encerra episódio)                          |
| Aproximação à bola           | (Δdist) × 2.0                                  |
| Proximidade contínua         | 0.03 × (1 − dist/diag) por passo              |
| Bola em direção ao gol       | max(0, Δdist\_gol) × 10 × fator\_proximidade  |
| Bola cruza o meio campo      | +0.5 (uma vez por episódio)                    |
| Velocidade da bola pro gol   | até +0.5 por passo                             |
| Robô parado > 30 passos      | −0.05 por passo                                |
| Sem progresso em 10 s        | −0.02 por passo                                |
| Penalidade de tempo          | −0.001 por passo                               |

O **fator de proximidade** escala o reward de avanço da bola pela distância robô-bola: `max(0, 1 − dist / 2.0)`. Isso força a sequência **ir até a bola → empurrar em direção ao gol**, evitando que o robô aprenda a ficar parado esperando a bola rolar sozinha.

---

## Curriculum de Posições

O treino usa um curriculum progressivo baseado no contador global de passos:

| Fase | Passos        | Epochs (60k/epoch)         | Spawn da bola                     | Spawn do robô          |
|------|---------------|----------------------------|-----------------------------------|------------------------|
| 1a   | 0 – 120k      | 0–1 (1× Viper + 1× Titan)  | 5–20 cm do gol, dentro das traves | 15–40 cm atrás da bola |
| 1b   | 120k – 240k   | 2–3 (1× Viper + 1× Titan)  | 50 cm – 1.5 m do gol              | Atrás da bola          |
| 2    | 240k – 360k   | 4–5 (1× Viper + 1× Titan)  | Campo de ataque (z > 0)           | Qualquer posição       |
| 3    | 360k+         | 6–25 (10× Viper + 10× Titan)| Completamente aleatório           | Completamente aleatório|

A fase 1a é crucial: com a bola a poucos centímetros do gol, qualquer toque marca ponto. Isso garante que o PPO veja o sinal de +50 nas primeiras horas de treino e não fique preso em ótimos locais.

---

## Estrutura do Projeto

```
omni/
├── controllers/
│   ├── shared_configs.py          # Constantes compartilhadas (campo, bola, IPC)
│   ├── robot_controller/
│   │   └── robot_controller.py   # Cinemática inversa + IPC (LiDAR)
│   └── soccer_supervisor/
│       ├── soccer_supervisor.py  # Ambiente Gymnasium + Supervisor Webots
│       ├── train.py              # Loop de treino PPO (SB3)
│       ├── checkpoints/          # Modelos salvos por epoch
│       ├── logs/                 # TensorBoard
│       └── plots/                # Curvas de treino
├── protos/
│   ├── Viper.proto               # Robô ágil — rodas menores, mais rápido
│   └── Titan.proto               # Robô robusto — corpo maior, mais lento
├── worlds/
│   └── soccer.wbt                # Mundo Webots (campo + gols + robôs)
├── monitor_viper.py              # Monitor em tempo real do Viper
├── monitor_titan.py              # Monitor em tempo real do Titan
└── requirements.txt
```

---

## Como Rodar

### Requisitos

- [Webots R2025a](https://cyberbotics.com)
- Python 3.11+

```bash
pip install -r requirements.txt
```

### Treino

1. Abra `worlds/soccer.wbt` no Webots
2. Clique em **Run** (`Ctrl+5`)
3. O treino inicia automaticamente — o supervisor é o controller da cena

Para monitorar em tempo real (janela separada):

```bash
python monitor_viper.py
python monitor_titan.py
```

Para acelerar o treino, desabilite a renderização 3D: **View → Rendering** (`Ctrl+4`).

### Outputs

| Arquivo                              | Conteúdo                              |
|--------------------------------------|---------------------------------------|
| `checkpoints/epoch_NN_<robot>.zip`   | Snapshot do modelo por epoch          |
| `checkpoints/final_model.zip`        | Modelo final após todos os epochs     |
| `checkpoints/*_vecnorm.pkl`          | Normalizador VecNormalize             |
| `logs/ppo_viper_0/` e `ppo_titan_0/` | Eventos TensorBoard                  |
| `plots/training_curves.png`          | Curva de reward e taxa de gols        |

---

## Decisões de Design

**Por que sem CNN?**
A versão anterior usava CNN 1D sobre 360 raios de LiDAR. O modelo nunca marcou um gol em 611k passos — o bônus de contato (+0.3/passo) dominava o sinal e o robô aprendeu a ficar parado perto da bola (*hover behavior*). A versão atual usa apenas features analíticas (GPS + direções calculadas), eliminando a CNN e tornando o aprendizado mais eficiente.

**Por que direções em frame local?**
As ações `[vx, vz, ω]` são em frame local do robô. Se as direções na observação fossem em frame mundo, a rede precisaria aprender implicitamente a rotação de coordenadas. Com direções já em frame local, `dir_bz > 0` significa diretamente "bola à frente" → `vz > 0`.

**Por que `max(0, Δprogresso)` no reward de avanço da bola?**
Sem o clamp, empurrar a bola na direção errada gera reward negativo, e o robô aprende racionalmente a não tocar na bola. Com `max(0, ...)`, qualquer contato é neutro ou positivo — o robô nunca é punido por tentar.

---

## Dependências

| Biblioteca        | Versão mínima |
|-------------------|---------------|
| gymnasium         | 0.29.0        |
| stable-baselines3 | 2.3.0         |
| torch             | 2.0.0         |
| numpy             | 1.24.0        |
| matplotlib        | 3.7.0         |
| Webots            | R2025a        |

---

## Licença

MIT — veja [LICENSE](LICENSE).
