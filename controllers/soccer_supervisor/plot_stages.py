"""
plot_stages.py
-------------------------------------------------------------------------------
Gráfico de barras "Training curves" (recompensa média + taxa de gols POR STAGE
de self-play, azul=Viper / vermelho=Titan) a partir do CSV de episódios.

NÃO precisa fechar o Webots nem parar o treino — só LÊ logs/episode_log.csv.
Use o MESMO Python dos monitores (que tem matplotlib):

    py -3.13 plot_stages.py
    py -3.13 plot_stages.py --no-show          # só salva o PNG
    py -3.13 plot_stages.py --out fig.png

Dependências: apenas `csv` (padrão) + `matplotlib` — sem pandas/numpy.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, "logs", "episode_log.csv")
_DEFAULT_OUT = os.path.join(_HERE, "plots", "stage_bars.png")

VIPER_COLOR = "#2f6f9f"   # azul
TITAN_COLOR = "#9f3a32"   # vermelho

# Deve bater com _MAX_STEPS_BY_PHASE em soccer_supervisor.py (1 valor por fase).
MAX_STEPS_BY_PHASE = [350, 500, 650, 800, 950, 1100, 1200, 1250]


def _estimate_phase(lengths):
    """Estima a fase atual pelo maior comprimento recente (timeout = cap da fase)."""
    mx = max(lengths)
    phase = min(range(len(MAX_STEPS_BY_PHASE)),
                key=lambda i: abs(MAX_STEPS_BY_PHASE[i] - mx))
    return phase, mx


def main() -> None:
    ap = argparse.ArgumentParser(description="Bar chart de reward/goal-rate por stage.")
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--no-show", action="store_true", help="só salva, não abre janela")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV não encontrado: {args.csv}\nRode um treino primeiro.")

    # ── Lê o CSV (sem pandas) ───────────────────────────────────────────────
    rewards   = defaultdict(list)   # stage -> [reward, ...]
    goals     = defaultdict(list)   # stage -> [0/1, ...]
    robots    = defaultdict(Counter)  # stage -> Counter({robot: n})
    n_eps = 0
    last_t = 0
    per_robot = defaultdict(list)       # robot -> [0/1, ...] (gols, para o resumo)
    per_robot_len = defaultdict(list)   # robot -> [length, ...] (p/ estimar fase)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                st = int(float(row["stage"]))
                rw = float(row["reward"])
                gl = int(float(row["goal"]))
                rb = row["robot"]
                ln = float(row["length"])
                last_t = max(last_t, int(float(row["timestep"])))
            except (KeyError, ValueError):
                continue
            rewards[st].append(rw)
            goals[st].append(gl)
            robots[st][rb] += 1
            per_robot[rb].append(gl)
            per_robot_len[rb].append(ln)
            n_eps += 1

    if not n_eps:
        raise SystemExit("CSV vazio — nenhum episódio ainda.")

    stages = sorted(rewards.keys())
    mean_reward = [sum(rewards[s]) / len(rewards[s]) for s in stages]
    goal_rate   = [100.0 * sum(goals[s]) / len(goals[s]) for s in stages]
    colors = [VIPER_COLOR if robots[s].most_common(1)[0][0] == "viper"
              else TITAN_COLOR for s in stages]

    # ── Plota (matplotlib puro) ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    fig.suptitle(f"Training curves  (azul=Viper / vermelho=Titan)  —  "
                 f"{n_eps} episódios, ~{last_t:,} timesteps", fontsize=13)

    ax1.bar(stages, mean_reward, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Recompensa média\npor episódio")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(stages, goal_rate, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Taxa de gols (%)")
    ax2.set_xlabel("Stage de self-play (alterna Viper/Titan)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Gráfico salvo em {args.out}")

    # ── Resumo por robô ─────────────────────────────────────────────────────
    for robo in ("viper", "titan"):
        g = per_robot.get(robo, [])
        if g:
            last50 = g[-50:]
            lens50 = per_robot_len.get(robo, [])[-50:]
            phase, mx = _estimate_phase(lens50) if lens50 else (0, 0)
            print(f"  {robo}: {len(g)} ep | gol geral {sum(g)/len(g):.0%} "
                  f"| últimos {len(last50)} {sum(last50)/len(last50):.0%} "
                  f"| FASE ESTIMADA {phase}/{len(MAX_STEPS_BY_PHASE)-1} "
                  f"(max len recente {mx:.0f})")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
