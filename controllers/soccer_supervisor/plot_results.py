"""
plot_results.py
-------------------------------------------------------------------------------
Curva de aprendizado POR EPISÓDIO (com média móvel) a partir do CSV escrito por
train.py (logs/episode_log.csv). NÃO precisa fechar o Webots — só lê o arquivo.

Use o MESMO Python dos monitores (que tem matplotlib):

    py -3.13 plot_results.py
    py -3.13 plot_results.py --window 100
    py -3.13 plot_results.py --no-show --out fig.png

Dependências: apenas `csv` (padrão) + `matplotlib` — sem pandas/numpy.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, "logs", "episode_log.csv")
_DEFAULT_OUT = os.path.join(_HERE, "plots", "learning_curve.png")


def _rolling(values, window):
    """Média móvel simples (janela deslizante), sem numpy."""
    out, acc = [], []
    s = 0.0
    for v in values:
        acc.append(v)
        s += v
        if len(acc) > window:
            s -= acc.pop(0)
        out.append(s / len(acc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Curva de aprendizado por episódio.")
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--window", type=int, default=50, help="janela da média móvel")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--no-show", action="store_true", help="só salva, não abre janela")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV não encontrado: {args.csv}\nRode um treino primeiro.")

    ts, rew, goal = [], [], []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["timestep"]))
                rew.append(float(row["reward"]))
                goal.append(int(float(row["goal"])))
            except (KeyError, ValueError):
                continue
    if not rew:
        raise SystemExit("CSV vazio — nenhum episódio ainda.")

    w = args.window
    ep = list(range(1, len(rew) + 1))
    rew_ma  = _rolling(rew, w)
    goal_ma = [100.0 * g for g in _rolling(goal, w)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) recompensa por episódio + média móvel
    ax = axes[0]
    ax.plot(ep, rew, alpha=0.20, color="tab:blue", label="recompensa bruta")
    ax.plot(ep, rew_ma, color="tab:blue", linewidth=2, label=f"média móvel ({w})")
    ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa acumulada")
    ax.set_title("Curva de aprendizado"); ax.legend(); ax.grid(alpha=0.3)

    # (b) recompensa (média móvel) vs timesteps totais
    ax = axes[1]
    ax.plot(ts, rew_ma, color="tab:green", linewidth=2)
    ax.set_xlabel("Timesteps totais"); ax.set_ylabel(f"Recompensa (média móvel {w})")
    ax.set_title("Eficiência amostral"); ax.grid(alpha=0.3)

    # (c) taxa de gols (média móvel)
    ax = axes[2]
    ax.plot(ep, goal_ma, color="tab:orange", linewidth=2)
    ax.set_xlabel("Episódio"); ax.set_ylabel("Taxa de gols (%)")
    ax.set_ylim(0, 100); ax.set_title("Taxa de gols (média móvel)"); ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Gráfico salvo em {args.out}")

    print("\n=== Resumo ===")
    print(f"Episódios:           {len(rew)}")
    print(f"Recompensa média:    {sum(rew)/len(rew):.2f}")
    print(f"Melhor episódio:     {max(rew):.2f}")
    tail = rew[-w:]
    print(f"Últimos {len(tail)} ep.:     {sum(tail)/len(tail):.2f}")
    print(f"Taxa de gols global: {sum(goal)/len(goal):.1%}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
