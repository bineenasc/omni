"""
Monitor em tempo real do treino do TITAN.
Rode com:  py -3.13 monitor_titan.py
Atualiza automaticamente a cada 30 segundos.

Lê logs/episode_log.csv (escrito por train.py), filtrando robot == "titan".
Dependências: só `csv` (padrão) + `matplotlib` — sem tensorboard/pandas/numpy.
"""

import os
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(_HERE, "controllers", "soccer_supervisor", "logs", "episode_log.csv")
ROBOT     = "titan"
COR       = "#F44336"
WINDOW    = 50          # janela da média móvel (episódios)


def _rolling(xs, ys, w):
    out, acc, s = [], [], 0.0
    for x, y in zip(xs, ys):
        acc.append(y); s += y
        if len(acc) > w:
            s -= acc.pop(0)
        out.append((x, s / len(acc)))
    return out


def ler_csv():
    """Lê episode_log.csv filtrando por robô. Retorna dict com séries."""
    if not os.path.isfile(CSV_PATH):
        return {}
    ts, rew, length, goal = [], [], [], []
    try:
        # Lê em binário e remove bytes NUL (o arquivo às vezes fica com padding
        # NUL no fim, o que faz o csv.reader estourar "line contains NUL").
        with open(CSV_PATH, "rb") as f:
            raw = f.read()
        text = raw.replace(b"\x00", b"").decode("utf-8", "replace")
        for row in csv.DictReader(text.splitlines()):
            if (row.get("robot") or "").strip() != ROBOT:
                continue
            try:
                ts.append(float(row["timestep"]))
                rew.append(float(row["reward"]))
                length.append(float(row["length"]))
                goal.append(int(float(row["goal"])))
            except (KeyError, ValueError, TypeError):
                continue
    except Exception as e:
        print("Falha lendo CSV:", e)
        return {}
    if not ts:
        return {}
    recent = goal[-WINDOW:]
    return {
        "ep_rew_mean": _rolling(ts, rew, WINDOW),
        "ep_len_mean": _rolling(ts, length, WINDOW),
        "_goal_rate":  sum(recent) / len(recent),
        "_steps":      ts[-1],
    }


def status(gr):
    if gr is None:        return "Sem dados ainda"
    if gr >= 0.60:        return f"Marcando gols! ({gr:.0%})"
    if gr >= 0.30:        return f"Aprendendo a marcar ({gr:.0%})"
    if gr >= 0.05:        return f"Gols ocasionais ({gr:.0%})"
    return f"Quase nao marca ({gr:.0%}) — normal no inicio"


plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor="#1a1a2e")
fig.suptitle("Monitor Titan — Robo de Futebol", fontsize=15, color=COR, fontweight="bold")
timer_text = fig.text(0.99, 0.01, "", ha="right", color="#666", fontsize=9)
status_box = fig.text(0.5, 0.01, "", ha="center", color="white", fontsize=11,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e", edgecolor=COR))
_contador = [30]


def desenhar(dados):
    ax1.clear(); ax2.clear()
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#333")

    if dados.get("ep_rew_mean"):
        pts = dados["ep_rew_mean"]; xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax1.plot(xs, ys, color=COR, linewidth=2.2)
        ax1.fill_between(xs, ys, alpha=0.15, color=COR)
        ax1.annotate(f" {ys[-1]:.1f}", xy=(xs[-1], ys[-1]), color=COR,
                     fontsize=10, va="center", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Aguardando dados do treino do Titan...",
                 ha="center", va="center", transform=ax1.transAxes, color="#666", fontsize=12)

    if dados.get("ep_len_mean"):
        pts = dados["ep_len_mean"]; xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax2.plot(xs, ys, color=COR, linewidth=2.2)
        ax2.fill_between(xs, ys, alpha=0.15, color=COR)

    ax1.set_ylabel(f"Recompensa/episodio\n(media movel {WINDOW})", color="#ccc", fontsize=10)
    ax1.axhline(0, color="#444", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.15, color="white")
    ax2.set_ylabel("Duracao do episodio", color="#ccc", fontsize=10)
    ax2.set_xlabel("Passos de treino (timesteps)", color="#ccc", fontsize=10)
    ax2.grid(axis="y", alpha=0.15, color="white")

    gr = dados.get("_goal_rate")
    tot = dados.get("_steps", 0)
    status_box.set_text(f"Titan | {tot:,.0f} passos | {status(gr)}")
    plt.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.canvas.draw_idle()


def atualizar(frame):
    _contador[0] -= 1
    if _contador[0] <= 0:
        _contador[0] = 30
        timer_text.set_text("Atualizando...")
        fig.canvas.draw_idle()
        desenhar(ler_csv())
        timer_text.set_text("Proxima atualizacao em 30s")
    else:
        timer_text.set_text(f"Proxima atualizacao em {_contador[0]}s")
        fig.canvas.draw_idle()


print("Carregando dados do Titan...")
desenhar(ler_csv())
timer_text.set_text("Proxima atualizacao em 30s")
anim = FuncAnimation(fig, atualizar, interval=1000, cache_frame_data=False)
plt.show()
