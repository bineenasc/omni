"""
Monitor em tempo real do treino do VIPER.
Rode com: py -3.13 monitor_viper.py
Atualiza automaticamente a cada 30 segundos.
"""

import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "controllers", "soccer_supervisor", "logs")
RUN_NAME  = "ppo_viper_0"
COR       = "#2196F3"
INTERVALO = 30_000  # ms


def limpar_logs_antigos():
    """Apaga todos os arquivos de log para começar o gráfico do zero."""
    run_dir = os.path.join(LOGS_DIR, RUN_NAME)
    arquivos = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    removidos = 0
    for arq in arquivos:
        try:
            os.remove(arq)
            removidos += 1
        except Exception:
            pass
    if removidos:
        print(f"🗑️  {removidos} log(s) removido(s) — gráfico começando do zero.")
    else:
        print("✅ Nenhum log encontrado — aguardando o Webots criar dados novos.")


def ler_logs():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    run_dir = os.path.join(LOGS_DIR, RUN_NAME)
    if not os.path.isdir(run_dir):
        return {}

    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    dados = {}
    for tag in tags:
        if "ep_rew_mean" in tag or "ep_len_mean" in tag:
            chave = "ep_rew_mean" if "ep_rew_mean" in tag else "ep_len_mean"
            dados[chave] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return dados


def status(ep_len):
    """Usa duração do episódio como indicador real de gols.
    _max_steps=1000 → duração colada em 1000 = sem gols.
    Duração a cair = episódios a terminar por gol."""
    if ep_len is None:    return "⚪ Sem dados ainda"
    if ep_len < 400:      return "🏆 Marcando gols frequentemente!"
    if ep_len < 700:      return "⚽ Marcando alguns gols"
    if ep_len < 950:      return "🟡 Gols raros — ainda a aprender"
    return "🔴 Sem gols — episódios a truncar por tempo"


plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor="#1a1a2e")
fig.suptitle("🔵 Monitor Viper — Robô de Futebol",
             fontsize=15, color=COR, fontweight="bold")

timer_text  = fig.text(0.99, 0.01, "", ha="right", color="#666", fontsize=9)
status_box  = fig.text(0.5,  0.01, "", ha="center", color="white", fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.4",
                                 facecolor="#16213e", edgecolor=COR))
_contador = [30]


def desenhar(dados):
    ax1.clear()
    ax2.clear()

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333")

    last_reward = None
    last_ep_len = None
    total_steps = 0

    if "ep_rew_mean" in dados and dados["ep_rew_mean"]:
        pts   = dados["ep_rew_mean"]
        steps = [p[0] for p in pts]
        vals  = [p[1] for p in pts]
        ax1.plot(steps, vals, color=COR, linewidth=2.2)
        ax1.fill_between(steps, vals, alpha=0.15, color=COR)
        last_reward = vals[-1]
        total_steps = steps[-1]
        ax1.annotate(f" {last_reward:.2f}", xy=(steps[-1], last_reward),
                     color=COR, fontsize=10, va="center", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Aguardando dados do treino do Viper...",
                 ha="center", va="center", transform=ax1.transAxes,
                 color="#666", fontsize=12)

    if "ep_len_mean" in dados and dados["ep_len_mean"]:
        pts   = dados["ep_len_mean"]
        steps = [p[0] for p in pts]
        vals  = [p[1] for p in pts]
        ax2.plot(steps, vals, color=COR, linewidth=2.2)
        ax2.fill_between(steps, vals, alpha=0.15, color=COR)
        last_ep_len = vals[-1]
        ax2.annotate(f" {last_ep_len:.0f} steps", xy=(steps[-1], last_ep_len),
                     color=COR, fontsize=10, va="center", fontweight="bold")

    ax1.set_ylabel("Recompensa por episódio\n(shaping acumulado — NÃO indica gols)", color="#ccc", fontsize=10)
    ax1.axhline(0, color="#444", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.15, color="white")

    ax2.set_ylabel("Duração dos episódios ⚽\n(CAIR = marcando gols reais)", color="#ccc", fontsize=10)
    ax2.set_xlabel("Passos de treino →", color="#ccc", fontsize=10)
    ax2.axhline(1000, color="#F44336", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.annotate("← max_steps (sem gol)", xy=(0, 1000), color="#F44336",
                 fontsize=8, xytext=(5, -12), textcoords="offset points")
    ax2.set_ylim(0, 1100)
    ax2.grid(axis="y", alpha=0.15, color="white")

    st = f"Viper | {total_steps:,} passos | {status(last_ep_len)}"
    status_box.set_text(st)

    plt.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.canvas.draw_idle()


def atualizar(frame):
    _contador[0] -= 1
    if _contador[0] <= 0:
        _contador[0] = 30
        timer_text.set_text("Atualizando...")
        fig.canvas.draw_idle()
        desenhar(ler_logs())
        timer_text.set_text("Próxima atualização em 30s")
    else:
        timer_text.set_text(f"Próxima atualização em {_contador[0]}s")
        fig.canvas.draw_idle()


limpar_logs_antigos()
print("Carregando dados do Viper...")
desenhar(ler_logs())
timer_text.set_text("Próxima atualização em 30s")

anim = FuncAnimation(fig, atualizar, interval=1000, cache_frame_data=False)
plt.show()
