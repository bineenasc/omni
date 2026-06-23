"""
Live training monitor for the 1v1 match.
Run with:  py -3.13 monitor_1v1.py

Reads  controllers/soccer_supervisor_1v1/logs/episode_log_1v1.csv
(written by train_1v1.py during training) and refreshes every 30 s.

Layout
──────
  Top row    : Viper — episode reward (left) | goal rate (right)
  Bottom row : Titan — episode reward (left) | goal rate (right)
  Footer bar : current phase, episode counts and live win/draw rates
"""

import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "controllers", "soccer_supervisor_1v1",
                        "logs", "episode_log_1v1.csv")

_COLORS = {"viper": "#2196F3", "titan": "#F44336"}
_INTERVAL_MS = 30_000   # refresh every 30 s


def _read_csv():
    """Parse episode_log_1v1.csv. Returns dict keyed by robot name."""
    data = defaultdict(lambda: {"rewards": [], "goals": [], "opp_goals": [],
                                 "lengths": [], "phases": []})
    if not os.path.isfile(CSV_PATH):
        return data
    try:
        with open(CSV_PATH, "rb") as f:
            raw = f.read()
        text = raw.replace(b"\x00", b"").decode("utf-8", "replace")
        for row in csv.DictReader(text.splitlines()):
            robot = (row.get("robot") or "").strip()
            if robot not in ("viper", "titan"):
                continue
            try:
                data[robot]["rewards"].append(float(row["reward"]))
                data[robot]["goals"].append(int(row["goal"]))
                data[robot]["opp_goals"].append(int(row.get("opp_goal", 0)))
                data[robot]["lengths"].append(int(row["length"]))
                data[robot]["phases"].append(str(row.get("phase", "")))
            except (ValueError, KeyError):
                pass
    except Exception as e:
        print("Failed reading CSV:", e)
    return data


def _smooth(values, window=10):
    """Simple moving average."""
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


def _status_label(goals, lengths, robot):
    if not goals:
        return f"{robot.capitalize()}: waiting for data..."
    rate = sum(goals[-20:]) / min(len(goals), 20)
    avg_len = sum(lengths[-20:]) / min(len(lengths), 20)
    if rate >= 0.70:
        icon = "🏆"
    elif rate >= 0.45:
        icon = "⚽"
    elif rate >= 0.20:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon}  {robot.capitalize()}  goal_rate={rate:.0%}  avg_len={avg_len:.0f}"


# ── Figure setup ──────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(14, 9), facecolor="#0d0d1a")
fig.suptitle("⚽  1v1 Live Training Monitor", fontsize=15,
             color="white", fontweight="bold")

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.88, bottom=0.12)
ax_vr = fig.add_subplot(gs[0, 0])   # Viper reward
ax_vg = fig.add_subplot(gs[0, 1])   # Viper goal rate
ax_tr = fig.add_subplot(gs[1, 0])   # Titan reward
ax_tg = fig.add_subplot(gs[1, 1])   # Titan goal rate

status_text = fig.text(0.5, 0.03, "", ha="center", va="bottom",
                       color="white", fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5",
                                 facecolor="#1a1a2e", edgecolor="#444"))
timer_text  = fig.text(0.99, 0.01, "", ha="right", color="#666", fontsize=9)
_counter    = [30]


def _style_ax(ax, title, ylabel, color):
    ax.set_facecolor("#111122")
    ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(axis="y", alpha=0.15, color="white")


def _draw(data):
    for ax in (ax_vr, ax_vg, ax_tr, ax_tg):
        ax.clear()

    _style_ax(ax_vr, "Viper — Episode Reward",    "reward",    _COLORS["viper"])
    _style_ax(ax_vg, "Viper — Goal Rate (20-ep)", "goal rate", _COLORS["viper"])
    _style_ax(ax_tr, "Titan — Episode Reward",    "reward",    _COLORS["titan"])
    _style_ax(ax_tg, "Titan — Goal Rate (20-ep)", "goal rate", _COLORS["titan"])

    status_lines = []

    for robot, (ax_r, ax_g) in [
        ("viper", (ax_vr, ax_vg)),
        ("titan", (ax_tr, ax_tg)),
    ]:
        c    = _COLORS[robot]
        d    = data[robot]
        eps  = list(range(1, len(d["rewards"]) + 1))

        if eps:
            smooth_r = _smooth(d["rewards"], window=15)
            ax_r.plot(eps, d["rewards"],  color=c, alpha=0.25, linewidth=0.8)
            ax_r.plot(eps, smooth_r,      color=c, linewidth=2.0,
                      label=f"MA-15  last={smooth_r[-1]:.1f}")
            ax_r.axhline(0, color="#555", linewidth=0.8, linestyle="--")
            ax_r.legend(fontsize=8, loc="upper left",
                        facecolor="#111122", edgecolor="#333")

            # goal rate rolling 20
            rates = [
                sum(d["goals"][max(0,i-19):i+1]) / min(i+1, 20)
                for i in range(len(d["goals"]))
            ]
            ax_g.plot(eps, [r * 100 for r in rates], color=c, linewidth=2.0)
            ax_g.fill_between(eps, [r * 100 for r in rates], alpha=0.15, color=c)
            ax_g.set_ylim(0, 100)
            ax_g.axhline(60, color="#888", linewidth=0.8, linestyle=":",
                         label="60 % threshold")
            ax_g.legend(fontsize=8, loc="upper left",
                        facecolor="#111122", edgecolor="#333")
        else:
            for ax in (ax_r, ax_g):
                ax.text(0.5, 0.5, f"Waiting for {robot} data…",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#555", fontsize=11)

        status_lines.append(_status_label(d["goals"], d["lengths"], robot))

    # Phase annotation
    all_phases = data["viper"]["phases"] + data["titan"]["phases"]
    last_phase = all_phases[-1] if all_phases else "—"
    total_v = len(data["viper"]["rewards"])
    total_t = len(data["titan"]["rewards"])
    status_text.set_text(
        f"Phase: {last_phase}   |   "
        f"Viper eps: {total_v}   Titan eps: {total_t}   |   "
        + "   ".join(status_lines)
    )

    plt.draw()


def _update(frame):
    _counter[0] -= 1
    if _counter[0] <= 0:
        _counter[0] = 30
        timer_text.set_text("Refreshing…")
        fig.canvas.draw_idle()
        _draw(_read_csv())
        timer_text.set_text("Next refresh in 30 s")
    else:
        timer_text.set_text(f"Next refresh in {_counter[0]} s")
        fig.canvas.draw_idle()


# ── Initial draw ──────────────────────────────────────────────────────────────
print(f"Reading: {CSV_PATH}")
if not os.path.isfile(CSV_PATH):
    print("  → File not found yet. Start training in Webots first.")
else:
    print("  → CSV found. Loading data…")

_draw(_read_csv())
timer_text.set_text("Next refresh in 30 s")
_anim = FuncAnimation(fig, _update, interval=1000, cache_frame_data=False)
plt.show()
