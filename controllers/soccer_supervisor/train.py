"""
Training script for the 1×0 soccer RL agent.

Called by soccer_supervisor.py (the Webots controller) so it never imports
that module directly — avoiding circular imports.  The raw SoccerEnv instance
is created in soccer_supervisor.py and passed here as a parameter.

Epoch loop (per robot)
──────────────────────
  Each robot (Viper, Titan) is trained independently with its own PPO model
  and VecNormalize statistics, avoiding catastrophic forgetting from alternation.

  Viper  → N_EPOCHS epochs → final_model_viper.zip
  Titan  → N_EPOCHS epochs → final_model_titan.zip

Outputs (relative to this file's directory)
────────────────────────────────────────────
  checkpoints/epoch_NN_<robot>.zip          — periodic snapshots
  checkpoints/final_model_<robot>.zip       — final model per robot
  logs/                                     — TensorBoard event files
  plots/training_curves_<robot>.png         — reward + goal-rate per robot
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_EPOCHS        = 13              # epochs per robot (13 Viper + 13 Titan = 26 total)
STEPS_PER_EPOCH = 120_000         # steps per epoch — 120k gives each robot time to stabilise
ROBOT_SEQUENCE  = ["viper", "titan"]

# [CRITICO-2 fix] Rede simétrica — value function aprende melhor
POLICY_KWARGS: dict = dict(
    net_arch = [256, 256],
)

# [CRITICO-3 fix] Learning rate com decaimento linear 3e-4 → 1e-5
def _lr_schedule(progress_remaining: float) -> float:
    """Linear decay: 3e-4 no início, 1e-5 no fim."""
    return max(1e-5, 3e-4 * progress_remaining)

PPO_KWARGS: dict = dict(
    policy        = "MlpPolicy",
    n_steps       = 1000,         # [CRITICO-1 fix] alinhado com _max_steps=1000 do env
    batch_size    = 250,          # [MEDIO-2 fix] era 64; 250 = n_steps/4
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.03,         # [MEDIO-3 fix] era 0.01; 0.03 evita entropy collapse
    learning_rate = _lr_schedule,
    verbose       = 1,
)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
_LOG_DIR  = os.path.join(_HERE, "logs")
_PLOT_DIR = os.path.join(_HERE, "plots")


# ── Main entry point ──────────────────────────────────────────────────────────

def train(env_raw) -> None:
    """
    Train independent PPO models for Viper and Titan sequentially.
    Each robot gets its own model and VecNormalize statistics — no cross-robot
    interference / catastrophic forgetting.
    """
    for d in (_CKPT_DIR, _LOG_DIR, _PLOT_DIR):
        os.makedirs(d, exist_ok=True)

    # Train Viper first (already in world at startup)
    print("\n" + "═" * 60)
    print("  TRAINING: VIPER")
    print("═" * 60)
    env_raw.set_robot_type("viper")
    env_raw._curriculum_step = 0   # reset curriculum for each robot
    _train_robot(env_raw, "viper")

    # Swap to Titan and train
    print("\n" + "═" * 60)
    print("  TRAINING: TITAN")
    print("═" * 60)
    env_raw.swap_robot("titan")
    env_raw._curriculum_step = 0   # reset curriculum for Titan
    _train_robot(env_raw, "titan")

    print("\n[train] All robots trained. Models saved to checkpoints/")


def _train_robot(env_raw, robot_name: str) -> PPO:
    """
    Train one robot for N_EPOCHS epochs with its own fresh model and VecNormalize.
    Saves checkpoints and a final model for this robot.
    """
    ckpt_dir = _CKPT_DIR
    log_dir  = _LOG_DIR

    # ── Wrapper stack (fresh for each robot) ─────────────────────────────────
    env     = Monitor(env_raw)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,
        norm_reward = True,
        clip_reward = 50.0,    # [MEDIO fix] era 10.0; 50 preserva sinal do gol (+300)
        gamma       = PPO_KWARGS["gamma"],
    )

    model = PPO(
        env             = vec_env,
        policy_kwargs   = POLICY_KWARGS,
        tensorboard_log = log_dir,
        **PPO_KWARGS,
    )

    epoch_rewards:    list[float] = []
    epoch_goal_rates: list[float] = []

    for epoch in range(N_EPOCHS):
        print(f"\n[{robot_name}] Epoch {epoch + 1}/{N_EPOCHS}  "
              f"steps_so_far={model.num_timesteps}")

        stats_cb = _StatsCallback()
        try:
            model.learn(
                total_timesteps     = STEPS_PER_EPOCH,
                reset_num_timesteps = False,
                callback            = stats_cb,
                tb_log_name         = f"ppo_{robot_name}",
                progress_bar        = True,
            )
        finally:
            # Crash-safe mid-epoch vecnorm snapshot
            vec_env.save(os.path.join(ckpt_dir,
                f"epoch_{epoch:02d}_{robot_name}_vecnorm.pkl"))

        stem = os.path.join(ckpt_dir, f"epoch_{epoch:02d}_{robot_name}")
        model.save(stem)
        vec_env.save(stem + "_vecnorm.pkl")

        if stats_cb.ep_rewards:
            mean_r  = float(np.mean(stats_cb.ep_rewards))
            goal_rt = float(np.mean(stats_cb.ep_goals))
        else:
            mean_r, goal_rt = 0.0, 0.0

        epoch_rewards.append(mean_r)
        epoch_goal_rates.append(goal_rt)
        print(f"  episodes={len(stats_cb.ep_rewards)}"
              f"  mean_reward={mean_r:.3f}"
              f"  goal_rate={goal_rt:.1%}")

    # ── Save final model for this robot ──────────────────────────────────────
    final_stem = os.path.join(ckpt_dir, f"final_model_{robot_name}")
    model.save(final_stem)
    vec_env.save(final_stem + "_vecnorm.pkl")
    print(f"\n[{robot_name}] Done. Final model → {final_stem}.zip")

    _plot_curves(epoch_rewards, epoch_goal_rates, robot_name)
    return model


# ── Episode-stats callback ────────────────────────────────────────────────────

class _StatsCallback(BaseCallback):
    """Collects per-episode reward and goal-rate within a single learn() call."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.ep_rewards: list[float] = []
        self.ep_goals:   list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(ep["r"])
                self.ep_goals.append(1.0 if info.get("goal_scored") else 0.0)
        return True


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_curves(
    rewards:    list[float],
    goal_rates: list[float],
    robot_name: str,
) -> None:
    if not rewards:
        return

    color  = "#2196F3" if robot_name == "viper" else "#F44336"
    epochs = list(range(1, len(rewards) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Training curves — {robot_name.capitalize()}",
                 fontweight="bold")

    for i, r in enumerate(rewards):
        ax1.bar(epochs[i], r, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Mean episode reward")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.3)

    for i, g in enumerate(goal_rates):
        ax2.bar(epochs[i], g * 100.0, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Goal rate (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_PLOT_DIR, f"training_curves_{robot_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[{robot_name}] Plot saved → {out}")
