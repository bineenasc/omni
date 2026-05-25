"""
Training script for the 1×0 soccer RL agent.

Called by soccer_supervisor.py (the Webots controller) so it never imports
that module directly — avoiding circular imports.  The raw SoccerEnv instance
is created in soccer_supervisor.py and passed here as a parameter.

Epoch loop
──────────
  Epoch 0  →  Viper
  Epoch 1  →  Titan
  Epoch 2  →  Viper
  …

The same PPO model (and CNN weights) is used throughout.  The robot type_id
in the observation vector lets the network distinguish the two robots.
Each epoch the physical robot node is hot-swapped via env.swap_robot().

Outputs (relative to this file's directory)
────────────────────────────────────────────
  checkpoints/epoch_NN_<robot>.zip — periodic model snapshots
  checkpoints/final_model.zip      — model after all epochs
  logs/                            — TensorBoard event files
  plots/training_curves.png        — reward + goal-rate per epoch
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# lidar_cnn.py lives alongside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_cnn import LidarCNNExtractor

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_EPOCHS        = 20
STEPS_PER_EPOCH = 30_000          # Webots env steps (not Webots basic timesteps)
ROBOT_SEQUENCE  = ["viper", "titan"]

POLICY_KWARGS: dict = dict(
    features_extractor_class  = LidarCNNExtractor,
    features_extractor_kwargs = dict(lidar_features=32, vector_features=32),
    net_arch = [64, 64],          # policy + value MLP layers after the extractor
)

PPO_KWARGS: dict = dict(
    policy        = "MultiInputPolicy",
    n_steps       = 1024,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    learning_rate = 3e-4,
    verbose       = 1,
)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
_LOG_DIR  = os.path.join(_HERE, "logs")
_PLOT_DIR = os.path.join(_HERE, "plots")


# ── Main training function ────────────────────────────────────────────────────

def train(env_raw) -> PPO:
    """
    Run the full Viper/Titan alternating training loop.

    Parameters
    ----------
    env_raw : SoccerEnv
        The unwrapped environment instance.  Kept separate from the Monitor
        wrapper so that swap_robot() / set_robot_type() remain accessible.

    Returns the final PPO model.
    """
    for d in (_CKPT_DIR, _LOG_DIR, _PLOT_DIR):
        os.makedirs(d, exist_ok=True)

    # Stack wrappers: Monitor → DummyVecEnv → VecNormalize
    #   • Monitor      : records per-episode (r, l) for stats callbacks
    #   • DummyVecEnv  : required by SB3 for VecNormalize
    #   • VecNormalize : normalises rewards using a running mean/std
    #                    (norm_obs=False because observations are already
    #                     manually scaled to [-1,1] / [0,1] in _get_obs /
    #                     _drain_receiver, so double-normalising would hurt)
    env     = Monitor(env_raw)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,   # obs already normalised by the environment
        norm_reward = True,    # normalise reward with running mean/std
        clip_reward = 10.0,    # clip after normalisation
        gamma       = PPO_KWARGS["gamma"],
    )

    model = PPO(
        env             = vec_env,
        policy_kwargs   = POLICY_KWARGS,
        tensorboard_log = _LOG_DIR,
        **PPO_KWARGS,
    )

    epoch_rewards:    list[float] = []
    epoch_goal_rates: list[float] = []

    # Viper is already in the world (placed in soccer.wbt).
    # Skip the swap at epoch 0 to avoid a remove/re-insert of the same proto.
    active_robot: str = ROBOT_SEQUENCE[0]   # "viper"
    env_raw.set_robot_type(active_robot)

    for epoch in range(N_EPOCHS):
        robot_name = ROBOT_SEQUENCE[epoch % len(ROBOT_SEQUENCE)]

        # Hot-swap the physical robot only when the type actually changes
        if robot_name != active_robot:
            print(f"\n[train] Swapping physical robot → {robot_name} …")
            env_raw.swap_robot(robot_name)
            active_robot = robot_name

        print(f"\n[train] Epoch {epoch + 1}/{N_EPOCHS}  robot={robot_name}"
              f"  total_steps_so_far={model.num_timesteps}")

        stats_cb = _StatsCallback()
        model.learn(
            total_timesteps     = STEPS_PER_EPOCH,
            reset_num_timesteps = False,       # accumulate global step counter
            callback            = stats_cb,
            tb_log_name         = f"ppo_{robot_name}",
            progress_bar        = True,
        )

        ckpt_stem = os.path.join(_CKPT_DIR, f"epoch_{epoch:02d}_{robot_name}")
        model.save(ckpt_stem)
        vec_env.save(ckpt_stem + "_vecnorm.pkl")

        if stats_cb.ep_rewards:
            mean_r  = float(np.mean(stats_cb.ep_rewards))
            goal_rt = float(np.mean(stats_cb.ep_goals))
            epoch_rewards.append(mean_r)
            epoch_goal_rates.append(goal_rt)
            print(
                f"  episodes={len(stats_cb.ep_rewards)}"
                f"  mean_reward={mean_r:.3f}"
                f"  goal_rate={goal_rt:.1%}"
            )
        else:
            epoch_rewards.append(0.0)
            epoch_goal_rates.append(0.0)

    model.save(os.path.join(_CKPT_DIR, "final_model"))
    vec_env.save(os.path.join(_CKPT_DIR, "final_model_vecnorm.pkl"))
    print("\n[train] Training complete.  Final model saved.")

    _plot_curves(epoch_rewards, epoch_goal_rates, N_EPOCHS, ROBOT_SEQUENCE)
    return model


# ── Episode-stats callback ────────────────────────────────────────────────────

class _StatsCallback(BaseCallback):
    """
    Collects per-episode reward and goal-rate within a single learn() call.

    SB3's Monitor wrapper adds info["episode"] = {"r": …, "l": …} to the info
    dict of the terminal step.  Our SoccerEnv also sets info["goal_scored"]
    on that same step, so both keys are readable here.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.ep_rewards: list[float] = []
        self.ep_goals:   list[float] = []   # 1.0 if episode ended with a goal

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:                             # terminal step
                self.ep_rewards.append(ep["r"])
                self.ep_goals.append(1.0 if info.get("goal_scored") else 0.0)
        return True


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_curves(
    rewards:    list[float],
    goal_rates: list[float],
    n_epochs:   int,
    robot_seq:  list[str],
) -> None:
    """Save bar-chart training curves to plots/training_curves.png."""
    if not rewards:
        return

    epochs = list(range(1, len(rewards) + 1))
    colors = [
        "#2196F3" if robot_seq[i % len(robot_seq)] == "viper" else "#F44336"
        for i in range(len(rewards))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i, (r, c) in enumerate(zip(rewards, colors)):
        ax1.bar(epochs[i], r, color=c, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Mean episode reward")
    ax1.set_title("Training curves  —  blue = Viper  /  red = Titan")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.3)

    for i, (g, c) in enumerate(zip(goal_rates, colors)):
        ax2.bar(epochs[i], g * 100.0, color=c, alpha=0.85, edgecolor="white",
                linewidth=0.5)
    ax2.set_ylabel("Goal rate (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(_PLOT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[train] Learning curves saved → {out_path}")
