"""
1v1 Training loop  —  called by soccer_supervisor_1v1.py.

Training flow
─────────────

  WARMUP  (60 k steps)
  ┌─ Only Viper learns, Titan is static.
  │  Loads Viper's 1v0 model and extends it from 24-D to 30-D obs via
  │  weight surgery (new opponent features start with zero weights).

  PHASE 1  (adaptive: advances when Viper goal_rate ≥ 60 %)
  ┌─ Viper only, Titan static in its own half.
  │  Goal: Viper re-masters ball approach/shooting with opponent present.

  PHASE 2  (adaptive: alternating, advances when both ≥ 60 %)
  ┌─ Alternating fine-tune.
  │  Viper's turn : Titan uses scripted rule (go-to-ball → shoot at −Z).
  │  Titan's turn : Viper uses its latest frozen checkpoint.
  │  Titan loads its own 1v0 weights (extended to 30-D) when it starts.

  PHASE 3  (N_EPOCHS_PHASE3 epochs)
  ┌─ Alternating fine-tune with frozen snapshots updated once per epoch.
  │  Each epoch:
  │    1. Viper learns STEPS_PER_HALF_EPOCH steps vs frozen Titan snapshot.
  │    2. Save Titan snapshot for next Viper turn.
  │    3. Titan learns STEPS_PER_HALF_EPOCH steps vs frozen Viper snapshot.
  │    4. Save Viper snapshot for next Titan turn.

Outputs  (relative to this file)
─────────────────────────────────
  checkpoints/warmup_viper.zip
  checkpoints/phase1_final_viper.zip
  checkpoints/phase2_final_{viper,titan}.zip
  checkpoints/phase3_epoch_{NN}_{viper,titan}.zip
  checkpoints/final_model_{viper,titan}.zip
  checkpoints/final_model_{viper,titan}_vecnorm.pkl
  plots/training_curves_1v1_{viper,titan}.png
  logs/  (TensorBoard)
"""

from __future__ import annotations

import copy
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Directories ───────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR    = os.path.join(_HERE, "checkpoints")
_LOG_DIR     = os.path.join(_HERE, "logs")
_PLOT_DIR    = os.path.join(_HERE, "plots")
EPISODE_CSV  = os.path.join(_LOG_DIR, "episode_log_1v1.csv")

# Path to 1v0 final models (same project, soccer_supervisor folder)
_1V0_DIR  = os.path.join(_HERE, "..", "soccer_supervisor", "checkpoints")


# ── CSV episode logger ────────────────────────────────────────────────────────
# Columns: episode, timestep, reward, length, goal, opp_goal, phase, robot

def _init_episode_csv_1v1(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["episode", "timestep", "reward", "length",
             "goal", "opp_goal", "phase", "robot"]
        )


class _EpisodeCSVCallback1v1(BaseCallback):
    """Appends one CSV row at the end of each episode during 1v1 training."""

    def __init__(self, csv_path: str, phase: str, robot: str) -> None:
        super().__init__(verbose=0)
        self.csv_path = csv_path
        self.phase    = phase
        self.robot    = robot
        self._ep      = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep += 1
                goal     = 1 if info.get("goal_scored")  else 0
                opp_goal = 1 if info.get("opp_goal")      else 0
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self._ep, self.num_timesteps, f"{ep['r']:.4f}",
                         ep["l"], goal, opp_goal, self.phase, self.robot]
                    )
        return True

# ── Hyperparameters ───────────────────────────────────────────────────────────
WARMUP_STEPS          = 60_000
STEPS_PER_EPOCH       = 120_000
STEPS_PER_HALF_EPOCH  = STEPS_PER_EPOCH // 2    # 60k per robot per alternation

N_EPOCHS_PHASE1_MAX   = 10    # max epochs in phase 1 (adaptive may cut short)
N_EPOCHS_PHASE2_MAX   = 10    # max epochs per robot in phase 2
N_EPOCHS_PHASE3       = 8     # fixed epochs in phase 3

GOAL_RATE_THRESH = 0.60   # phase-advance threshold
GOAL_RATE_WINDOW = 10     # episodes window for goal-rate calculation

POLICY_KWARGS: dict = dict(net_arch=[256, 256])

def _lr_schedule(progress_remaining: float) -> float:
    return max(1e-5, 3e-4 * progress_remaining)

PPO_KWARGS: dict = dict(
    policy        = "MlpPolicy",
    n_steps       = 1000,
    batch_size    = 250,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.03,
    learning_rate = _lr_schedule,
    verbose       = 1,
)

OBS_DIM_1V0 = 24
OBS_DIM_1V1 = 30


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def train_1v1(env_raw) -> None:
    """Full 1v1 training pipeline.  ``env_raw`` is a live SoccerEnv1v1 instance."""
    for d in (_CKPT_DIR, _LOG_DIR, _PLOT_DIR):
        os.makedirs(d, exist_ok=True)
    _init_episode_csv_1v1(EPISODE_CSV)

    # ── Build VecEnv wrappers (shared; active robot is switched in env_raw) ───
    # Two independent wrapper stacks keep VecNormalize reward statistics separate.
    viper_vec, titan_vec = _make_vec_envs(env_raw)

    # ── Load and extend 1v0 weights to 30-D ───────────────────────────────────
    viper_model = _load_and_extend(
        path_1v0  = os.path.join(_1V0_DIR, "final_model_viper.zip"),
        vec_env   = viper_vec,
        robot_tag = "viper",
    )
    titan_model = _load_and_extend(
        path_1v0  = os.path.join(_1V0_DIR, "final_model_titan.zip"),
        vec_env   = titan_vec,
        robot_tag = "titan",
    )

    rewards_log: dict[str, list[float]] = {"viper": [], "titan": []}
    goals_log:   dict[str, list[float]] = {"viper": [], "titan": []}

    # ═══════════════════════════════════════════════════════════════════════════
    # WARMUP — Viper only, Titan static, 1v0-style curriculum
    # ═══════════════════════════════════════════════════════════════════════════
    print(_header("WARMUP  (Viper only — adapting to 30-D obs)"))
    env_raw.set_phase("warmup")
    env_raw.set_active_robot("viper")

    stats = _learn(viper_model, viper_vec, WARMUP_STEPS, "ppo_viper_warmup",
                   phase="warmup", robot="viper")
    _append_stats(stats, rewards_log["viper"], goals_log["viper"])
    viper_model.save(os.path.join(_CKPT_DIR, "warmup_viper"))
    print(f"[Warmup] Done.  goal_rate={np.mean(stats.ep_goals):.1%}" if stats.ep_goals else "[Warmup] Done.")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Viper only, Titan static, adaptive curriculum
    # ═══════════════════════════════════════════════════════════════════════════
    print(_header("PHASE 1  (Viper vs static Titan)"))
    env_raw.set_phase("phase1")
    env_raw.set_active_robot("viper")

    for epoch in range(N_EPOCHS_PHASE1_MAX):
        print(f"\n[Phase1 | Viper] Epoch {epoch+1}/{N_EPOCHS_PHASE1_MAX}")
        stats = _learn(viper_model, viper_vec, STEPS_PER_EPOCH, "ppo_viper_p1",
                       phase="phase1", robot="viper")
        _append_stats(stats, rewards_log["viper"], goals_log["viper"])
        viper_model.save(os.path.join(_CKPT_DIR, f"phase1_epoch{epoch:02d}_viper"))
        gr = float(np.mean(stats.ep_goals)) if stats.ep_goals else 0.0
        print(f"  goal_rate={gr:.1%}")

        if _curriculum_ready(env_raw):
            print(f"[Phase1] Viper goal_rate threshold met — advancing to Phase 2.")
            break

    viper_model.save(os.path.join(_CKPT_DIR, "phase1_final_viper"))

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Alternating fine-tune, scripted/frozen opponent
    # ═══════════════════════════════════════════════════════════════════════════
    print(_header("PHASE 2  (Alternating — scripted/frozen opponent)"))
    env_raw.set_phase("phase2")

    viper_phase2_done = False
    titan_phase2_done = False

    for epoch in range(N_EPOCHS_PHASE2_MAX):
        print(f"\n[Phase2] Epoch {epoch+1}/{N_EPOCHS_PHASE2_MAX}")

        # ── Viper's turn (Titan scripted) ─────────────────────────────────────
        if not viper_phase2_done:
            env_raw.set_active_robot("viper")
            env_raw.set_opp_policy(None)    # scripted opponent
            stats = _learn(viper_model, viper_vec, STEPS_PER_HALF_EPOCH, "ppo_viper_p2",
                           phase="phase2", robot="viper")
            _append_stats(stats, rewards_log["viper"], goals_log["viper"])
            gr = float(np.mean(stats.ep_goals)) if stats.ep_goals else 0.0
            print(f"  [Viper] goal_rate={gr:.1%}")
            if _curriculum_ready(env_raw):
                print("  [Viper] Phase 2 threshold met.")
                viper_phase2_done = True

        # ── Titan's turn (Viper frozen checkpoint) ────────────────────────────
        if not titan_phase2_done:
            env_raw.set_active_robot("titan")
            viper_snap = _snapshot_policy(viper_model)
            env_raw.set_opp_policy(viper_snap)
            stats = _learn(titan_model, titan_vec, STEPS_PER_HALF_EPOCH, "ppo_titan_p2",
                           phase="phase2", robot="titan")
            _append_stats(stats, rewards_log["titan"], goals_log["titan"])
            gr = float(np.mean(stats.ep_goals)) if stats.ep_goals else 0.0
            print(f"  [Titan] goal_rate={gr:.1%}")
            if _curriculum_ready(env_raw):
                print("  [Titan] Phase 2 threshold met.")
                titan_phase2_done = True

        if viper_phase2_done and titan_phase2_done:
            print("[Phase2] Both robots ready — advancing to Phase 3.")
            break

    viper_model.save(os.path.join(_CKPT_DIR, "phase2_final_viper"))
    titan_model.save(os.path.join(_CKPT_DIR, "phase2_final_titan"))

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Alternating with frozen snapshots (updated per epoch)
    # ═══════════════════════════════════════════════════════════════════════════
    print(_header("PHASE 3  (Self-play with periodic snapshot updates)"))
    env_raw.set_phase("phase3")
    env_raw.set_opp_policy(None)   # will be set inside loop

    # Initialise snapshots from the Phase 2 finals
    titan_snap = _snapshot_policy(titan_model)
    viper_snap = _snapshot_policy(viper_model)

    for epoch in range(N_EPOCHS_PHASE3):
        print(f"\n[Phase3] Epoch {epoch+1}/{N_EPOCHS_PHASE3}")

        # ── Viper's turn (vs frozen Titan snapshot from previous epoch) ────────
        env_raw.set_active_robot("viper")
        env_raw.set_opp_policy(titan_snap)
        stats = _learn(viper_model, viper_vec, STEPS_PER_HALF_EPOCH, "ppo_viper_p3",
                       phase="phase3", robot="viper")
        _append_stats(stats, rewards_log["viper"], goals_log["viper"])
        viper_model.save(os.path.join(_CKPT_DIR, f"phase3_epoch{epoch:02d}_viper"))
        viper_vec.save(os.path.join(_CKPT_DIR, f"phase3_epoch{epoch:02d}_viper_vecnorm.pkl"))
        print(f"  [Viper] goal_rate={float(np.mean(stats.ep_goals)):.1%}" if stats.ep_goals else "")

        # Update Titan snapshot BEFORE Titan's turn
        titan_snap = _snapshot_policy(titan_model)

        # ── Titan's turn (vs frozen Viper snapshot from previous epoch) ────────
        env_raw.set_active_robot("titan")
        env_raw.set_opp_policy(viper_snap)
        stats = _learn(titan_model, titan_vec, STEPS_PER_HALF_EPOCH, "ppo_titan_p3",
                       phase="phase3", robot="titan")
        _append_stats(stats, rewards_log["titan"], goals_log["titan"])
        titan_model.save(os.path.join(_CKPT_DIR, f"phase3_epoch{epoch:02d}_titan"))
        titan_vec.save(os.path.join(_CKPT_DIR, f"phase3_epoch{epoch:02d}_titan_vecnorm.pkl"))
        print(f"  [Titan] goal_rate={float(np.mean(stats.ep_goals)):.1%}" if stats.ep_goals else "")

        # Update Viper snapshot for the NEXT epoch
        viper_snap = _snapshot_policy(viper_model)

    # ── Save final models ─────────────────────────────────────────────────────
    for model, vec, tag in [
        (viper_model, viper_vec, "viper"),
        (titan_model, titan_vec, "titan"),
    ]:
        stem = os.path.join(_CKPT_DIR, f"final_model_{tag}")
        model.save(stem)
        vec.save(stem + "_vecnorm.pkl")
        print(f"[train_1v1] Final model → {stem}.zip")
        _plot_curves(rewards_log[tag], goals_log[tag], tag)

    print("\n[train_1v1] Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Weight surgery: extend 1v0 (24-D) policy to 1v1 (30-D)
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_extend(
    path_1v0:  str,
    vec_env,
    robot_tag: str,
) -> PPO:
    """
    Load a 1v0 PPO model and extend it to the 30-D observation space.

    The policy network's first linear layer is extended from (256, 24) to
    (256, 30).  Weights for the existing 24 inputs are copied unchanged;
    the 6 new opponent-feature columns are zero-initialised so the network
    initially ignores opponent information and starts from its 1v0 skill set.

    If the 1v0 model is not found, a fresh 30-D model is created instead.
    """
    # Fresh 30-D model to hold the extended weights
    new_model = PPO(
        env             = vec_env,
        policy_kwargs   = POLICY_KWARGS,
        tensorboard_log = _LOG_DIR,
        **PPO_KWARGS,
    )

    if not os.path.isfile(path_1v0 + ".zip") and not os.path.isfile(path_1v0):
        print(f"[train_1v1] WARNING: 1v0 model not found at {path_1v0}. "
              "Starting from random weights.")
        return new_model

    path = path_1v0 if path_1v0.endswith(".zip") else path_1v0
    if not path.endswith(".zip"):
        path = path + ".zip"

    print(f"[train_1v1] Loading 1v0 weights from {path}")
    old_model = PPO.load(path)   # loads without env — obs-space check skipped

    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()

    copied = 0
    extended = 0
    for key in new_sd:
        if key not in old_sd:
            continue
        old_p = old_sd[key]
        new_p = new_sd[key]

        if old_p.shape == new_p.shape:
            new_sd[key] = old_p.clone()
            copied += 1
        elif (old_p.dim() == 2
              and old_p.shape[1] == OBS_DIM_1V0
              and new_p.shape[1] == OBS_DIM_1V1):
            # First linear layer: [out_features, in_features]
            # Copy old 24-input weights, zero-init the 6 new columns.
            new_sd[key][:, :OBS_DIM_1V0].copy_(old_p)
            new_sd[key][:, OBS_DIM_1V0:].zero_()
            extended += 1

    new_model.policy.load_state_dict(new_sd)
    print(f"[train_1v1] {robot_tag}: {copied} layers copied, "
          f"{extended} input-layer(s) extended (24→30). "
          f"New opponent features initialised to 0.")
    return new_model


# ══════════════════════════════════════════════════════════════════════════════
# Frozen policy snapshot helper
# ══════════════════════════════════════════════════════════════════════════════

def _snapshot_policy(model: PPO) -> PPO:
    """Return a deep-frozen copy of the PPO policy for use as a static opponent.

    The copy is an independent PPO object with identical weights but no env
    attachment.  Its ``predict()`` method works normally.
    """
    snap = copy.deepcopy(model)
    # Disable gradient computation for the frozen copy
    for param in snap.policy.parameters():
        param.requires_grad_(False)
    return snap


# ══════════════════════════════════════════════════════════════════════════════
# Helper: check if env signals curriculum readiness
# ══════════════════════════════════════════════════════════════════════════════

def _curriculum_ready(env_raw) -> bool:
    """Return True if the env's adaptive curriculum buffer has hit the threshold."""
    outcomes = env_raw._curriculum_outcomes
    if len(outcomes) < env_raw._CURRICULUM_WINDOW:
        return False
    return sum(outcomes) / len(outcomes) >= env_raw._CURRICULUM_THRESH


# ══════════════════════════════════════════════════════════════════════════════
# VecEnv helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_vec_envs(env_raw):
    """Build two independent VecNormalize stacks sharing the same raw env.

    Because DummyVecEnv holds a reference (not a copy) to the underlying env,
    both stacks wrap the same SoccerEnv1v1 object.  We simply rebuild the
    Monitor/VecNormalize wrappers when switching active robot.
    """
    def _build(tag: str):
        mon     = Monitor(env_raw)
        vec     = DummyVecEnv([lambda: mon])
        vec_n   = VecNormalize(
            vec,
            norm_obs    = False,
            norm_reward = True,
            clip_reward = 50.0,
            gamma       = PPO_KWARGS["gamma"],
        )
        return vec_n

    return _build("viper"), _build("titan")


# ══════════════════════════════════════════════════════════════════════════════
# Single model.learn() wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _learn(
    model: PPO,
    vec_env,
    n_steps: int,
    tb_name: str,
    phase: str = "",
    robot: str = "",
) -> "_StatsCallback":
    stats_cb = _StatsCallback()
    csv_cb   = _EpisodeCSVCallback1v1(EPISODE_CSV, phase=phase, robot=robot)
    try:
        model.learn(
            total_timesteps     = n_steps,
            reset_num_timesteps = False,
            callback            = CallbackList([stats_cb, csv_cb]),
            tb_log_name         = tb_name,
            progress_bar        = True,
        )
    except Exception as exc:
        print(f"[train_1v1] learn() raised: {exc}")
    return stats_cb


def _append_stats(
    stats:    "_StatsCallback",
    rewards:  list[float],
    goals:    list[float],
) -> None:
    if stats.ep_rewards:
        rewards.append(float(np.mean(stats.ep_rewards)))
        goals.append(float(np.mean(stats.ep_goals)))


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks and plotting
# ══════════════════════════════════════════════════════════════════════════════

class _StatsCallback(BaseCallback):
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


def _plot_curves(rewards: list[float], goals: list[float], tag: str) -> None:
    if not rewards:
        return
    color  = "#2196F3" if tag == "viper" else "#F44336"
    epochs = list(range(1, len(rewards) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"1v1 Training — {tag.capitalize()}", fontweight="bold")
    for i, r in enumerate(rewards):
        ax1.bar(epochs[i], r, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Mean episode reward")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.3)
    for i, g in enumerate(goals):
        ax2.bar(epochs[i], g * 100.0, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Goal rate (%)")
    ax2.set_xlabel("Learning step (cumulative)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_PLOT_DIR, f"training_curves_1v1_{tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[{tag}] Plot saved → {out}")


def _header(title: str) -> str:
    line = "═" * 60
    return f"\n{line}\n  {title}\n{line}"
