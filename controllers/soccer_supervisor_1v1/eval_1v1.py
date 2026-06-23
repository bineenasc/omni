"""
1v1 Evaluation script  —  called by soccer_supervisor_1v1.py when MODE == "eval".

Behaviour
─────────
  • Switches Webots to REAL-TIME mode so you can watch the match visually.
  • Loads the final trained models (final_model_viper.zip / final_model_titan.zip).
    If a model is missing, that robot falls back to the scripted policy.
  • Runs N_EVAL_EPISODES episodes in phase3 (fully random spawns, both models active).
  • Prints per-episode results and a summary table at the end.

Outputs
───────
  checkpoints/eval_results.csv   — one row per episode
  (printed to stdout)            — summary: goal rate, win rate, draw rate, avg length
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared_configs import FIELD, SIM

# ── Configuration ─────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 20
_HERE           = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR       = os.path.join(_HERE, "checkpoints")
_RESULTS_CSV    = os.path.join(_CKPT_DIR, "eval_results.csv")

_VIPER_MODEL_PATH = os.path.join(_CKPT_DIR, "final_model_viper.zip")
_TITAN_MODEL_PATH = os.path.join(_CKPT_DIR, "final_model_titan.zip")


def eval_1v1(env_raw) -> None:
    """Run N_EVAL_EPISODES evaluation episodes and print/save statistics."""
    from stable_baselines3 import PPO

    # ── Switch to real-time so the match is visible in Webots ─────────────────
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_REAL_TIME)
    print("[eval_1v1] Real-time mode ON — you can watch the match in Webots.")

    # ── Load models ───────────────────────────────────────────────────────────
    viper_model = _try_load(PPO, _VIPER_MODEL_PATH, "Viper")
    titan_model = _try_load(PPO, _TITAN_MODEL_PATH, "Titan")

    # ── Set up environment for evaluation ─────────────────────────────────────
    env_raw.set_phase("phase3")
    env_raw.set_opp_policy(titan_model)   # Titan is the frozen opponent for Viper

    # ── Result accumulators ───────────────────────────────────────────────────
    os.makedirs(_CKPT_DIR, exist_ok=True)
    rows: list[dict] = []

    viper_wins = 0
    titan_wins = 0
    draws      = 0
    ep_lengths: list[int] = []

    print(f"\n{'─'*60}")
    print(f"  1v1 Evaluation — {N_EVAL_EPISODES} episodes")
    print(f"{'─'*60}")
    print(f"  {'Ep':>3}  {'Result':^12}  {'Steps':>6}  {'Scorer'}")
    print(f"{'─'*60}")

    for ep in range(1, N_EVAL_EPISODES + 1):
        # ── Alternate active robot every episode to evaluate both sides ────────
        if ep % 2 == 1:
            env_raw.set_active_robot("viper")
            env_raw.set_opp_policy(titan_model)
            agent_name = "viper"
        else:
            env_raw.set_active_robot("titan")
            env_raw.set_opp_policy(viper_model)
            agent_name = "titan"

        obs, _ = env_raw.reset()
        done   = False
        steps  = 0
        result = "draw"
        scorer = "—"

        while not done:
            # Choose action for the active robot
            if agent_name == "viper" and viper_model is not None:
                action, _ = viper_model.predict(obs, deterministic=True)
            elif agent_name == "titan" and titan_model is not None:
                action, _ = titan_model.predict(obs, deterministic=True)
            else:
                action = env_raw.action_space.sample()

            obs, _reward, terminated, truncated, info = env_raw.step(action)
            done   = terminated or truncated
            steps += 1

            if info.get("agent_goal"):
                result = "agent_goal"
                scorer = agent_name
            elif info.get("opp_goal"):
                result = "opp_goal"
                scorer = "titan" if agent_name == "viper" else "viper"

        # ── Tally results from Viper's perspective ─────────────────────────────
        if result == "agent_goal" and agent_name == "viper":
            viper_wins += 1
        elif result == "opp_goal" and agent_name == "viper":
            titan_wins += 1
        elif result == "agent_goal" and agent_name == "titan":
            titan_wins += 1
        elif result == "opp_goal" and agent_name == "titan":
            viper_wins += 1
        else:
            draws += 1

        ep_lengths.append(steps)
        rows.append({
            "episode":    ep,
            "agent":      agent_name,
            "result":     result,
            "scorer":     scorer,
            "steps":      steps,
        })

        label = {"agent_goal": "GOAL ✓", "opp_goal": "OPP GOAL", "draw": "DRAW"}.get(result, result)
        print(f"  {ep:>3}  {label:^12}  {steps:>6}  {scorer}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = viper_wins + titan_wins + draws
    print(f"\n{'═'*60}")
    print(f"  RESULTS  ({total} episodes)")
    print(f"{'═'*60}")
    print(f"  Viper  wins : {viper_wins:>3}  ({viper_wins/total*100:5.1f}%)")
    print(f"  Titan  wins : {titan_wins:>3}  ({titan_wins/total*100:5.1f}%)")
    print(f"  Draws       : {draws:>3}  ({draws/total*100:5.1f}%)")
    print(f"  Avg length  : {np.mean(ep_lengths):.1f} steps  "
          f"({np.mean(ep_lengths)*SIM['basic_time_step']*SIM['steps_per_action']/1000:.1f} s)")
    print(f"{'═'*60}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "agent", "result", "scorer", "steps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval_1v1] Results saved → {_RESULTS_CSV}")


def _try_load(PPO, path: str, name: str):
    """Load a PPO model, returning None (with a warning) if not found."""
    if os.path.isfile(path) or os.path.isfile(path + ".zip"):
        actual = path if path.endswith(".zip") else path + ".zip"
        if not os.path.isfile(actual):
            actual = path
        try:
            model = PPO.load(actual)
            print(f"[eval_1v1] {name} model loaded from {actual}")
            return model
        except Exception as exc:
            print(f"[eval_1v1] WARNING: could not load {name} model ({exc}). "
                  "Using random actions.")
            return None
    print(f"[eval_1v1] WARNING: {name} model not found at {path}. "
          "Using random actions.")
    return None
