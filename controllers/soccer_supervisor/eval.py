# ------------------------------- #
# EVALUATE/COMPARE TRAINED MODELS #
# ------------------------------- #

''' 
Metrics reported
 -> Goal rate (goals scored / total episodes)
 -> Own-goal rate
 -> Ball-out rate
 -> Mean / std reward per episode
 -> Mean / std final distance of ball to attack goal (metres)
 -> Mean / std episode length (steps)
 -> Per-episode breakdown table
'''

# ------- #
# IMPORTS #
# ------- #

from __future__ import annotations
 
import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from soccer_supervisor import SoccerEnv
from shared_configs import FIELD


#  Allow running as both a Webots controller and a plain Python script 
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))   # for shared_configs
 
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
MODEL_PATH = os.path.join(_CKPT_DIR, "final_model.zip")
N_EPISODES = 10
DETERMINISTIC = True # False = stochastic actions (more variety in eval)



# --------- #
# FUNCTIONS #
# --------- #

def _find_vecnorm(model_path: str) -> str | None:
    """
    Given  .../checkpoints/epoch_05_viper.zip
    return .../checkpoints/epoch_05_viper_vecnorm.pkl  if it exists.
    """
    stem = os.path.splitext(model_path)[0]
    pkl  = stem + "_vecnorm.pkl"
    return pkl if os.path.isfile(pkl) else None


# --- EVAL --- #

def model_evaluate(
        model_path: str, 
        n_episodes: int, 
        deterministic: bool
    ) -> list[dict]:

    # returns list of dicts, one per episode, with restlts
    # prints global avg and std of metrics --- performance of a singular model

    GOAL_Z = FIELD["goal_z_attack"]

    # load model
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}  |  Deterministic: {deterministic}")
    print(f"{'─'*51}\n")
 
    #model = PPO.load(model_path, env=env)    # still deciding wher to load model

    env_raw  = SoccerEnv()
    # Run at real-time speed so the 3D view is smooth (not a flickering fast-sim).
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_REAL_TIME)
    env_mon  = Monitor(env_raw)
    vec_env  = DummyVecEnv([lambda: env_mon])

    vecnorm_path = _find_vecnorm(model_path)
    if vecnorm_path:
        print(f" VecNorm: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False   # freeze running stats during eval
        vec_env.norm_reward = False   # return raw rewards for reporting
    else:
        print(" VecNorm: not found — rewards will be raw (unnormalised)")
        vec_env = VecNormalize(
            vec_env,
            norm_obs = False,
            norm_reward = False,
            gamma = 0.99,
        )

    model = PPO.load(model_path, env=vec_env)
    # ----------§---------- #

    results = []

    for ep in range(n_episodes):
        #model = PPO.load(model_path, env=vec_env) # load model inside loop to reset any VecNormalize stats if used during training

        obs, _  = vec_env.reset() # if model load inside, comment this line
        ep_reward = 0.0
        ep_steps = 0
        done = False
        last_info = {}

        # run sim and get in episode metrics
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            last_info = infos[0]
            done = bool(dones[0])


        # final ball dist to goal (read directly from the Webots node)
        ball_p = env_raw._ball_node.getPosition()
        ball_x = ball_p[0]
        ball_z = ball_p[2]
        dist_to_goal= math.hypot(ball_x, ball_z - GOAL_Z)


        # --- metrics in episode --- #
        results.append(
            dict(
                ep = ep + 1,
                reward = ep_reward,
                steps = ep_steps,
                goal_scored = last_info.get("goal_scored", False),
                own_goal = last_info.get("own_goal", False),
                ball_out = last_info.get("ball_out", False),
                dist_to_goal = dist_to_goal,
                ball_x = ball_x,
                ball_z = ball_z,
            )
        )

        print(
            f"Episode {ep + 1:>2} | "
            f"R={ep_reward:+8.2f}  steps={ep_steps:>4}"
            f"dist_to_goal={dist_to_goal:.3f} m\n"
            f"{'─'*51}\n"
        )

    vec_env.close()


    # ------------------------------ #
    # --- avg of episode metrics --- #
    # ------------------------------ #

    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    dists = np.array([r["dist_to_goal"] for r in results])
    n_goals = sum(r["goal_scored"] for r in results)
    n_own_goals = sum(r["own_goal"] for r in results)
    n_ball_out = sum(r["ball_out"] for r in results)

    print(f"Avg reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Avg steps: {steps.mean():.1f} ± {steps.std():.1f}")
    print(f"Avg final dist to goal: {dists.mean():.3f} ± {dists.std():.3f} m")
    print(f"Goal rate: {n_goals}/{n_episodes} = {n_goals/n_episodes:.1%}")
    print(f"Own-goal rate: {n_own_goals}/{n_episodes} = {n_own_goals/n_episodes:.1%}")
    print(f"Ball-out rate: {n_ball_out}/{n_episodes} = {n_ball_out/n_episodes:.1%}")


    return results

def model_compare(
        list_of_model_paths: list[str], 
        n_episodes: int, 
        deterministic: bool,
        cmap: str = "rainbow"
    ) -> None:

    # plots bar charts comparing multiple models across the above metrics
    # 1 subplot per metric

    # --- get evals of all models --- #
    summaries = []

    for model_path in list_of_model_paths:

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*70}\n")

        results = model_evaluate(
            model_path=model_path,
            n_episodes=n_episodes,
            deterministic=deterministic
        )

        # avg of model performance
        rewards = np.array([r["reward"] for r in results])
        steps = np.array([r["steps"] for r in results])
        dists = np.array([r["dist_to_goal"] for r in results])
        n_goals = sum(r["goal_scored"] for r in results)
        n_own_goals = sum(r["own_goal"] for r in results)
        n_ball_out = sum(r["ball_out"] for r in results)

        summaries.append(
            {
                "name": Path(model_path).stem,

                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),

                "steps_mean": steps.mean(),
                "steps_std": steps.std(),

                "dist_mean": dists.mean(),
                "dist_std": dists.std(),

                "goal_rate": n_goals/n_episodes,
                "own_goal_rate": n_own_goals/n_episodes,
                "ball_out_rate": n_ball_out/n_episodes,
            }
        )


    # --- plots --- #
    model_names = [s["name"] for s in summaries]
    metrics = [
        ("reward_mean", "Mean Reward"),
        ("goal_rate", "Goal Rate"),
        ("own_goal_rate", "Own Goal Rate"),
        ("ball_out_rate", "Ball Out Rate"),
        ("dist_mean", "Final Distance To Goal (m)"),
        ("steps_mean", "Episode Length (steps)"),
    ]

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(model_names)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):

        values = [s[key] for s in summaries]

        ax.bar(
            model_names,
            values,
            color=colors,
            edgecolor="black"
        )

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # rotate labels if long
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Model Comparison", fontsize=16)
    plt.tight_layout()

    plt.show()



# --- CLI --- #
def _cli() -> None:
    """Parse CLI args and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SoccerEnv model.")
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to .zip model file (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EPISODES})"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (default: deterministic)"
    )
    args = parser.parse_args()
    model_evaluate(args.model, args.episodes, not args.stochastic)





# ------------ #
# --- MAIN --- #
# ------------ #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        _cli()
    else:
        model_evaluate(MODEL_PATH, N_EPISODES, DETERMINISTIC)
