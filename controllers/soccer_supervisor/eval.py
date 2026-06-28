# ------------------------------- #
# EVALUATE/COMPARE TRAINED MODELS #
# ------------------------------- #

''' 
Metrics reported
For 1-0
    core
 -> Goal rate (goals scored / total episodes) (mean & std)
 -> Time to score (mean & std)
 // -> distance traveled (mean & std)
 -> ball possession time (dist smaller than x)(mean & std)
 -> number of ckiks (mean & std)

    navigation
 // -> path efficiency (min dist robot-ball+ball-goal)/distance traveled till goal

    control
 // -> internal vel and rotation vel
 -> Ball-out rate
 -> Mean / std reward per episode

For 1-1
 -> Goal score
 -> Enemy score

    attack
 -> Time to score
 -> ball possession time
 -> shot quality(vell&angle to goal of ball)

    defence
 // -> %ball on opoent side of field
    
    control
 -> Mean / std reward per episode
'''

# ------- #
# IMPORTS #
# ------- #

from __future__ import annotations
 
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from soccer_supervisor import SoccerEnv
from shared_configs import BALL, FIELD, IPC, ROBOT_CONFIGS, SIM, get_robot_config


#  Allow running as both a Webots controller and a plain Python script 
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))   # for shared_configs
 
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
MODEL_PATH = os.path.join(_CKPT_DIR, "final_model.zip")
N_EPISODES = 10
DETERMINISTIC = True # False = stochastic actions (more variety in eval)
TOUCH_TH = 0.16 # dist to count as ball contact (in m)
POSS_TH = 0.30 # dist to count as possession (in m)


# --------- #
# FUNCTIONS #
# --------- #

def _find_vecnorm(model_path: str) -> str | None:
    """
    Given  .../checkpoints/epoch_05_viper.zip
    return .../checkpoints/epoch_05_viper_vecnorm.pkl  if it exists.
    """
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    stem = os.path.splitext(model_path)[0]
    pkl = stem + "_vecnorm.pkl"
    return pkl if os.path.isfile(pkl) else None


def _build_vec_env(env_raw: SoccerEnv, model_path: str) -> VecNormalize:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    vec = DummyVecEnv([lambda: Monitor(env_raw)])
    pkl = _find_vecnorm(model_path)
    if pkl:
        print(f"VecNorm : {pkl}")
        vec = VecNormalize.load(pkl, vec)
        vec.training = False
        vec.norm_reward = False
    else:
        print("VecNorm : not found - raw rewards")
        vec = VecNormalize(vec, norm_obs=False, norm_reward=False, gamma=0.99)
    return vec


# --- EVAL --- #

def eval(
        model_path: str,
        n_simulations: int = 100,
        game_type: int = 0, #0-1: 0, 1-1: 1
        curr_stage: int = 3,
        deterministic: bool = True,
        env_raw: SoccerEnv | None = None,
) -> dict:
    """
    Description
        given an environment and the trained model, runs varous simulations to calculate evaluation metrics and then share and plot the results
    -----------
    Inputs:
        model_path: 
            path to .zip model file
        n_simulations: 
            number of episodes per robot type
        game_type: 
            0 for 1v0, 1 for 1v1
        curr_stage:    
            curriculum phase to use for spawn positions and time limit
        deterministic: 
            if True uses policy mean action, if False samples from distribution
        env_raw:   
            existing SoccerEnv instance -> pass from __main__ to avoid creating a second Supervisor. 
            if None, creates one internally
    
    -----------
    Output:
        dict: { "viper": {metric: value, ...}, "titan": {metric: value, ...} }
    """
    
    # gets environment, makes sure everything is properly setup, if there is a ball, and the number of robots determined by game_type (0 implies a 1-0, and 1 a 1-1 game) if incrrect send warning and use the real type
    # each simulation shall last has long as the default curr_stage tme defined in the SoccerEnv
    # 1-1 uses 1-0 metrics and a few more, only if in game_type==1 do the extra metrics get added to the dictionary of results
    # run simulation n_simulations times per robot types and get metrics
    # do the average of the metrics and store them in a dict with robot_type: eval_results_dict (per robot get average of n_simulations score)
    # 0-1 metrics: avg_reward, avg_goal_rate, avg_time_to_score, avg_internal_energy_usage
    # 1-1 metrics: same but also with avg_oponent_goal_rate, avg_robots_colision 

    owns_env = env_raw is None
    if owns_env:
        env_raw = SoccerEnv()

    opponent_node = getattr(env_raw, "_opponent_node", None)
    opponent_present = opponent_node is not None
    if game_type == 1 and not opponent_present:
        print("WARNING: game_type=1 (1v1) but no OPPONENT node found, will use game_type = 0\n")
        game_type = 0
    if game_type == 0 and opponent_present:
        print("WARNING: game_type=0 (1v0) but OPPONENT node found, will use game_type = 1\n")
        game_type = 1


    env_raw._curriculum_phase = curr_stage
    env_raw._curriculum_step = env_raw._step_from_phase(curr_stage)
    env_raw._lock_curriculum_phase = True
    #env_raw._max_steps(curr_stage)

    vec_env = _build_vec_env(env_raw, model_path)
    model = PPO.load(model_path, env=vec_env)

    results = {}

    for robot_name in ["viper", "titan"]:
        env_raw.swap_robot(robot_name)

        episode_results = []

        for ep in range(n_simulations):

            obs = vec_env.reset()
            ep_reward = 0.0
            ep_steps = 0
            touches = 0
            possession = 0
            shot_speeds = []
            collisions = 0
            done = False
            last_info = {}
            #to remove
            print(f"t check: it is at em: {ep} of {n_simulations}")

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = vec_env.step(action)
                ep_reward += float(reward[0])
                ep_steps += 1
                last_info = infos[0]
                done = bool(dones[0])

                ball_p = env_raw._ball_node.getPosition()
                robot_p = env_raw._robot_node.getPosition()
                dist_rb = math.hypot(ball_p[0]-robot_p[0], ball_p[2]-robot_p[2])

                if dist_rb < POSS_TH:
                    possession += 1

                if dist_rb < TOUCH_TH:
                    touches += 1
                    
                    try:
                        bv = env_raw._ball_node.getVelocity()
                        bspd = math.hypot(bv[0], bv[2])
                        to_x = -ball_p[0]
                        to_z = FIELD["goal_z_attack"] - ball_p[2]
                        n = math.hypot(to_x, to_z)
                        if n > 1e-6 and bspd > 0.01:
                            cos_sim = (bv[0]*to_x + bv[2]*to_z) / (n * bspd)
                            shot_speeds.append(bspd * max(0.0, cos_sim))
                    except Exception:
                        pass

                if game_type == 1:
                    opp_p = env_raw._opponent_node.getPosition()
                    dist_ro = math.hypot(robot_p[0]-opp_p[0], robot_p[2]-opp_p[2])
                    if dist_ro < 0.25: 
                        collisions += 1

            goal_scored = last_info.get("goal_scored", False)
            ep_dict = dict(
                reward = ep_reward,
                goal_scored = goal_scored,
                own_goal = last_info.get("own_goal",  False),
                ball_out = last_info.get("ball_out",  False),
                steps = ep_steps, #can remove
                touches = touches,
                possession_percent = possession / max(ep_steps, 1),
                shot_quality = float(np.mean(shot_speeds)) if shot_speeds else 0.0,
                time_to_score = ep_steps if goal_scored else None,
            )
            if game_type == 1:
                ep_dict["collisions"] = collisions

            episode_results.append(ep_dict)
        
        def mean_std(key):
            vals = [r[key] for r in episode_results if r[key] is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

        n_goals = sum(r["goal_scored"] for r in episode_results)
        summary = {
            "reward": mean_std("reward"),
            "goal_rate": n_goals / n_simulations,
            "ball_out_rate": sum(r["ball_out"] for r in episode_results) / n_simulations,
            "touches": mean_std("touches"),
            "possession_percent": mean_std("possession_percent"),
            "shot_quality": mean_std("shot_quality"),
            "time_to_score": mean_std("time_to_score"),
        }

        if game_type == 1:
            summary["opponent_goal_rate"] = (
                sum(r["own_goal"] for r in episode_results) / n_simulations
            )
            summary["collisions"] = mean_std("collisions")

        results[robot_name] = summary


    vec_env.close()
    env_raw._lock_curriculum_phase = False
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_FAST)
    if owns_env:
        env_raw.close()

    print("exited/finished")

    return results


# prob not gonna be used, for now can be ignored
def model_compare(
        list_of_model_paths: list[str], 
        n_episodes: int, 
        deterministic: bool,
        cmap: str = "rainbow"
    ) -> None:

    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """

    #adapt to new function eval()
    # plots bar charts comparing multiple models across the above metrics
    # 1 subplot per metric

    return None


# --- PLOTS --- #


#given the results: dict with robot_name: metrics plot 2 diferent graphs (percentual metrics, and non percentual metrics)
# percentual metrics: "goal_rate", "ball_out_rate", "possession_percent"
# non percentual metrics: all others
#plot per metric 2 columns (2 beacuse there are 2 robots) each robot has a colour associated (robot_colours)
# if save_path is not None save the plots in it (save_path last name - create 2 last_name_prcnt, last_name_n_prcnt)
def plot_eval_results(
        results: dict,
        game_type: int = 0,
        robot_colours: dict = {"viper": "red", "titan": "blue"},
        save_path: str | None = None,
) -> None:
    """
    Description
        plots evaluation metrics.

    -----------
    Inputs:
        results : dict
            dictionary returned by eval()
        game_type : int
            0 -> 1v0
            1 -> 1v1
        robot_colours : dict
            colour associated with each robot
        save_path : str | None
            base filename where figures should be saved
            produces:
                results/eval_percent.png
                results/eval_non_percent.png
    -----------
    Output:
        None -> plots (and saves) graphs
    """

    robots = list(results.keys())

    percent_metrics = [
        "goal_rate", "ball_out_rate", "possession_percent",
    ]

    non_percent_metrics = [
        "reward", "touches",
        "shot_quality", "time_to_score",
    ]

    if game_type == 1:
        percent_metrics.append("opponent_goal_rate")
        non_percent_metrics.append("collisions")

    def plot_metric_group(metrics, title):
        n = len(metrics)

        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(6 * cols, 4 * rows)
        )

        axes = np.array(axes).reshape(-1)

        for ax, metric in zip(axes, metrics):

            means = []
            stds = []

            for robot in robots:
                value = results[robot][metric]

                if isinstance(value, tuple):
                    mean, std = value
                else:
                    mean = value
                    std = 0.0

                means.append(mean)
                stds.append(std)

            x = np.arange(len(robots))

            ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                color=[robot_colours.get(r, "gray") for r in robots],
            )

            ax.set_xticks(x)
            ax.set_xticklabels([r.capitalize() for r in robots])
            ax.set_title(metric.replace("_", " ").title())

            if metric in percent_metrics:
                ax.set_ylim(0, 1)

            ax.grid(axis="y", alpha=0.3)

        for ax in axes[n:]:
            fig.delaxes(ax)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()

        return fig

    fig_percent = plot_metric_group(
        percent_metrics,
        "Percentage Metrics",
    )

    fig_non_percent = plot_metric_group(
        non_percent_metrics,
        "Non-Percentage Metrics",
    )

    if save_path is not None:
        save_path = Path(save_path)

        percent_path = (
            save_path.parent
            / f"{save_path.stem}_percent{save_path.suffix}"
        )

        non_percent_path = (
            save_path.parent
            / f"{save_path.stem}_non_percent{save_path.suffix}"
        )

        fig_percent.savefig(percent_path, dpi=300, bbox_inches="tight")
        fig_non_percent.savefig(non_percent_path, dpi=300, bbox_inches="tight")

    plt.show()
    return None





# --- SIMULATIONS --- #

def play_simulation(
    model_path: str = "checkpoints/ppo/final_model.zip",
    reward_fn: str = "_compute_reward",
    time: float = 112, # seconds
    deterministic: bool = True,
    env_raw: SoccerEnv | None = None,
    curr_stage: int = 3,
) -> None:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """

    

    GOAL_Z = FIELD["goal_z_attack"]
    owns_env = env_raw is None
    if owns_env:
        env_raw = SoccerEnv()

    # robot node - if none force it
    if env_raw.getFromDef("VIPER") is None:
        print("VIPER node not found - inserting default robot...")
        env_raw.getRoot().getField("children").importMFNodeFromString(
            -1,
            'DEF VIPER Viper {\n'
            '  translation 0 0.055 -1\n'
            '  rotation 1 0 0 -1.5707953071795862\n'
            '  name "viper"\n'
            '  controller "robot_controller"\n'
            '}'
        )
        env_raw._robot_node  = env_raw.getFromDef("VIPER")
        env_raw._active_robot = "viper"
        env_raw.set_reward_fn(reward_fn)

        # let the controller initialise
        for _ in range(10):
            env_raw._send_action(0.0, 0.0, 0.0)
            env_raw._sim_step()

    env_raw._robot_node = env_raw.getFromDef("VIPER")

    if env_raw._robot_node is None:
        raise RuntimeError(
            "VIPER node still not found after insertion attempt.\n"
            "Check that the Viper PROTO is declared as EXTERNPROTO in soccer.wbt."
        )
    

    # chose Curriculum step
    env_raw._curriculum_phase = curr_stage
    env_raw._curriculum_step = env_raw._step_from_phase(curr_stage)
    env_raw._lock_curriculum_phase = True

    # Switch to real-time so it's watchable
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_REAL_TIME)

    vec_env = _build_vec_env(env_raw, model_path)
    model = PPO.load(model_path, env=vec_env)

    # Convert time to max steps 
    # steps_per_act=5, timestep=8ms 
    # 40ms per RL step -> 25 steps/second
    steps_per_second = 1.0 / (SIM["steps_per_action"] * env_raw._timestep / 1000.0)
    max_steps = int(time * steps_per_second)

    print(f"\n --- Model: {model_path}  \n")
    print(f" --- Duration: {time}s  ({max_steps} steps) ")

    obs = vec_env.reset()
    ep = 1
    step = 0
    ep_reward = 0.0

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(reward[0])
        step += 1
        info = infos[0]

        if bool(dones[0]):
            outcome = (
                "GOAL" if info.get("goal_scored") else
                "OWN GOAL" if info.get("own_goal") else
                "OUT" if info.get("ball_out") else
                "TRUNC"
            )
            print(f"Episode {ep:>3}  {outcome}  R={ep_reward:+.2f}  "
                  f"steps_used={info.get('step', '?')}")
            ep += 1
            ep_reward = 0.0
            obs = vec_env.reset()

    vec_env.close()

    # Restore fast mode so training works normally if you switch back
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_FAST)

    if owns_env:
        env_raw.close()

    print(f"\nSSimulation done: {ep-1} episodes in {time}s")

    return None



def set_camera(
        env_raw: SoccerEnv, 
        orientation: list = [0.307, -0.674, -0.672, 2.55], 
        position: list = [9.48, 10.1, 0.275]
) -> None:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    vp = env_raw.getFromDef("VIEWPOINT")
    if vp is None:
        print("WARNING: VIEWPOINT node not found camera not set")
        return
    vp.getField("orientation").setSFRotation(orientation)
    vp.getField("position").setSFVec3f(position)

def start_recording(env_raw: SoccerEnv, filepath: str) -> bool:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    try:
        # quality 0-100, codec depends on extension
        env_raw.movieStartRecording(
            filepath,
            width=1280, height=720,
            codec=0,       # 0=default for the extension
            quality=90,
            acceleration=1,  # 1=real-time speed
            caption=False,
        )
        return True
    except Exception as e:
        print(f"[recording] Failed to start: {e}")
        return False
    
def stop_recording(env_raw: SoccerEnv) -> None:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    try:
        env_raw.movieStopRecording()
        # wait until file is written
        while env_raw.movieIsReady() is False:
            env_raw._sim_step()
    except Exception as e:
        print(f"[recording] Failed to stop: {e}")



def calculate_sim_oerformance(
        env_raw: SoccerEnv,
        vec_env: VecNormalize,
        model: PPO,
        c_orientation: list,
        c_position: list,
        reference_metrics: list[str] = ['goal_scored'],
        curr_best_score: float | None = None,
        save_file: str = "plays", 
        
) -> float:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    # will run a simulation and get the results considering reference_metrics
    # if they are better that the curr_best_score, it will be uptdates and the video of the play will be saved 
    
    os.makedirs(save_file, exist_ok=True)

    tmp_path = os.path.join(save_file, f"tmp_recording_{reference_metrics[0]}.mp4")
    set_camera(env_raw, c_orientation, c_position)
    recording_ok = start_recording(env_raw, tmp_path)

    obs = vec_env.reset()
    ep_reward = 0.0
    done = False
    last_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(reward[0])
        last_info = infos[0]
        done = bool(dones[0])

    if recording_ok:
        stop_recording(env_raw)

    ep_data = {
        "goal_scored": float(last_info.get("goal_scored", False)),
        "reward": ep_reward,
        "own_goal": float(last_info.get("own_goal", False)),
        "ball_out": float(last_info.get("ball_out", False)),
    }

    score = sum(abs(ep_data.get(m, 0.0))for m in reference_metrics)

    if curr_best_score is None or score > curr_best_score:
        final_path = os.path.join(save_file, f"best_play_{reference_metrics[0]}.mp4")
        if recording_ok and os.path.isfile(tmp_path):
            os.replace(tmp_path, final_path)
            print("video saved.")
        else:
            print("!new best score, but recording failed!")
        return score


    if recording_ok and os.path.isfile(tmp_path):
        os.remove(tmp_path)
    return curr_best_score

def force_cenario(
        model_path: str,
        max_simulations: int = 25, 
        robot_type: str = "viper",
        reference_metrics: list[str] = ['goal_scored'],
        save_file: str = "plays",
        deterministic: bool = True,
        env_raw: SoccerEnv | None = None,
        curr_stage: int = 3,
        c_orientation: list[float] = [0.307, -0.674, -0.672, 2.55],
        c_position: list[float] = [9.48, 10.1, 0.275],
) -> None:
    """
    Description
        description
    -----------
    Inputs:
        name(type): 
            description
    -----------
    Output:
        type: description
    """
    #set up
    owns_env = env_raw is None
    if owns_env:
        env_raw = SoccerEnv()

    env_raw.swap_robot(robot_type)
    env_raw._curriculum_phase = curr_stage
    env_raw._curriculum_step = env_raw._step_from_phase(curr_stage)
    env_raw._lock_curriculum_phase = True

    vec_env = _build_vec_env(env_raw, model_path)
    model = PPO.load(model_path, env=vec_env)

    best_score = None

    for i in range(max_simulations):
        print(f"Simulation {i+1}/{max_simulations}")

        best_score = calculate_sim_oerformance(
            env_raw=env_raw,
            vec_env=vec_env,
            model=model,
            c_orientation=c_orientation,
            c_position=c_position,
            reference_metrics=reference_metrics,
            curr_best_score=best_score,
            save_file=save_file,
        )

    vec_env.close()
    env_raw._lock_curriculum_phase = False
    env_raw.simulationSetMode(env_raw.SIMULATION_MODE_FAST)
    if owns_env:
        env_raw.close()


    #reference_metrics metrics can be: reward, goal_scored, ball_out, own_goal

    # run throu multiple simulations where the robot is robot_type, 
    # reference_metrics is the list of metrics that will define the best rank

    return None





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
    #model_evaluate(args.model, args.episodes, not args.stochastic)




# ------------ #
# --- MAIN --- #
# ------------ #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        _cli()
    else:
        model_evaluate(MODEL_PATH, N_EPISODES, DETERMINISTIC)