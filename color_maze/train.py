#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
from ruamel.yaml import YAML

from env.color_maze_world import ColorMazeEnv
from rl_scripts.ppo_toy import PPO_Toy


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--result_dir",
        type=str,
        help="where to save the results.",
    )
    parser.add_argument(
        "--cfg_file", type=str, default="config.yaml", help="Config file name"
    )
    args = parser.parse_args()
    return args


def main():
    # initialization setup
    args = arg_parse()
    train_config = YAML().load(open(args.cfg_file, "r"))

    log_dir = os.path.join(
        args.result_dir, time.strftime("%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(log_dir, exist_ok=False)

    env = ColorMazeEnv(
        grid_size=train_config["grid_size"],
        margin_size=train_config["margin_size"],
        n_envs=train_config["n_envs"],
    )
    eval_env = ColorMazeEnv(
        grid_size=train_config["grid_size"],
        margin_size=train_config["margin_size"],
        n_envs=15,
    )

    # PPO
    model = PPO_Toy(
        policy="MlpPolicy",
        env=env,
        eval_env=eval_env,
        n_steps=train_config["n_steps"],
        batch_size=train_config["batch_size"],
        tensorboard_log=log_dir,
        verbose=1,
        ent_coef=train_config["ent_coef"],
        use_student=train_config["use_student"],
    )
    model.learn(
        total_timesteps=int(1 * 1e8),
        log_interval=10,
        eval_interval=100,
    )


if __name__ == "__main__":
    main()
