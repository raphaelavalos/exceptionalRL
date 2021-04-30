import sys
import os
import ray
from ray.rllib.agents.dqn import DQNTrainer

from exceptional_rl_envs import Warehouse
from ray.tune.registry import register_trainable, register_env
from ray import tune


if __name__ == '__main__':
    ray.init(address='auto')

    def warehouse_env_creator(args):
        return Warehouse()



    register_env('warehouse_env', warehouse_env_creator)
    path = "/home/raphael/ray_results/DQN/DQN_warehouse_env_42ecd_00000_0_2021-02-03_12-45-06"
    for dir in filter(lambda  x: os.path.isdir(os.path.join(path, x)), os.listdir(path)):
        print(dir)
