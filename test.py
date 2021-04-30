import sys

import ray
from ray.rllib.agents.dqn import DQNTrainer

from exceptional_rl_envs import Warehouse, MAP_WITH_EXCEPTION, MAP
from ray.tune.registry import register_trainable, register_env
from ray import tune
from ray.rllib import rollout

if __name__ == '__main__':
    ray.init(address='auto')

    def warehouse_env_creator(args):
        return Warehouse(**args)

    register_env('warehouse_env', warehouse_env_creator)

    if sys.gettrace():
        trainer = DQNTrainer(config={"env": "warehouse_env", "framework": "torch"})
        results = trainer.train()
        print(results)
        results = trainer.train()
        print(results)

    tune.run('DQN',
             config={"env": "warehouse_env",
                     "framework": "torch",
                     "num_gpus": 0.25,
                     "num_gpus_per_worker": 0.1,
                     'num_envs_per_worker': 6,
                     "evaluation_interval": 100,
                     "gamma": 0.95,
                     # "train_batch_size": 1024,
                     # "rollout_fragment_length": 50,
                     # "buffer_size": 100000,
                     "env_config": {
                         "var_pos": 0.125,
                         "var_dt": 0.1,
                         "maps": [MAP, MAP_WITH_EXCEPTION],
                     },
                     "evaluation_config": {"monitor": False,
                                           "explore": False,},
                     },
             metric='episode_reward_mean',
             checkpoint_freq=100,
             mode='max',)