import numpy as np
import pandas as pd
import ray
from ray.rllib import VectorEnv
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray.cloudpickle import cloudpickle
import os
from exceptional_rl_envs import Warehouse, MAP, MAP_WITH_EXCEPTION

if __name__ == '__main__':
    ray.init(address='auto')

    NUM_EPISODES = 10000
    NUM_ENVS = 24

    columns = ["episode_id", "time_step", "x", "y", "v", "ort", "has_package",
               "package_x", "package_y", "action", "q_value_0", "q_value_1",
               "q_value_2", "q_value_3", "q_value_4", "reward", "done",
               "next_x", "next_y", "next_v", "next_ort", "next_has_package",
               "next_package_x", "next_package_y", "is_exception", "real_x", "real_y", "dt"]
    dtypes = {
        "episode_id": np.int,
        "time_step": np.int,
        "x": np.float,
        "y": np.float,
        "v": np.float,
        "ort": np.float,
        "has_package": np.bool,
        "package_x": np.float,
        "package_y": np.float,
        "action": np.int,
        "q_value_0": np.float,
        "q_value_1": np.float,
        "q_value_2": np.float,
        "q_value_3": np.float,
        "reward": np.float,
        "done": np.bool,
        "next_x": np.float,
        "next_y": np.float,
        "next_v": np.float,
        "next_ort": np.float,
        "next_has_package": np.bool,
        "next_package_x": np.float,
        "next_package_y": np.float,
        "is_exception": np.bool,
        "real_x": np.float,
        "real_y": np.float,
        "dt": np.float,
    }

    info_keys = ["exception", "real_x", "real_y", "dt"]
    def warehouse_env_creator(args):
        return Warehouse(**args)

    path = "/home/raphael/Experiments/ray/DQN/DQN_warehause_env_906fe_00000/checkpoint_{:06}/checkpoint-{}"
    checkpoint = 1800
    params_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'params.pkl')
    a = np.vectorize(lambda x, e: x[e])
    def process_info(infos):
        return np.array([[info[key] for key in info_keys] for info in infos])
    register_env('warehouse_env', warehouse_env_creator)
    # env_config = {'map': MAP}
    # env = VectorEnv.wrap(existing_envs=[warehouse_env_creator(env_config) for _ in range(NUM_ENVS)],
    #                      num_envs=NUM_ENVS)
    # config = {"env": "warehouse_env",
    #           "framework": "torch",
    #           "num_gpus": 0.1,
    #           "num_gpus_per_worker": 0.1,
    #           'num_envs_per_worker': 6,
    #           "evaluation_interval": 5, }
    with open(params_path, "rb") as f:
        config = cloudpickle.load(f)
    config["explore"] = False
    config['num_envs_per_worker'] = 1
    print("Trained on map: \n", config["env_config"]["maps"])
    config["env_config"]["maps"] = MAP_WITH_EXCEPTION
    trainer = DQNTrainer(config=config)
    trainer.restore(path.format(checkpoint, checkpoint))
    policy = trainer.get_policy()
    trainer._evaluate()
    samples = (trainer.evaluation_workers.local_worker().sample()
               for _ in range(NUM_EPISODES))
    rows = map(lambda x: np.concatenate([
        x["unroll_id"][:, None],
        np.arange(0, x.count)[:,None],
        x["obs"],
        x["actions"][:, None],
        x["q_values"],
        x["rewards"][:, None],
        x["dones"][:, None],
        x["new_obs"],
        process_info(x["infos"])],
        -1),
               samples)

    data = np.concatenate(list(rows), 0)



    df = pd.DataFrame(data, columns=columns)
    df = df.astype(dtypes)

    # sucess_rate
    df_ = df[["episode_id", "reward"]].copy()
    df_["success_rate"] = df.reward > 9
    df_ = df_.groupby("episode_id").sum()
    print("success rate: {}".format(np.mean(df_.success_rate==2.0)))
    print(df_.reward.describe())

    df.to_csv('data_exception2.csv', index=False,)


    #
    # registry = {
    #     env_id: df.copy() for env_id in range(NUM_ENVS)
    # }
    #
    # env_ids = np.arange(NUM_ENVS)[:, None]
    # max_env_ids = NUM_ENVS - 1
    # time_step = np.zeros((NUM_ENVS, 1))
    # obs = np.stack(env.vector_reset())
    # while max_env_ids < NUM_EPISODES:
    #     actions, _, extra_fetch = policy.compute_actions(obs)
    #     q_values = extra_fetch["q_values"]
    #     new_obs, reward, done, _ = env.vector_step(actions)
    #     new_obs = np.stack(new_obs)
    #     actions = actions.reshape([-1, 1])
    #     reward = np.array(reward)[:, None]
    #     done = np.array(done)[:, None]
    #     df = df.append(pd.DataFrame(
    #         np.concatenate(
    #             [env_ids, time_step, obs, actions, q_values, reward, done,
    #              new_obs], axis=-1),
    #         columns=columns,
    #     ))
    #     obs = new_obs
    #     time_step += 1
    #     for idx in np.argwhere(done.flatten()).flatten():
    #         max_env_ids += 1
    #         time_step[idx] = 0
    #         env_ids[idx] = max_env_ids
    #         obs[idx] = env.reset_at(idx)
    #
    # # Remove episodes
    # df = df.astype(dtypes)
    # df = df.reset_index(drop=True)
    # b = df.sort_values(by=["episode_id", "time_step"])
    # e = (b.groupby("episode_id").sum().done == 0)
    # c = b[~b.episode_id.isin(set(e[e].index))]
    # # df.to_csv("data.csv", index=False)