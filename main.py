import os
import yaml
import argparse
import numpy as np
from mpi4py import MPI
# import tensorflow as tf
from torch import nn
from CNNModel import CustomCombinedExtractor

import gymnasium as gym
from gymnasium import envs
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback


def regist(task_env):
    if task_env == 'HybridGait-v0':
        register(
            id=task_env,
            entry_point='hybrid_gait_gym:HybridGaitGym',
        )
        import hybrid_gait_gym
    else:
        return None
    # We check that it was really registered
    all_envs = envs.registry.keys()
    env_ids = [env_spec for env_spec in all_envs]
    assert (task_env in env_ids), "The Task_Robot_ENV given is not Registered ==>" + str(task_env)

    print(str(task_env)+" Registed")
    return task_env


def build_env(task_env):
    task_env = regist(task_env)
    # task_env = "Pendulum-v0"
    env = gym.make(task_env)
    return env


def build_agent(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "features_extractor_class" : CustomCombinedExtractor,
        "net_arch": {"pi": [768, 768],
                      "vf": [768, 768]},
        # "act_fun": nn.ReLU
    }
    # timesteps_per_actorbatch = int(
    #     np.ceil(float(timesteps_per_actorbatch) / num_procs))
    # optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    agent = PPO(policy="MultiInputPolicy",
                 learning_rate = 1e-3,
                 env=env,
                 gamma=0.95,
                 n_steps=timesteps_per_actorbatch,
                 n_epochs=10,
                 batch_size=optim_batchsize,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log=output_dir,
                 verbose=1)
    return agent


def train(agent, env, num_train_episodes, output_dir="", int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            if not os.path.exists(int_dir):
                os.makedirs(int_dir)
            callbacks.append(CheckpointCallback(save_freq=int_save_freq,
                                                save_path=int_dir,
                                                name_prefix='model'))
    print("save_path", save_path)
    agent.learn(total_timesteps=num_train_episodes,
                # save_path=save_path,
                callback=callbacks)

    return


def test(agent, env, num_procs, num_episodes=None):

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    obs, _ = env.reset()
    for _ in range(num_local_episodes):
        action, _states = agent.predict(obs)
        obs, rewards, dones, _, info = env.step(action)

        if dones:
            break
        # if MPI.COMM_WORLD.Get_rank() == 0:
            # env.render()
    env.reset()
    env.close()
    return


def main():
    if not os.path.exists("output"):
        os.mkdir("output")

    if not os.path.exists("models"):
        os.mkdir("models")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task", dest="task",
                            type=str, default="HybridGait-v0")
    args = arg_parser.parse_args()

    with open('config/trainning_params.yaml') as f:
        training_params_dict = yaml.safe_load(f)
        if args.task in list(training_params_dict.keys()):
            training_params = training_params_dict[args.task]
        else:
            raise ValueError(
                "task not found for pybullet_sim_config.yaml")

    num_procs = MPI.COMM_WORLD.Get_size()

    env = build_env(args.task)
    agent = build_agent(env=env,
                        num_procs=num_procs,
                        timesteps_per_actorbatch=training_params["timesteps_per_actorbatch"],
                        optim_batchsize=training_params["optim_batchsize"],
                        output_dir=training_params["output_dir"])

    if training_params["model_file"] != "":
        agent = agent.load(training_params["model_file"])
        agent.set_env(env=env)
        # agent.load(training_params["model_file"])

    if training_params["mode"] == "train":
        train(agent=agent,
              env=env,
              num_train_episodes=training_params["num_train_episodes"],
              output_dir=training_params["output_dir"],
              int_save_freq=training_params["int_save_freq"])
    elif training_params["mode"] == "test":
        test(agent=agent,
             env=env,
             num_procs=num_procs,
             num_episodes=training_params["num_test_episodes"])
    else:
        assert False, "Unsupported mode: " + training_params["mode"]

    return


if __name__ == '__main__':
    main()
