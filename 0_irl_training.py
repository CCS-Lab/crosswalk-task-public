import os

import random
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger
import gymnasium as gym

import crosswalk_env
from rl_util import load_traj

class InfiniteHorizonEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            if self.env.cur_seed is not None:
                obs, info = self.reset(self.env.cur_seed+1)
            else:
                obs, info = self.reset()
        return obs, rew, False, False, info

def train_AIRL(
        subjId,
        ppo_n_steps,
        gen_train_timesteps,
        web,
        gen_replay_buffer_capacity=2,

        # optimal PPO configuration
        ppo_lr=1e-4,
        ppo_batch_size=32,
        ppo_n_epochs=8,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_range=0.2,
        ppo_ent_coef=0.01,
        ppo_vf_coef=0.5,
        ppo_net_arch=[256, 256, 256],
        ppo_activation=nn.LeakyReLU,

        demo_batch_size=256,
        resume_round: int | None = None,
        airl_train_n_rounds=4000,
        n_disc_updates_per_round=1,
        disc_optimizer_lr=1e-3,
        disc_optimizer_weight_decay=5e-4,
        env_norm_obs=False,
        env_norm_reward=False,
        reward_net_hid_sizes=[8, 8],
        reward_net_activation=nn.LeakyReLU,

        data_augmentation=True,
        seed=42,
        sub_dir=None,
    ):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_dir = f"./tb_logs/{subjId}/{sub_dir}"
    save_dir = f"./irl_models/{subjId}/{sub_dir}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        gym.register_envs(crosswalk_env)
        env = gym.make("crosswalk-v1")
        env = InfiniteHorizonEnvWrapper(env)
        env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    if resume_round is None or not (env_norm_obs or env_norm_reward):
        env = VecNormalize(env, norm_obs=env_norm_obs, norm_reward=env_norm_reward)
    else:
        # resuming AND env_norm_obs==True or env_norm_reward==True
        vec_path = os.path.join(save_dir, f"vecnormalize_{resume_round}.pkl")
        env = VecNormalize.load(vec_path, env)
        env.training = True
        env.norm_obs = env_norm_obs
        env.norm_reward = env_norm_reward

    demonstrations = load_traj(
        subjId=subjId,
        fixed_horizon=False,
        obs_norm_env=env if env_norm_obs else None,
        data_augmentation=data_augmentation,
        connect_traj=True,
        remove_initial_noop=True,
        web=web,
        remove_same_action_only_eps=True,
    )
    if resume_round is None:
        gen_algo = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_lr,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=ppo_n_epochs,
            gamma=ppo_gamma,
            gae_lambda=ppo_gae_lambda,
            clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef,
            vf_coef=ppo_vf_coef,
            tensorboard_log=f"{log_dir}ppo/",
            policy_kwargs=dict(net_arch=ppo_net_arch, activation_fn=ppo_activation), # or nn.LeakyReLU
            verbose=1,
            seed=seed,
        )
    else:
        ppo_path = os.path.join(save_dir, f"generator_ppo_{resume_round}.zip")
        gen_algo = PPO.load(ppo_path, env=env, device="auto")

    reward_net = BasicRewardNet(
        env.observation_space,
        env.action_space,
        hid_sizes=reward_net_hid_sizes,
        activation=reward_net_activation,
    )
    if resume_round is not None:
        rn_path = os.path.join(save_dir, f"reward_net_{resume_round}.pt")
        ckpt = torch.load(rn_path, map_location="cpu", weights_only=False)
        reward_net.load_state_dict(ckpt["model_state_dict"])   
    if gen_replay_buffer_capacity is None:
        grbc = gen_train_timesteps
    else:
        grbc = gen_train_timesteps * gen_replay_buffer_capacity         
    trainer = airl.AIRL(
        demonstrations=demonstrations,
        demo_batch_size=demo_batch_size,
        # demo_minibatch_size=demo_batch_size,
        venv=env,
        gen_algo=gen_algo,
        reward_net=reward_net,

        # The number of discriminator updates after each round of generator updates in AdversarialTrainer.learn().
        # # default: 2
        n_disc_updates_per_round=n_disc_updates_per_round,

        # The number of steps to train the generator policy for each iteration.
        # If None, then defaults to the batch size (for on-policy) or number of environments (for off-policy).
        # default: gen_algo.n_steps * gen_algo_env.num_envs
        gen_train_timesteps=gen_train_timesteps,

        # The capacity of the generator replay buffer (the number of obs-action-obs samples from the generator that can be stored). 
        # By default this is equal to `gen_train_timesteps`, meaning that we sample only from the most recent batch of generator samples.
        # default: gen_train_timesteps
        gen_replay_buffer_capacity=grbc,

        # default for Adam: dict(lr=1e-3, weight_decay=0),
        disc_opt_kwargs=dict(lr=disc_optimizer_lr, weight_decay=disc_optimizer_weight_decay),

        log_dir=log_dir,
        custom_logger=logger.configure(
            folder=log_dir,
            format_strs=["stdout", "csv", "tensorboard"],
        ),
        init_tensorboard=True, # If True, makes various discriminator TensorBoard summaries.
        init_tensorboard_graph=True, # If both this and `init_tensorboard` are True, then write a Tensorboard graph summary to disk.
    )
    if resume_round is not None:
        trainer._global_step = int(resume_round)
        trainer._disc_step = int(resume_round) * int(n_disc_updates_per_round)
        gen_algo.num_timesteps = int(resume_round) * int(gen_train_timesteps)
    def train_callback(round):
        if resume_round is not None:
            round += resume_round
        if round % 10 == 0:
            torch.save({
                'model_state_dict': reward_net.state_dict(),
                'obs_space': reward_net.observation_space,
                'action_space': reward_net.action_space
            }, os.path.join(save_dir, f"reward_net_{round}.pt"))
            gen_algo.save(os.path.join(save_dir, f"generator_ppo_{round}.zip"))
            if env_norm_obs or env_norm_reward:
                if isinstance(env, VecNormalize):
                    env.save(os.path.join(save_dir, f"vecnormalize_{round}.pkl"))

            metrics = {}
            for key in [
                "mean/disc/disc_acc",
                "mean/disc/disc_acc_expert",
                "mean/disc/disc_acc_gen",
            ]:
                metrics[key] = (
                    trainer.logger.name_to_value.get(key)
                    if key in trainer.logger.name_to_value
                    else gen_algo.logger.name_to_value.get(key)
                )

            for k, v in metrics.items():
                if v is None:
                    metrics[k] = float("nan")

            readme_path = os.path.join(save_dir, "README.md")
            with open(readme_path, "a") as f:
                f.write(
                    f"Round {round:>6d} | "
                    f"disc_acc={metrics['mean/disc/disc_acc']:.4f} | "
                    f"disc_acc_expert={metrics['mean/disc/disc_acc_expert']:.4f} | "
                    f"disc_acc_gen={metrics['mean/disc/disc_acc_gen']:.4f}\n"
                )
            print(f"[INFO] Saved models and metrics at round {round}")
    if resume_round is None:
        rounds_to_run = airl_train_n_rounds
    else:
        rounds_to_run = airl_train_n_rounds - resume_round
    trainer.train(total_timesteps=gen_train_timesteps * rounds_to_run, callback=train_callback)

    torch.save({
        'model_state_dict': reward_net.state_dict(),
        'obs_space': reward_net.observation_space,
        'action_space': reward_net.action_space
    }, f"{save_dir}reward_net_{airl_train_n_rounds}.pt")
    gen_algo.save(f"{save_dir}generator_ppo_{airl_train_n_rounds}.zip")
    if env_norm_obs or env_norm_reward:
        if isinstance(env, VecNormalize):
            env.save(os.path.join(save_dir, f"vecnormalize_{airl_train_n_rounds}.pkl"))

if __name__ == "__main__":
    subjId = "" # TODO
    web=False
    sub_dir="rewnet_8_8_best/"
    ppo_n_steps=512
    train_AIRL(
        subjId=subjId,
        web=web,
        ppo_n_steps=ppo_n_steps,
        gen_train_timesteps=ppo_n_steps * 1,
        resume_round=None,
        airl_train_n_rounds=20_000,
        sub_dir=sub_dir,
        data_augmentation=True,
    )