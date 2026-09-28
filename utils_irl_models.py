from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicRewardNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from util import data_dir, get_sorted_episodes, load_episode_full_log

BASE_DIR = Path(__file__).resolve().parent

def load_trained_model(
    subj_id,
    model_type = "rewnet_8_8_best",
    reward_hid_sizes=[8, 8],
    reward_net_activation = nn.LeakyReLU,
    train_rounds = None,
):
    model_path = BASE_DIR / "irl_models" / subj_id / model_type

    reward_data_path = model_path / f"reward_net{'' if train_rounds is None else f'_{train_rounds}'}.pt"
    reward_data = torch.load(reward_data_path, weights_only=False, map_location=torch.device('cpu'))
    reward_net = BasicRewardNet(
        observation_space=reward_data["obs_space"],
        action_space=reward_data["action_space"],
        hid_sizes=reward_hid_sizes,
        activation=reward_net_activation,
    )
    reward_state_dict = reward_data["model_state_dict"]
    reward_net.load_state_dict(reward_state_dict)
    reward_net.eval()

    gen_algo_path = model_path / f"generator_ppo{'' if train_rounds is None else f'_{train_rounds}'}.zip"
    gen_algo = PPO.load(gen_algo_path)
    gen_algo.policy.eval()

    return reward_net, gen_algo

def load_connected_traj(subjId, web, remove_initial_noop=False):
    subj_data_dir = data_dir(subjId, web)
    raw_episodes = get_sorted_episodes(subj_data_dir)
    episodes = []
    for e in raw_episodes:
        ep = int(e.strip())
        if ep < 1000: continue # filter practice session
        episodes.append(ep)
    full_log_list = [load_episode_full_log(subj_data_dir, episode, remove_initial_noop=remove_initial_noop, web=web) for episode in episodes]

    obs = []
    actions = []
    rewards = []
    infos = []
    for i in range(len(full_log_list)):
        full_log = full_log_list[i]
        episode_metadata = full_log["episode_metadata"]
        actions.append(full_log["actions"])
        rewards.append(full_log["rewards"])
        traj_len = len(full_log["actions"])
        infos.append([{"play_infos": pl, "episode_metadata": episode_metadata} for pl in full_log["play_infos"][:traj_len]])
        if i < len(full_log_list) - 1:
            obs.append(full_log["observations"][:traj_len])
        else:
            obs.append(full_log["observations"])

    obs=np.concatenate(obs, axis=0)
    actions=np.concatenate(actions, axis=0)
    rewards=np.concatenate(rewards, axis=0)
    infos=np.concatenate(infos, axis=0)

    return obs, actions, rewards, infos

def load_traj_per_episode(subjId, web, remove_initial_noop=False):
    subj_data_dir = data_dir(subjId, web)
    raw_episodes = get_sorted_episodes(subj_data_dir)
    episodes = []
    for e in raw_episodes:
        ep = int(e.strip())
        if ep < 1000: continue # filter practice session
        episodes.append(ep)
    full_log_list = [load_episode_full_log(subj_data_dir, episode, remove_initial_noop=remove_initial_noop, web=web) for episode in episodes]

    obs = []
    next_obs = []
    actions = []
    rewards = []
    infos = []
    for i in range(len(full_log_list)):
        full_log = full_log_list[i]
        episode_metadata = full_log["episode_metadata"]
        actions.append(full_log["actions"])
        rewards.append(full_log["rewards"])
        traj_len = len(full_log["actions"])
        infos.append([{"play_infos": pl, "episode_metadata": episode_metadata} for pl in full_log["play_infos"][:traj_len]])
        obs.append(full_log["observations"][:traj_len])
        next_obs.append(full_log["observations"][1:traj_len+1])

    obs=np.concatenate(obs, axis=0)
    next_obs=np.concatenate(next_obs, axis=0)
    actions=np.concatenate(actions, axis=0)
    rewards=np.concatenate(rewards, axis=0)
    infos=np.concatenate(infos, axis=0)
    return obs, next_obs, actions, rewards, infos

def calculate_action_prediction_performance(
    subjId, 
    reward_hid_sizes=[8,8],
    reward_net_activation = nn.LeakyReLU, 
    rounds = None,
    per_action = True, 
    verbose=False,
    web=False,
):
    obs, actions, rewards, infos = load_connected_traj(subjId, web, remove_initial_noop=True)
    reward_net, gen_algo = load_trained_model(
        subjId, 
        reward_hid_sizes=reward_hid_sizes, 
        reward_net_activation=reward_net_activation, 
        train_rounds=rounds,
    )
    predicted_actions, _ = gen_algo.predict(obs, deterministic=True)
    predicted_actions = predicted_actions[:len(actions)]
    assert len(actions) == len(predicted_actions)

    y_true = actions
    y_pred = predicted_actions

    ret = {}
    ret["overall_accuracy"] = accuracy_score(y_true, y_pred)
    ret["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    if per_action:
        for act in range(5):
            idxs = [i for i, a in enumerate(actions) if a == act]
            if len(idxs) == 0:
                ret[f"action_{act}_precision"] = np.nan
                ret[f"action_{act}_recall"] = np.nan
                ret[f"action_{act}_f1"] = np.nan
                continue
            ret[f"action_{act}_precision"] = precision[act]
            ret[f"action_{act}_recall"] = recall[act]
            ret[f"action_{act}_f1"] = f1[act]
            action_frequency = len([i for i, a in enumerate(actions) if a == act])
            if verbose:
                print(f"action_{act}_f1={f1[act]:.4f} (count={action_frequency})")

    if verbose:
        print(f"----- subjId: {subjId} -----")
        print(f"accuracy={ret['overall_accuracy']:.4f}")
        print(f"weighted_f1={ret['weighted_f1']:.4f}")
    return ret
