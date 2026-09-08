import numpy as np
import gymnasium as gym

from imitation.data.types import TrajectoryWithRew

from util import data_dir, get_sorted_episodes, load_episode_full_log

def load_subject_play_log(
        subjId, 
        add_reverse_obs=False, 
        remove_initial_noop=False, 
        web=False,
        remove_same_action_only_eps=False, # NOTE: assumes that remove_initial_noop is True
    ):
    subj_data_dir = data_dir(subjId, web)
    raw_episodes = get_sorted_episodes(subj_data_dir)
    episodes = []
    for e in raw_episodes:
        ep = int(e.strip())
        if ep < 1000: continue # filter practice session
        episodes.append(ep)
    full_log_list = []
    for episode in episodes:
        full_log = load_episode_full_log(
            subj_data_dir, 
            episode, 
            remove_initial_noop=remove_initial_noop, 
            web=web,
        )
        if remove_same_action_only_eps and len(set(full_log["actions"])) == 1:
            continue
        full_log_list.append(full_log)
    max_traj_size = max(len(full_log["actions"]) for full_log in full_log_list)

    if add_reverse_obs:
        for episode in episodes:
            full_log_list.append(load_episode_full_log(subj_data_dir, episode, reverse_obs=add_reverse_obs, remove_initial_noop=remove_initial_noop, web=web))

    return full_log_list, max_traj_size, episodes

def load_traj(subjId, fixed_horizon=True, obs_norm_env=None, data_augmentation=False, connect_traj=False, remove_initial_noop=False, web=False, remove_same_action_only_eps=False):
    full_log_list, max_traj_size, _ = load_subject_play_log(subjId, add_reverse_obs=data_augmentation, remove_initial_noop=remove_initial_noop, web=web, remove_same_action_only_eps=remove_same_action_only_eps)
    expert_trajectories = []
    if connect_traj:
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

        traj = TrajectoryWithRew(
            obs=np.concatenate(obs, axis=0),
            acts=np.concatenate(actions, axis=0),
            rews=np.concatenate(rewards, axis=0),
            infos=np.concatenate(infos, axis=0),
            terminal=False
        )
        expert_trajectories.append(traj)
    else:
        for full_log in full_log_list:
            episode_metadata = full_log["episode_metadata"]
            observations = full_log["observations"]
            actions = full_log["actions"]
            rewards = full_log["rewards"]
            infos = [{"play_infos": pl, "episode_metadata": episode_metadata} for pl in full_log["play_infos"][1:]]
            if fixed_horizon:
                # add abosorb flag to obs
                traj = create_fixed_horizon_trajectory(observations, actions, rewards, infos, max_traj_size)
            else:
                traj = TrajectoryWithRew(
                    obs=np.asarray(observations),
                    acts=actions,
                    rews=rewards,
                    infos=infos,
                    terminal=True
                )
            expert_trajectories.append(traj)

    # normalize
    if obs_norm_env is not None:
        obs_all = np.concatenate([tr.obs for tr in expert_trajectories], axis=0).astype(np.float32)
        D = obs_all.shape[1]
        bool_idx_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 33, 34, 35, 36, 37, 38, 39, 40]
        if fixed_horizon and not connect_traj:
            ABSORB_IDX = D-1
            bool_idx_list.append(ABSORB_IDX)
        bool_idxs = np.array(bool_idx_list, dtype=int)
        _set_vecnorm_stats_from_demos(obs_norm_env, obs_all, bool_idxs=bool_idxs)
        for i, traj in enumerate(expert_trajectories):
            obs = traj.obs.astype(np.float32, copy=True)
            raw_bool = obs[:, bool_idxs].copy()
            obs_norm = obs_norm_env.normalize_obs(obs) # 전체 정규화                       
            obs_norm[:, bool_idxs] = raw_bool # boolean 복원
            expert_trajectories[i] = TrajectoryWithRew(
                obs=obs_norm,
                acts=traj.acts,
                rews=traj.rews,
                infos=traj.infos,
                terminal=traj.terminal,
            )
    return expert_trajectories

# connect_traj not implemented
def load_full_traj(web_subj_ids=[], desktop_subj_ids=[], fixed_horizon=True, obs_norm_env=None, data_augmentation=False, connect_traj=False, remove_initial_noop=False, remove_same_action_only_eps=False):
    subj_to_full_log_list = {}
    max_traj_size_full = -1
    for subjId in web_subj_ids:
        web=True
        full_log_list, max_traj_size, _ = load_subject_play_log(subjId, add_reverse_obs=data_augmentation, remove_initial_noop=remove_initial_noop, web=web, remove_same_action_only_eps=remove_same_action_only_eps)
        subj_to_full_log_list[subjId] = full_log_list
        max_traj_size_full = max(max_traj_size_full, max_traj_size)
    for subjId in desktop_subj_ids:
        web=False
        full_log_list, max_traj_size, _ = load_subject_play_log(subjId, add_reverse_obs=data_augmentation, remove_initial_noop=remove_initial_noop, web=web, remove_same_action_only_eps=remove_same_action_only_eps)
        subj_to_full_log_list[subjId] = full_log_list
        max_traj_size_full = max(max_traj_size_full, max_traj_size)

    expert_trajectories = []
    for subjId in subj_to_full_log_list.keys():
        full_log_list = subj_to_full_log_list[subjId]
        for full_log in full_log_list:
            episode_metadata = full_log["episode_metadata"]
            observations = full_log["observations"]
            actions = full_log["actions"]
            rewards = full_log["rewards"]
            infos = [{"play_infos": pl, "episode_metadata": episode_metadata} for pl in full_log["play_infos"][1:]]
            if fixed_horizon:
                # add abosorb flag to obs
                traj = create_fixed_horizon_trajectory(observations, actions, rewards, infos, max_traj_size_full)
            else:
                traj = TrajectoryWithRew(
                    obs=np.asarray(observations),
                    acts=actions,
                    rews=rewards,
                    infos=infos,
                    terminal=True
                )
            expert_trajectories.append(traj)

    # normalize
    if obs_norm_env is not None:
        obs_all = np.concatenate([tr.obs for tr in expert_trajectories], axis=0).astype(np.float32)
        D = obs_all.shape[1]
        bool_idx_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 33, 34, 35, 36, 37, 38, 39, 40]
        if fixed_horizon and not connect_traj:
            ABSORB_IDX = D-1
            bool_idx_list.append(ABSORB_IDX)
        bool_idxs = np.array(bool_idx_list, dtype=int)
        _set_vecnorm_stats_from_demos(obs_norm_env, obs_all, bool_idxs=bool_idxs)
        for i, traj in enumerate(expert_trajectories):
            obs = traj.obs.astype(np.float32, copy=True)
            raw_bool = obs[:, bool_idxs].copy()
            obs_norm = obs_norm_env.normalize_obs(obs) # 전체 정규화                       
            obs_norm[:, bool_idxs] = raw_bool # boolean 복원
            expert_trajectories[i] = TrajectoryWithRew(
                obs=obs_norm,
                acts=traj.acts,
                rews=traj.rews,
                infos=traj.infos,
                terminal=traj.terminal,
            )
    return expert_trajectories, max_traj_size_full

def _set_vecnorm_stats_from_demos(env, obs_all, bool_idxs):
    D = obs_all.shape[1]
    all_idx = np.arange(D, dtype=int)
    cont_idxs = np.setdiff1d(all_idx, bool_idxs, assume_unique=False)
    mean_full = np.zeros(D, dtype=np.float32)
    var_full  = np.ones(D,  dtype=np.float32)
    if cont_idxs.size > 0:
        cont = obs_all[:, cont_idxs]
        mean_full[cont_idxs] = cont.mean(axis=0).astype(np.float32)
        var_est = cont.var(axis=0).astype(np.float32)
        var_est[var_est < 1e-8] = 1e-8
        var_full[cont_idxs] = var_est
    rms = env.obs_rms # stable-baselines3.common.running_mean_std.RunningMeanStd
    rms.mean[:] = mean_full
    rms.var[:]  = var_full
    if hasattr(rms, "count"):
        rms.count = float(obs_all.shape[0])
    env.training = False

################################################ FixedHorizon ################################################
class FixedHorizonEnvWrapper(gym.Wrapper):
    """
    Add absorbing indicator feature for all the observations (0=ongoing, 1=absorbing)

    Add additional transitions until reaching `max_traj_size`.
    `truncated = True` only when reaching `max_traj_size`

    (check `Variable Horizon Environments Considered Harmful`_).

    .. _Variable Horizon Environments Considered Harmful: https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
    """
    def __init__(self, env, max_traj_size: int):
        super().__init__(env)
        self.max_traj_size = int(max_traj_size)
        self.cur_step = 0
        self.is_absorbing = False

        assert isinstance(env.observation_space, gym.spaces.Box)
        low  = env.observation_space.low
        high = env.observation_space.high
        obs_space_size = len(low) + 1
        assert obs_space_size == 114
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, [0.0]], axis=0),
            high=np.concatenate([high, [1.0]], axis=0),
            dtype=env.observation_space.dtype
        )
        self._absorbing_obs = _make_absorbing_obs(obs_space_size)

    def reset(self, **kw):
        self.cur_step = 0
        self.is_absorbing = False
        obs, info = self.env.reset(**kw)
        return _wrap_non_absorbing_obs(obs), info

    def step(self, action):
        self.cur_step += 1
        if not self.is_absorbing:
            obs, rew, terminated, truncated, info = self.env.step(action)
            obs = _wrap_non_absorbing_obs(obs)
            if terminated or truncated:
                self.is_absorbing = True # start absorbing after current transition
        else:
            obs = self._absorbing_obs
            rew = 0.0
            info = {}
        truncated = self.cur_step >= self.max_traj_size
        return obs, rew, False, truncated, info

def create_fixed_horizon_trajectory(
        observations, # shape: (T+1,), dtype: float32
        actions,      # shape: (T,), dtype: int64
        rewards,      # shape: (T,), dtype: float32
        infos,        # shape: (T,)
        max_traj_size
    ):
    T = len(actions)
    assert len(observations) == T+1
    assert len(rewards) == T
    assert len(infos) == T
    assert max_traj_size >= T

    # add absorbing indicator feature (0=ongoing, 1=truncated)
    observations = [_wrap_non_absorbing_obs(obs) for obs in observations]
    obs_space_size = len(observations[0])
    assert obs_space_size == 114
    pad_T = max_traj_size - T
    if pad_T > 0:
        absorbing_obs = _make_absorbing_obs(obs_space_size)
        observations = observations + [absorbing_obs]*pad_T # shape: (max_traj_size+1, obs_space_size)

        # do nothing and get no reward
        actions_pad = np.full((pad_T), 0, dtype=np.int64) # NOTE: 0 == Action.NOTHING.value
        rewards_pad = np.zeros((pad_T), dtype=np.float32)
        actions = np.concatenate([actions, actions_pad], axis=0)
        rewards = np.concatenate([rewards, rewards_pad], axis=0)
        
        # fill up empty infos 
        infos = list(infos) + [{} for _ in range(pad_T)]

    return TrajectoryWithRew(
        obs=np.asarray(observations),
        acts=actions,
        rews=rewards,
        infos=infos,
        terminal=True
    )

# NOTE: absorbing indicator should not be normalized
def _make_absorbing_obs(obs_space_size):
    absorbing_obs = np.zeros(obs_space_size, dtype=np.float32)
    absorbing_obs[-1] = 1.0
    return absorbing_obs

def _wrap_non_absorbing_obs(obs):
    return np.concatenate([obs, [0]], axis=0)
