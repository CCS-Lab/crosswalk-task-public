import sys
import os

import numpy as np
from pathlib import Path
import json
import gzip

def data_dir(subjId, web=False):
    if web:
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        prolific_id_dir = current_dir / "data" / "web" / str(subjId)
        # prolific_id_dir = current_dir / "web" / "server" / "data" / str(subjId)

        study_ids = [p for p in prolific_id_dir.iterdir() if p.is_dir()]
        if len(study_ids) != 1:
            raise ValueError(f"{subjId} must contain exactly 1 study directory, found {len(study_ids)}")
        study_id_dir = study_ids[0]
        
        session_ids = [p for p in study_id_dir.iterdir() if p.is_dir()]
        if len(session_ids) != 1:
            raise ValueError(f"{subjId} must contain exactly 1 session directory, found {len(session_ids)}")
        session_id_dir = session_ids[0]
        return session_id_dir
    else:
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        data_dir = current_dir / "data" / "desktop" / str(subjId)
        return data_dir

def get_sorted_episodes(subj_data_dir):
    episodes = []
    for sub_dir in subj_data_dir.iterdir():
        if sub_dir.is_dir():
            episodes.append(sub_dir.name)
    return sorted(episodes)

def _load_gzip_float32(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype="<i4").copy()  # little-endian int32

def _load_gzip_float32_list(path, num_cols = 113):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype="<f4").copy()  # little-endian float32, copy to make it non-read-only
    if arr.size % num_cols != 0:
        raise ValueError("Data size not divisible by num_cols")
    return arr.reshape(-1, num_cols)

def load_episode_full_log(subj_data_dir, episode, reverse_obs=False, remove_initial_noop=False, web=False):
    episode = str(episode)
    episode_dir = subj_data_dir / episode

    if web:
        EPISODE_METADATA = "episode_metadata.json"
        PLAY_INFOS = "play_infos.jsonl"
        OBSERVATIONS = "observations.gzip"
        ACTIONS = "actions.gzip"
        REWARDS = "rewards.gzip"
        obs = _load_gzip_float32_list(episode_dir / OBSERVATIONS)
        actions = _load_gzip_float32(episode_dir / ACTIONS)
        rewards = _load_gzip_float32(episode_dir / REWARDS).astype(np.float32)
        with open(episode_dir / EPISODE_METADATA, "r", encoding="utf-8") as f:
            episode_metadata = json.load(f)
        with open(episode_dir / PLAY_INFOS, "r", encoding="utf-8") as f:
            play_infos = [json.loads(line) for line in f]
    else:
        EPISODE_METADATA = "episode_metadata.npy"
        PLAY_INFOS = "play_infos.npy"
        OBSERVATIONS = "observations.npy"
        ACTIONS = "actions.npy"
        REWARDS = "rewards.npy"
        obs = np.load(episode_dir / OBSERVATIONS)
        play_infos = np.load(episode_dir / PLAY_INFOS, allow_pickle=True)
        actions = np.load(episode_dir / ACTIONS)
        rewards = np.load(episode_dir / REWARDS).astype(np.float32)
        episode_metadata = np.load(episode_dir / EPISODE_METADATA, allow_pickle=True)[0]

    # NOTE: only right/left info inside obs are swapped, not info
    if reverse_obs:
        for ob in obs:
            assert ob.size == 113
            orig_ob = ob.copy()
            swap_idx_pairs = [
                # Meta info [0–4]
                (3, 4), # left_end_closeness <=> right_end_closeness
                # Tile types [5–12]
                (11, 12), # is_crosswalk: right <=> left
                # Target road info [13–23]
                (18, 19), # dangerous_car_counts: NW <=> NE
                (20, 21), # dangerous_car_counts: SE <=> SW
                (22, 23), # dangerous_car_counts: left <=> right
                # Distance from crosswalks [24–32]
                (25, 26), # crosswalk1_closeness_list: right <=> left
                (28, 29), # crosswalk2_closeness_list: right <=> left
                (31, 32), # crosswalk3_closeness_list: right <=> left
                # Cars
                (33, 35), # car_in_adjecent_tiles: NW <=> NE
                (36, 40), # car_in_adjecent_tiles: E <=> W
                (37, 39), # car_in_adjecent_tiles: SE <=> SW
            ]
            for a, b in swap_idx_pairs:
                ob[a], ob[b] = orig_ob[b], orig_ob[a]
            
            # dangerous_car_info_list
            start_idx, grid_rows, stride = 41, 6, 12
            for k in range(grid_rows):
                base = start_idx + stride * k
                # [head closeness, tail closeness, tail distance, speed]: left_cars <=> right_cars
                ob[base:base+4] = orig_ob[base+4:base+8]
                ob[base+4:base+8] = orig_ob[base:base+4]
                # [head closeness, tail closeness]: left_cars <=> right_cars
                ob[base+8:base+10] = ob[base+10:base+12]
                ob[base+10:base+12] = ob[base+8:base+10]

        RIGHT, LEFT = 3, 4
        for i in range(len(actions)):
            action = actions[i]
            if action == RIGHT:
                actions[i] = LEFT
            elif action == LEFT:
                actions[i] = RIGHT

    if remove_initial_noop:
        consequtive_do_nothings = 0
        for i in range(len(actions)):
            if actions[i] > 0: break
            consequtive_do_nothings += 1
        obs = obs[consequtive_do_nothings:]
        play_infos = play_infos[consequtive_do_nothings:]
        actions = actions[consequtive_do_nothings:]
        rewards = rewards[consequtive_do_nothings:]

    return {
        "episode_metadata": episode_metadata,
        "observations": obs,
        "play_infos":play_infos,
        "actions": actions,
        "rewards": rewards,
    }

def validate_no_op_episodes(subjIds, web, verbose=False):
    same_movement_detected = False
    noop_detected = []
    noop_10sec_detected = []
    for subjId in subjIds:
        subj_data_dir = data_dir(subjId, web=web)
        episodes = []
        for e in get_sorted_episodes(subj_data_dir):
            ep = int(e.strip())
            if ep < 1000: continue # filter practice session
            episodes.append(ep)        
        same_movements = []
        same_movement_eps = []
        noop_only = []
        noop_only_eps = []
        noop_for_10sec = []
        noop_for_10sec_eps = []
        for episode in episodes:
            full_log_trimmed = load_episode_full_log(subj_data_dir, episode, remove_initial_noop=True, web=web)
            full_log = load_episode_full_log(subj_data_dir, episode, remove_initial_noop=False, web=web)
            if len(set(full_log_trimmed["actions"])) == 1:
                same_movements.append(full_log["actions"])
                same_movement_eps.append(episode)
            if len(set(full_log["actions"])) == 1:
                noop_only.append(full_log["actions"])
                noop_only_eps.append(episode)
            if len(set(full_log["actions"][:50])) == 1:
                noop_for_10sec.append(full_log["actions"])
                noop_for_10sec_eps.append(episode)
        
        if len(same_movements) >= 1:
            print(f"{subjId}: same movement episode detected - {len(same_movements)} out of {len(episodes)} episodes")
            if verbose:
                for i in range(len(same_movements)):
                    print(same_movement_eps[i], same_movements[i]) # NOTE: 0=do nothing, 1=UP, 2=DOWN, 3=RIGHT, 4=LEFT
            same_movement_detected = True
        if len(noop_only) >= 1:
            print(f"{subjId}: noop episode detected - {len(noop_only)} out of {len(episodes)} episodes")
            print(noop_only_eps)
            for noop in noop_only:
                print(noop) # NOTE: 0=do nothing, 1=UP, 2=DOWN, 3=RIGHT, 4=LEFT
            noop_detected.append(subjId)
        if len(noop_for_10sec) >= 1:
            print(f"{subjId}: noop for over 10 seconds episode detected - {len(noop_for_10sec)} out of {len(episodes)} episodes")
            print(noop_for_10sec_eps)
            for noop in noop_for_10sec:
                print(noop) # NOTE: 0=do nothing, 1=UP, 2=DOWN, 3=RIGHT, 4=LEFT
            noop_10sec_detected.append(subjId)

    print(f"same movement detection: {same_movement_detected}")
    print(f"no op detection: {noop_detected}")
    print(f"no op for 10 seconds detection: {noop_10sec_detected}")

if __name__ == "__main__":
    web = True
    subjId = "6938b3a102d6dfae6c72a140"
    # web = False
    # subjId = 100
    double_saved_cnt = 0

    subj_data_dir = data_dir(subjId, web=web)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode, web=web)
        print()
        print(subj_data_dir / episode)
        print("env_configuration")
        print(full_log["episode_metadata"]["env_configuration"])
        print("episode_configuration")
        print(full_log["episode_metadata"]["episode_configuration"])
        print("play_infos")
        print(full_log["play_infos"][0])
        print("observations")
        # print(full_log["observations"][0])
        # print(full_log["observations"][1])
        # print(full_log["observations"][2])
        print("observation size:", len(full_log["observations"][0]))
        # print("actions:", full_log["actions"][:10])
        # print("rewards:", full_log["rewards"][:10])
        assert type(full_log["rewards"][0]) == np.float32 or type(full_log["rewards"][0]) == np.float64
        print("trajectory size:", len(full_log["actions"]))

        print(len(full_log["play_infos"]), len(full_log["observations"]))

        print(full_log["play_infos"][0] == full_log["play_infos"][1])

        if len(full_log["play_infos"]) > len(full_log["observations"]):
            assert len(full_log["play_infos"]) == len(full_log["observations"]) * 2
            double_saved_cnt += 1
            print("double saved")
            # for i in range(len(full_log["actions"])):
            #     print("true")
            #     assert full_log["play_infos"][i*2] == full_log["play_infos"][i*2 + 1]
        else:
            assert len(full_log["play_infos"]) == len(full_log["observations"])

        assert len(full_log["actions"]) == len(full_log["rewards"])
        assert len(full_log["observations"]) == len(full_log["actions"]) + 1
        # break
    print("episodes:", len(episodes))
    print("double_saved_cnt:", double_saved_cnt)