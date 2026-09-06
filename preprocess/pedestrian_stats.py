import pandas as pd

import statistics
from collections import Counter

from util import data_dir, get_sorted_episodes, load_episode_full_log

def build_pedestrian_task_results(df, subjIds, web=False):
    scores_df, episode_result_df, penalty_df, action_proportion_df, sideway_movements_df, visited_tiles_df = basic_statistics(subjIds, web=web)
    df = pd.merge(df, scores_df, on="subj_id", how="inner")
    df = pd.merge(df, episode_result_df, on="subj_id", how="inner")
    df = pd.merge(df, penalty_df, on="subj_id", how="inner")
    df = pd.merge(df, action_proportion_df, on="subj_id", how="inner")
    df = pd.merge(df, sideway_movements_df, on="subj_id", how="inner")
    df = pd.merge(df, visited_tiles_df, on="subj_id", how="inner")
    crosswalk_df = crosswalk_statistics(subjIds, web)
    df = pd.merge(df, crosswalk_df, on="subj_id", how="inner")
    specific_condition_df = specific_condition_statistics(subjIds, web)
    df = pd.merge(df, specific_condition_df, on="subj_id", how="inner")
    return df

def basic_statistics(subjIds, web):
    scores_dfs = []
    episode_result_dfs = []
    penalty_dfs = []
    action_proportion_dfs = []
    sideway_movements_dfs = []
    visited_tiles_dfs = []
    for subjId in subjIds:
        scores, episode_results, penalties, action_proportions, sideway_movements, visited_tiles = basic_statistics_ind(subjId, web=web)
        scores_dfs.append(pd.DataFrame(scores))
        episode_result_dfs.append(pd.DataFrame(episode_results))
        penalty_dfs.append(pd.DataFrame(penalties))
        action_proportion_dfs.append(pd.DataFrame(action_proportions))
        sideway_movements_dfs.append(pd.DataFrame(sideway_movements))
        visited_tiles_dfs.append(pd.DataFrame(visited_tiles))
    scores_df = pd.concat(scores_dfs, ignore_index=True)
    episode_result_df = pd.concat(episode_result_dfs, ignore_index=True)
    penalty_df = pd.concat(penalty_dfs, ignore_index=True)
    action_proportion_df = pd.concat(action_proportion_dfs, ignore_index=True)
    sideway_movements_df = pd.concat(sideway_movements_dfs, ignore_index=True)
    visited_tiles_df = pd.concat(visited_tiles_dfs, ignore_index=True)
    return scores_df, episode_result_df, penalty_df, action_proportion_df, sideway_movements_df, visited_tiles_df

def basic_statistics_ind(subjId, web):
    subj_data_dir = data_dir(subjId, web=web)
    episodes = []
    for e in get_sorted_episodes(subj_data_dir):
        ep = int(e.strip())
        if ep < 1000: continue # filter practice session
        episodes.append(ep)   

    episode_scores_dict = {
        "full": [],
        "1": [],
        "2": [],
    }
    # bonus_score_episode_cnt = 0
    bonus_score_episode_cnt_dict = {
        "full": 0,
        "1": 0,
        "2": 0,
    }
    # bonus_score_sum = 0
    bonus_score_sum_dict = {
        "full": 0,
        "1": 0,
        "2": 0,
    }
    # run_over_episode_cnt = 0
    run_over_episode_cnt_dict = {
        "full": 0,
        "1": 0,
        "2": 0,
    }
    # penalty_cnt_dict = {-100: 0, -500: 0, -1000: 0}
    penalty_cnt_dict_dict = {
        "full": {-100: 0, -500: 0, -1000: 0},
        "1": {-100: 0, -500: 0, -1000: 0},
        "2": {-100: 0, -500: 0, -1000: 0},
    }
    # actions_list = []
    actions_list_dict = {
        "full": [],
        "1": [],
        "2": [],
    }
    # obs_list = []
    obs_list_dict = {
        "full": [],
        "1": [],
        "2": [],
    }
    # movement_list = []
    movement_list_dict = {
        "full": [],
        "1": [],
        "2": [],
    }
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode, web=web)
        episode_reward = int(sum(full_log["rewards"]))
        episode_scores_dict["full"].append(episode_reward)
        # Practice: 0001, 0002, ... (or 1, 2, 3, ...)
        # Session 1: 1001, 1002, ...
        # Session 2: 2001, 2002, ...
        episode_id = str(int(episode))
        if len(episode_id) < 4: continue
        session_id = episode_id[0]
        episode_scores_dict[session_id].append(episode_reward)

        game_end_extra_score = full_log["play_infos"][-1]["game_info"]["game_end_extra_score"]
        if game_end_extra_score > 0:
            bonus_score_episode_cnt_dict["full"] += 1
            bonus_score_episode_cnt_dict[session_id] += 1

            bonus_score_sum_dict["full"] += game_end_extra_score
            bonus_score_sum_dict[session_id] += game_end_extra_score
        elif game_end_extra_score < 0:
            run_over_episode_cnt_dict["full"] += 1
            run_over_episode_cnt_dict[session_id] += 1
            penalty_cnt_dict_dict["full"][game_end_extra_score] += 1
            penalty_cnt_dict_dict[session_id][game_end_extra_score] += 1
        
        actions_list_dict["full"] += list(full_log["actions"])
        actions_list_dict[session_id] += list(full_log["actions"])

        obs_list_dict["full"] += list(full_log["observations"][:len(full_log["observations"])-1])
        obs_list_dict[session_id] += list(full_log["observations"][:len(full_log["observations"])-1])

        movement = []
        for play_info in full_log["play_infos"]:
            cur_pos = (play_info["agent"]["grid_x"], play_info["agent"]["grid_y"])
            if len(movement) > 0 and movement[-1] == cur_pos: continue
            movement.append(cur_pos)
        movement_list_dict["full"].append(movement)
        movement_list_dict[session_id].append(movement)

    if len(episode_scores_dict["2"]) == 0:
        raise Exception(f"Session 2 not finished for {subjId}")
    if bonus_score_episode_cnt_dict["full"] == 0:
        print(f"Never cleared the game (bonus_score_episode_cnt == 0): {subjId}!!!!!!!!")
    #     raise Exception(f"Never cleared the game (bonus_score_episode_cnt == 0): {subjId}")

    # NOTE: re-add last play_info if needed for analysis
    assert len(actions_list_dict["full"]) == len(obs_list_dict["full"])
    assert len(actions_list_dict["1"]) == len(obs_list_dict["1"])
    assert len(actions_list_dict["2"]) == len(obs_list_dict["2"])
    scores = {
        "subj_id": [subjId],
        "total_score_full": [sum(episode_scores_dict["full"])],
        "total_score_session1": [sum(episode_scores_dict["1"])],
        "total_score_session2": [sum(episode_scores_dict["2"])],

        "best_score_full": [max(episode_scores_dict["full"])],
        "best_score_session1": [max(episode_scores_dict["1"])],
        "best_score_session2": [max(episode_scores_dict["2"])],

        "average_score_full": [statistics.mean(episode_scores_dict["full"])],
        "std_score_full": [statistics.stdev(episode_scores_dict["full"])],
        "average_score_session1": [statistics.mean(episode_scores_dict["1"])],
        "average_score_session2": [statistics.mean(episode_scores_dict["2"])],
    }
    episode_results = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        total_episode_cnt = len(episode_scores_dict[session])
        bonus_score_episode_ratio = bonus_score_episode_cnt_dict[session] / total_episode_cnt
        timeover_episode_cnt = total_episode_cnt - (bonus_score_episode_cnt_dict[session] + run_over_episode_cnt_dict[session])
        assert timeover_episode_cnt >= 0
        timeover_episode_ratio = timeover_episode_cnt / total_episode_cnt
        run_over_episode_ratio = run_over_episode_cnt_dict[session] / total_episode_cnt
        suffix = "" if session == "full" else f"_session{session}"

        episode_results[f"total_episode_cnt{suffix}"] = [total_episode_cnt]
        episode_results[f"bonus_score_episode_cnt{suffix}"] = [bonus_score_episode_cnt_dict[session]]
        episode_results[f"bonus_score_episode_ratio{suffix}"] = [bonus_score_episode_ratio]
        episode_results[f"bonus_score_sum{suffix}"] = [bonus_score_sum_dict[session]]
        episode_results[f"bonus_score_average{suffix}"] = [bonus_score_sum_dict[session] / bonus_score_episode_cnt_dict[session] if bonus_score_episode_cnt_dict[session] > 0 else 0]
        episode_results[f"bonus_score_average_full{suffix}"] = [bonus_score_sum_dict[session] / total_episode_cnt]
        episode_results[f"timeover_episode_cnt{suffix}"] = [timeover_episode_cnt]
        episode_results[f"timeover_episode_ratio{suffix}"] = [timeover_episode_ratio]
        episode_results[f"run_over_episode_cnt{suffix}"] = [run_over_episode_cnt_dict[session]]
        episode_results[f"run_over_episode_ratio{suffix}"] = [run_over_episode_ratio]

    penalties = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        low_penalty_cnt = penalty_cnt_dict_dict[session][-100]
        low_penalty_ratio = low_penalty_cnt / run_over_episode_cnt_dict[session] if run_over_episode_cnt_dict[session] > 0 else 0
        low_penalty_full_ratio = low_penalty_cnt / total_episode_cnt
        mid_penalty_cnt = penalty_cnt_dict_dict[session][-500]
        mid_penalty_ratio = mid_penalty_cnt / run_over_episode_cnt_dict[session] if run_over_episode_cnt_dict[session] > 0 else 0
        mid_penalty_full_ratio = mid_penalty_cnt / total_episode_cnt
        high_penalty_cnt = penalty_cnt_dict_dict[session][-1000]
        high_penalty_ratio = high_penalty_cnt / run_over_episode_cnt_dict[session] if run_over_episode_cnt_dict[session] > 0 else 0
        high_penalty_full_ratio = high_penalty_cnt / total_episode_cnt
        suffix = "" if session == "full" else f"_session{session}"

        penalties[f"low_penalty_cnt{suffix}"] = [low_penalty_cnt]
        penalties[f"low_penalty_ratio{suffix}"] = [low_penalty_ratio]
        penalties[f"low_penalty_full_ratio{suffix}"] = [low_penalty_full_ratio]
        penalties[f"mid_penalty_cnt{suffix}"] = [mid_penalty_cnt]
        penalties[f"mid_penalty_ratio{suffix}"] = [mid_penalty_ratio]
        penalties[f"mid_penalty_full_ratio{suffix}"] = [mid_penalty_full_ratio]
        penalties[f"high_penalty_cnt{suffix}"] = [high_penalty_cnt]
        penalties[f"high_penalty_ratio{suffix}"] = [high_penalty_ratio]
        penalties[f"high_penalty_full_ratio{suffix}"] = [high_penalty_full_ratio]

    action_proportions = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        # NOTHING = 0, UP = 1, DOWN = 2, RIGHT = 3, LEFT = 4
        action_counter = Counter(actions_list_dict[session])
        total_actions = len(actions_list_dict[session])
        adjusted_action_counter = Counter({ key: (v * 4 if key != 0 else v) for key, v in action_counter.items() })
        total_adjusted_actions = sum(adjusted_action_counter.values())
        suffix = "" if session == "full" else f"_session{session}"

        action_proportions[f"action_Nothing_cnt{suffix}"] = [action_counter[0]]
        action_proportions[f"action_Nothing_ratio{suffix}"] = [action_counter[0] / total_actions]
        action_proportions[f"action_UP_cnt{suffix}"] = [action_counter[1]]
        action_proportions[f"action_UP_ratio{suffix}"] = [action_counter[1] / total_actions]
        action_proportions[f"action_DOWN_cnt{suffix}"] = [action_counter[2]]
        action_proportions[f"action_DOWN_ratio{suffix}"] = [action_counter[2] / total_actions]
        action_proportions[f"action_RIGHT_cnt{suffix}"] = [action_counter[3]]
        action_proportions[f"action_RIGHT_ratio{suffix}"] = [action_counter[3] / total_actions]
        action_proportions[f"action_LEFT_cnt{suffix}"] = [action_counter[4]]
        action_proportions[f"action_LEFT_ratio{suffix}"] = [action_counter[4] / total_actions]

        action_proportions[f"action_Nothing_cnt_cleansed{suffix}"] = [adjusted_action_counter[0]]
        action_proportions[f"action_Nothing_ratio_cleansed{suffix}"] = [adjusted_action_counter[0] / total_adjusted_actions]
        action_proportions[f"action_UP_cnt_cleansed{suffix}"] = [adjusted_action_counter[1]]
        action_proportions[f"action_UP_ratio_cleansed{suffix}"] = [adjusted_action_counter[1] / total_adjusted_actions]
        action_proportions[f"action_DOWN_cnt_cleansed{suffix}"] = [adjusted_action_counter[2]]
        action_proportions[f"action_DOWN_ratio_cleansed{suffix}"] = [adjusted_action_counter[2] / total_adjusted_actions]
        action_proportions[f"action_RIGHT_cnt_cleansed{suffix}"] = [adjusted_action_counter[3]]
        action_proportions[f"action_RIGHT_ratio_cleansed{suffix}"] = [adjusted_action_counter[3] / total_adjusted_actions]
        action_proportions[f"action_LEFT_cnt_cleansed{suffix}"] = [adjusted_action_counter[4]]
        action_proportions[f"action_LEFT_ratio_cleansed{suffix}"] = [adjusted_action_counter[4] / total_adjusted_actions]

    sideway_movements = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        sideway_movement_cnt_in_safe_zone = 0.0
        total_actions = len(actions_list_dict[session])
        for i in range(total_actions):
            action = actions_list_dict[session][i]
            in_safe_row = obs_list_dict[session][i][5] == 0
            if action == 3 or action == 4:
                if in_safe_row:
                    sideway_movement_cnt_in_safe_zone += 1
        suffix = "" if session == "full" else f"_session{session}"
        sideway_movements[f"sideway_movement_ratio{suffix}"] = [r + l for r, l in zip(action_proportions["action_RIGHT_ratio"], action_proportions["action_LEFT_ratio"])]
        sideway_movements[f"sideway_movement_cnt_in_safe_zone{suffix}"] = [(sideway_movement_cnt_in_safe_zone * 4) / total_adjusted_actions]

    visited_tiles = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        visited_tile_cnt_list = []
        visited_unique_tile_cnt_list = []
        for movement in movement_list_dict[session]:
            movement = movement[1:] # remove starting point
            visited_tile_cnt = len(movement)
            visited_unique_tile_cnt = len(set(movement))
            visited_tile_cnt_list.append(visited_tile_cnt)
            visited_unique_tile_cnt_list.append(visited_unique_tile_cnt)
        visited_tile_cnt_mean = statistics.mean(visited_tile_cnt_list)
        visited_unique_tile_cnt_mean = statistics.mean(visited_unique_tile_cnt_list)
        visited_unique_tile_ratio = sum(visited_unique_tile_cnt_list) / sum(visited_tile_cnt_list)
        suffix = "" if session == "full" else f"_session{session}"

        visited_tiles[f"visited_tile_cnt_mean{suffix}"] = [visited_tile_cnt_mean]
        visited_tiles[f"visited_unique_tile_cnt_mean{suffix}"] = [visited_unique_tile_cnt_mean]
        visited_tiles[f"visited_unique_tile_ratio{suffix}"] = [visited_unique_tile_ratio]

    return scores, episode_results, penalties, action_proportions, sideway_movements, visited_tiles

def crosswalk_statistics(subjIds, web):
    dfs = []
    for subjId in subjIds:
        df = crosswalk_statistics_ind(subjId, web)
        dfs.append(pd.DataFrame(df))
    return pd.concat(dfs, ignore_index=True)

def crosswalk_statistics_ind(subjId, web):
    subj_data_dir = data_dir(subjId, web)
    episodes = get_sorted_episodes(subj_data_dir)

    full_dict = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:
        def init_counts():
            return {
                "total": 0, "easy": 0, "hard": 0,
                "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0,
                "easy_low_penalty": 0, "easy_mid_penalty": 0, "easy_high_penalty": 0,
                "hard_low_penalty": 0, "hard_mid_penalty": 0, "hard_high_penalty": 0,
            }

        # all the target roads that the player started to cross (includes getting hit by car & time overs)
        target_road_total_counts = init_counts()
        target_road_with_crosswalk_visible_total_counts = init_counts()
        # started crossing the road after activating it
        enter_road_with_crosswalk_counts = init_counts()
        # started crossing the road without using the crosswalk in sight (includes getting hit by car)
        enter_road_without_crosswalk_counts = init_counts()
        # activated the crosswalk and reached the end of the road
        crosswalk_used_counts = init_counts()
        # never used the crosswalk that is in sight
        jaywalking_counts = init_counts()
        # was outside of the crosswalk that is in sight
        jaywalking_soft_counts = init_counts()

        # all the target roads that the player finished crossing
        finished_target_road_total_counts = init_counts()
        finished_road_after_entering_crosswalk_visible_road = init_counts()
        finished_road_after_entering_crosswalk_not_visible_road = init_counts()

        for episode in episodes:
            full_log = load_episode_full_log(subj_data_dir, episode, remove_initial_noop=True, web=web)
            episode_id = str(int(episode))
            if len(episode_id) < 4: continue
            session_id = episode_id[0]
            if (session != "full") and (str(session) != session_id): continue

            road_dict = {}
            for road in full_log["episode_metadata"]["episode_configuration"]["roads"]["roads"]:
                if "topY" in road:
                    road["top_y"] = road.pop("topY")
                if "bottomY" in road:
                    road["bottom_y"] = road.pop("bottomY")

            for road in full_log["episode_metadata"]["episode_configuration"]["roads"]["roads"]:
                road_size = road["bottom_y"] - road["top_y"] + 1
                assert road_size == 2 or road_size == 4

                risk = "high_penalty"
                if road["penalty"] == 100:
                    risk = "low_penalty"
                elif road["penalty"] == 500:
                    risk = "mid_penalty"

                road_dict[road["uid"]] = {
                    "difficulty": "easy" if road_size == 2 else "hard",
                    "risk": risk, 
                    "crosswalk": road["crosswalk"], # {"uid": road.crosswalk.uid, "col": road.crosswalk.col}
                }

            encountered_road_uid_set = set()
            entered_road_uid_set = set()
            crossed_road_uid_set = set()
            last_crossing_road_info = None
            for i in range(len(full_log["actions"])):
                play_info = full_log["play_infos"][i]
                cur_pos = (play_info["agent"]["grid_x"], play_info["agent"]["grid_y"])
                action = full_log["actions"][i]
                is_up = action == 1
                crosswalk_activated = play_info["map"]["activated_crosswalk_uid"] != 0

                target_road_uid = play_info["map"]["target_road_uid"]
                road_difficulty = road_dict[target_road_uid]["difficulty"]
                road_risk = road_dict[target_road_uid]["risk"]
                keys = ["total", road_difficulty, road_risk, f"{road_difficulty}_{road_risk}"]

                is_danger_zone = full_log["episode_metadata"]["episode_configuration"]["roads"]["is_dangerous_row"]
                cur_safe_zone = not is_danger_zone[cur_pos[1]]
                next_safe_zone = not is_danger_zone[cur_pos[1]-1]
                # Entered the road
                if is_up and cur_safe_zone:
                    # left the safe zone and just started crossing the target road
                    crosswalk_visible = play_info["map"]["target_road_crosswalk_visible"]

                    # only add the first time entering the target road
                    if target_road_uid not in encountered_road_uid_set:
                        encountered_road_uid_set.add(target_road_uid)
                        for key in keys:
                            if crosswalk_visible:
                                target_road_with_crosswalk_visible_total_counts[key] += 1

                    if target_road_uid not in entered_road_uid_set:
                        entered_road_uid_set.add(target_road_uid)
                        for key in keys:
                            target_road_total_counts[key] += 1
                            if crosswalk_activated:
                                enter_road_with_crosswalk_counts[key] += 1
                            else:
                                if crosswalk_visible:
                                    enter_road_without_crosswalk_counts[key] += 1
                    # restart saving the trajectory every time the player enters the road until the first successful crossing
                    if target_road_uid not in crossed_road_uid_set:
                        last_crossing_road_info = {
                            "uid": target_road_uid,
                            "trajectory": [cur_pos],
                            "crosswalk_sometimes_visible": crosswalk_visible,
                            "crosswalk_always_activated": crosswalk_activated,
                            "crosswalk_never_activated": not crosswalk_activated,
                            "crosswalk_sometimes_activated": crosswalk_activated,
                            "entered_road_with_crosswalk_visible": crosswalk_visible,
                        }

                if last_crossing_road_info is not None:
                    # Left the road
                    if target_road_uid == last_crossing_road_info["uid"]:
                        if last_crossing_road_info["trajectory"][-1] != cur_pos:
                            last_crossing_road_info["trajectory"].append(cur_pos)
                            crosswalk_visible = play_info["map"]["target_road_crosswalk_visible"]
                            if crosswalk_visible:
                                last_crossing_road_info["crosswalk_sometimes_visible"] = True
                            if not crosswalk_activated:
                                last_crossing_road_info["crosswalk_always_activated"] = False
                            else:
                                last_crossing_road_info["crosswalk_never_activated"] = False
                                last_crossing_road_info["crosswalk_sometimes_activated"] = True

                        finished_crossing = is_up and next_safe_zone # finished crossing the road and entered the safe zone for the first time
                        # run_over = full_log["play_infos"][i+1]["game_info"]["game_end_extra_score"] < 0 # include getting hit by car
                        if finished_crossing:
                            if target_road_uid not in crossed_road_uid_set:
                                crossed_road_uid_set.add(target_road_uid)
                                for key in keys:
                                    finished_target_road_total_counts[key] += 1
                                    entered_road_with_crosswalk_visible = last_crossing_road_info["entered_road_with_crosswalk_visible"]
                                    if entered_road_with_crosswalk_visible:
                                        finished_road_after_entering_crosswalk_visible_road[key] += 1
                                    else:
                                        finished_road_after_entering_crosswalk_not_visible_road[key] += 1

                                    crosswalk_always_activated = last_crossing_road_info["crosswalk_always_activated"]
                                    crosswalk_sometimes_visible = last_crossing_road_info["crosswalk_sometimes_visible"]
                                    if crosswalk_always_activated:
                                        crosswalk_used_counts[key] += 1
                                    if crosswalk_sometimes_visible and last_crossing_road_info["crosswalk_never_activated"]:
                                        jaywalking_counts[key] += 1
                                    if crosswalk_sometimes_visible and not crosswalk_always_activated:
                                        jaywalking_soft_counts[key] += 1
                        # elif run_over:
                    else:
                        # went back to previously crossed road. should be ignored from the statistics.
                        last_crossing_road_info = None

        for dictionary in [target_road_total_counts, target_road_with_crosswalk_visible_total_counts, enter_road_with_crosswalk_counts, enter_road_without_crosswalk_counts, crosswalk_used_counts, jaywalking_counts]:
            assert dictionary["total"] == dictionary["easy"] + dictionary["hard"]
            assert dictionary["total"] == dictionary["low_penalty"] + dictionary["mid_penalty"] + dictionary["high_penalty"]
            assert dictionary["total"] == dictionary["easy_low_penalty"] + dictionary["easy_mid_penalty"] + dictionary["easy_high_penalty"] + \
                dictionary["hard_low_penalty"] + dictionary["hard_mid_penalty"] + dictionary["hard_high_penalty"]

        def ratio(dictionary, col):
            if target_road_total_counts[col] == 0: return 0
            return dictionary[col] / target_road_total_counts[col]

        def crosswalk_visible_road_ratio(dictionary, col):
            if target_road_with_crosswalk_visible_total_counts[col] == 0: return 0
            return dictionary[col] / target_road_with_crosswalk_visible_total_counts[col]
        
        suffix = "" if session == "full" else f"_session{session}"

        full_dict[f"total_target_roads_total{suffix}"] = target_road_total_counts["total"]
        full_dict[f"total_target_roads_easy{suffix}"] = target_road_total_counts["easy"]
        full_dict[f"total_target_roads_hard{suffix}"] = target_road_total_counts["hard"]
        full_dict[f"total_target_roads_low_penalty{suffix}"] = target_road_total_counts["low_penalty"]
        full_dict[f"total_target_roads_mid_penalty{suffix}"] = target_road_total_counts["mid_penalty"]
        full_dict[f"total_target_roads_high_penalty{suffix}"] = target_road_total_counts["high_penalty"]

        full_dict[f"total_target_roads_with_crosswalk_visible_total{suffix}"] = target_road_with_crosswalk_visible_total_counts["total"]
        full_dict[f"total_target_roads_with_crosswalk_visible_easy{suffix}"] = target_road_with_crosswalk_visible_total_counts["easy"]
        full_dict[f"total_target_roads_with_crosswalk_visible_hard{suffix}"] = target_road_with_crosswalk_visible_total_counts["hard"]
        full_dict[f"total_target_roads_with_crosswalk_visible_low_penalty{suffix}"] = target_road_with_crosswalk_visible_total_counts["low_penalty"]
        full_dict[f"total_target_roads_with_crosswalk_visible_mid_penalty{suffix}"] = target_road_with_crosswalk_visible_total_counts["mid_penalty"]
        full_dict[f"total_target_roads_with_crosswalk_visible_high_penalty{suffix}"] = target_road_with_crosswalk_visible_total_counts["high_penalty"]

        full_dict[f"road_crossing_success_ratio_total{suffix}"] = ratio(finished_target_road_total_counts, "total")
        full_dict[f"road_crossing_success_ratio_easy{suffix}"] = ratio(finished_target_road_total_counts, "easy")
        full_dict[f"road_crossing_success_ratio_hard{suffix}"] = ratio(finished_target_road_total_counts, "hard")
        full_dict[f"road_crossing_success_ratio_low_penalty{suffix}"] = ratio(finished_target_road_total_counts, "low_penalty")
        full_dict[f"road_crossing_success_ratio_mid_penalty{suffix}"] = ratio(finished_target_road_total_counts, "mid_penalty")
        full_dict[f"road_crossing_success_ratio_high_penalty{suffix}"] = ratio(finished_target_road_total_counts, "high_penalty")
        full_dict[f"road_crossing_success_ratio_easy_low_penalty{suffix}"] = ratio(finished_target_road_total_counts, "easy_low_penalty")
        full_dict[f"road_crossing_success_ratio_easy_mid_penalty{suffix}"] = ratio(finished_target_road_total_counts, "easy_mid_penalty")
        full_dict[f"road_crossing_success_ratio_easy_high_penalty{suffix}"] = ratio(finished_target_road_total_counts, "easy_high_penalty")
        full_dict[f"road_crossing_success_ratio_hard_low_penalty{suffix}"] = ratio(finished_target_road_total_counts, "hard_low_penalty")
        full_dict[f"road_crossing_success_ratio_hard_mid_penalty{suffix}"] = ratio(finished_target_road_total_counts, "hard_mid_penalty")
        full_dict[f"road_crossing_success_ratio_hard_high_penalty{suffix}"] = ratio(finished_target_road_total_counts, "hard_high_penalty")

        full_dict[f"crosswalk_visible_road_crossing_success_ratio_total{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "total")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_easy{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "easy")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_hard{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "hard")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "low_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "mid_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "high_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_easy_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "easy_low_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_easy_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "easy_mid_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_easy_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "easy_high_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_hard_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "hard_low_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_hard_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "hard_mid_penalty")
        full_dict[f"crosswalk_visible_road_crossing_success_ratio_hard_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_visible_road, "hard_high_penalty")

        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_total{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "total")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_easy{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "easy")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_hard{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "hard")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "low_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "mid_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "high_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_easy_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "easy_low_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_easy_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "easy_mid_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_easy_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "easy_high_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_hard_low_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "hard_low_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_hard_mid_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "hard_mid_penalty")
        full_dict[f"crosswalk_not_visible_road_crossing_success_ratio_hard_high_penalty{suffix}"] = ratio(finished_road_after_entering_crosswalk_not_visible_road, "hard_high_penalty")

            ##########
        full_dict[f"entered_road_with_crosswalk_total{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "total")
        full_dict[f"entered_road_with_crosswalk_easy{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "easy")
        full_dict[f"entered_road_with_crosswalk_hard{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "hard")
        full_dict[f"entered_road_with_crosswalk_low_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "low_penalty")
        full_dict[f"entered_road_with_crosswalk_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "mid_penalty")
        full_dict[f"entered_road_with_crosswalk_high_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_with_crosswalk_counts, "high_penalty")

        full_dict[f"entered_road_with_crosswalk_cnt_total{suffix}"] = enter_road_with_crosswalk_counts["total"]
        full_dict[f"entered_road_with_crosswalk_cnt_easy{suffix}"] = enter_road_with_crosswalk_counts["easy"]
        full_dict[f"entered_road_with_crosswalk_cnt_hard{suffix}"] = enter_road_with_crosswalk_counts["hard"]
        full_dict[f"entered_road_with_crosswalk_cnt_low_penalty{suffix}"] = enter_road_with_crosswalk_counts["low_penalty"]
        full_dict[f"entered_road_with_crosswalk_cnt_mid_penalty{suffix}"] = enter_road_with_crosswalk_counts["mid_penalty"]
        full_dict[f"entered_road_with_crosswalk_cnt_high_penalty{suffix}"] = enter_road_with_crosswalk_counts["high_penalty"]

            ##########
        full_dict[f"entered_road_ignoring_crosswalk_total{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "total")
        full_dict[f"entered_road_ignoring_crosswalk_easy{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "easy")
        full_dict[f"entered_road_ignoring_crosswalk_hard{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "hard")
        full_dict[f"entered_road_ignoring_crosswalk_low_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "low_penalty")
        full_dict[f"entered_road_ignoring_crosswalk_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "mid_penalty")
        full_dict[f"entered_road_ignoring_crosswalk_high_penalty{suffix}"] = crosswalk_visible_road_ratio(enter_road_without_crosswalk_counts, "high_penalty")

        full_dict[f"entered_road_ignoring_crosswalk_cnt_total{suffix}"] = enter_road_without_crosswalk_counts["total"]
        full_dict[f"entered_road_ignoring_crosswalk_cnt_easy{suffix}"] = enter_road_without_crosswalk_counts["easy"]
        full_dict[f"entered_road_ignoring_crosswalk_cnt_hard{suffix}"] = enter_road_without_crosswalk_counts["hard"]
        full_dict[f"entered_road_ignoring_crosswalk_cnt_low_penalty{suffix}"] = enter_road_without_crosswalk_counts["low_penalty"]
        full_dict[f"entered_road_ignoring_crosswalk_cnt_mid_penalty{suffix}"] = enter_road_without_crosswalk_counts["mid_penalty"]
        full_dict[f"entered_road_ignoring_crosswalk_cnt_high_penalty{suffix}"] = enter_road_without_crosswalk_counts["high_penalty"]

        ##########
        full_dict[f"crosswalk_used_ratio_total{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "total")
        full_dict[f"crosswalk_used_ratio_easy{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "easy")
        full_dict[f"crosswalk_used_ratio_hard{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "hard")
        full_dict[f"crosswalk_used_ratio_low_penalty{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "low_penalty")
        full_dict[f"crosswalk_used_ratio_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "mid_penalty")
        full_dict[f"crosswalk_used_ratio_high_penalty{suffix}"] = crosswalk_visible_road_ratio(crosswalk_used_counts, "high_penalty")

        full_dict[f"crosswalk_used_cnt_total{suffix}"] = crosswalk_used_counts["total"]
        full_dict[f"crosswalk_used_cnt_easy{suffix}"] = crosswalk_used_counts["easy"]
        full_dict[f"crosswalk_used_cnt_hard{suffix}"] = crosswalk_used_counts["hard"]
        full_dict[f"crosswalk_used_cnt_low_penalty{suffix}"] = crosswalk_used_counts["low_penalty"]
        full_dict[f"crosswalk_used_cnt_mid_penalty{suffix}"] = crosswalk_used_counts["mid_penalty"]
        full_dict[f"crosswalk_used_cnt_high_penalty{suffix}"] = crosswalk_used_counts["high_penalty"]

        ##########
        full_dict[f"jaywalking_ratio_total{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "total")
        full_dict[f"jaywalking_ratio_easy{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "easy")
        full_dict[f"jaywalking_ratio_hard{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "hard")
        full_dict[f"jaywalking_ratio_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "low_penalty")
        full_dict[f"jaywalking_ratio_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "mid_penalty")
        full_dict[f"jaywalking_ratio_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "high_penalty")
        full_dict[f"jaywalking_ratio_easy_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "easy_low_penalty")
        full_dict[f"jaywalking_ratio_easy_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "easy_mid_penalty")
        full_dict[f"jaywalking_ratio_easy_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "easy_high_penalty")
        full_dict[f"jaywalking_ratio_hard_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "hard_low_penalty")
        full_dict[f"jaywalking_ratio_hard_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "hard_mid_penalty")
        full_dict[f"jaywalking_ratio_hard_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_counts, "hard_high_penalty")

        full_dict[f"jaywalking_cnt_total{suffix}"] = jaywalking_counts["total"]
        full_dict[f"jaywalking_cnt_easy{suffix}"] = jaywalking_counts["easy"]
        full_dict[f"jaywalking_cnt_hard{suffix}"] = jaywalking_counts["hard"]
        full_dict[f"jaywalking_cnt_low_penalty{suffix}"] = jaywalking_counts["low_penalty"]
        full_dict[f"jaywalking_cnt_mid_penalty{suffix}"] = jaywalking_counts["mid_penalty"]
        full_dict[f"jaywalking_cnt_high_penalty{suffix}"] = jaywalking_counts["high_penalty"]
        full_dict[f"jaywalking_cnt_easy_low_penalty{suffix}"] = jaywalking_counts["easy_low_penalty"]
        full_dict[f"jaywalking_cnt_easy_mid_penalty{suffix}"] = jaywalking_counts["easy_mid_penalty"]
        full_dict[f"jaywalking_cnt_easy_high_penalty{suffix}"] = jaywalking_counts["easy_high_penalty"]
        full_dict[f"jaywalking_cnt_hard_low_penalty{suffix}"] = jaywalking_counts["hard_low_penalty"]
        full_dict[f"jaywalking_cnt_hard_mid_penalty{suffix}"] = jaywalking_counts["hard_mid_penalty"]
        full_dict[f"jaywalking_cnt_hard_high_penalty{suffix}"] = jaywalking_counts["hard_high_penalty"]

        ##########
        full_dict[f"jaywalking_soft_ratio_total{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "total")
        full_dict[f"jaywalking_soft_ratio_easy{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "easy")
        full_dict[f"jaywalking_soft_ratio_hard{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "hard")
        full_dict[f"jaywalking_soft_ratio_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "low_penalty")
        full_dict[f"jaywalking_soft_ratio_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "mid_penalty")
        full_dict[f"jaywalking_soft_ratio_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "high_penalty")
        full_dict[f"jaywalking_soft_ratio_easy_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "easy_low_penalty")
        full_dict[f"jaywalking_soft_ratio_easy_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "easy_mid_penalty")
        full_dict[f"jaywalking_soft_ratio_easy_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "easy_high_penalty")
        full_dict[f"jaywalking_soft_ratio_hard_low_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "hard_low_penalty")
        full_dict[f"jaywalking_soft_ratio_hard_mid_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "hard_mid_penalty")
        full_dict[f"jaywalking_soft_ratio_hard_high_penalty{suffix}"] = crosswalk_visible_road_ratio(jaywalking_soft_counts, "hard_high_penalty")

        full_dict[f"jaywalking_soft_cnt_total{suffix}"] = jaywalking_soft_counts["total"]
        full_dict[f"jaywalking_soft_cnt_easy{suffix}"] = jaywalking_soft_counts["easy"]
        full_dict[f"jaywalking_soft_cnt_hard{suffix}"] = jaywalking_soft_counts["hard"]
        full_dict[f"jaywalking_soft_cnt_low_penalty{suffix}"] = jaywalking_soft_counts["low_penalty"]
        full_dict[f"jaywalking_soft_cnt_mid_penalty{suffix}"] = jaywalking_soft_counts["mid_penalty"]
        full_dict[f"jaywalking_soft_cnt_high_penalty{suffix}"] = jaywalking_soft_counts["high_penalty"]
        full_dict[f"jaywalking_soft_cnt_easy_low_penalty{suffix}"] = jaywalking_soft_counts["easy_low_penalty"]
        full_dict[f"jaywalking_soft_cnt_easy_mid_penalty{suffix}"] = jaywalking_soft_counts["easy_mid_penalty"]
        full_dict[f"jaywalking_soft_cnt_easy_high_penalty{suffix}"] = jaywalking_soft_counts["easy_high_penalty"]
        full_dict[f"jaywalking_soft_cnt_hard_low_penalty{suffix}"] = jaywalking_soft_counts["hard_low_penalty"]
        full_dict[f"jaywalking_soft_cnt_hard_mid_penalty{suffix}"] = jaywalking_soft_counts["hard_mid_penalty"]
        full_dict[f"jaywalking_soft_cnt_hard_high_penalty{suffix}"] = jaywalking_soft_counts["hard_high_penalty"]
    return full_dict

def specific_condition_statistics(subjIds, web):
    dfs = []
    for subjId in subjIds:
        df = specific_condition_statistics_ind(subjId, web)
        dfs.append(pd.DataFrame(df))
    return pd.concat(dfs, ignore_index=True)

def specific_condition_statistics_ind(subjId, web):
    subj_data_dir = data_dir(subjId, web)
    episodes = get_sorted_episodes(subj_data_dir)

    full_dict = {
        "subj_id": [subjId],
    }
    for session in ["full", "1", "2"]:

        def init_counts():
            return {
                "total": 0, "easy": 0, "hard": 0,
                "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0,
                "easy_low_penalty": 0, "easy_mid_penalty": 0, "easy_high_penalty": 0,
                "hard_low_penalty": 0, "hard_mid_penalty": 0, "hard_high_penalty": 0,
            }
        
        # all the target roads that the player started to cross (includes getting hit by car & time overs)
        target_road_total_counts = init_counts()
        # move back to safe zone after reaching the danger zone (allow multiple go backs for the same road)
        move_back_to_safe_zone_counts = init_counts()
        # move up even though car is still in front of them (include both run-overs and survived cases)
        move_up_at_car_counts = init_counts()
        # move up to evade approaching same lane car
        move_up_to_evade_same_lane_car_counts = init_counts()
        # move down to evade approaching same lane car
        move_down_to_evade_same_lane_car_counts = init_counts()
        # move left or right to evade approaching same lane car
        move_sideways_to_evade_same_lane_car_counts = init_counts()
        # move up to next lane with approaching car visible
        move_up_to_lane_with_approching_car_counts = init_counts()
        # move sideways in safe zone
        sideway_movements_in_safe_zone_counts = init_counts()
        # move sideways in danger zone
        sideway_movements_in_danger_zone_counts = init_counts()
        # move sideways in safe zone when crosswalk is not visible
        sideway_movements_in_crosswalk_not_visible_safe_zone_counts = init_counts()
        # move toward crosswalks
        move_toward_crosswalk_counts = init_counts()
        move_toward_crosswalk_safe_zone_counts = init_counts()
        move_toward_crosswalk_danger_zone_counts = init_counts()

        actions_list = []
        obs_list = []
        road_info_dicts = []
        crosswalk_activated_list = []
        crosswalk_visible_list = []
        for episode in episodes:
            full_log = load_episode_full_log(subj_data_dir, episode, web=web)
            # episode_reward = int(sum(full_log["rewards"]))

            # Practice: 0001, 0002, ... (or 1, 2, 3, ...)
            # Session 1: 1001, 1002, ...
            # Session 2: 2001, 2002, ...
            episode_id = str(int(episode))
            if len(episode_id) < 4: continue
            session_id = episode_id[0]
            if (session != "full") and (str(session) != session_id): continue

            road_dict = {}
            for road in full_log["episode_metadata"]["episode_configuration"]["roads"]["roads"]:
                if "topY" in road:
                    road["top_y"] = road.pop("topY")
                if "bottomY" in road:
                    road["bottom_y"] = road.pop("bottomY")

            for road in full_log["episode_metadata"]["episode_configuration"]["roads"]["roads"]:
                road_size = road["bottom_y"] - road["top_y"] + 1
                assert road_size == 2 or road_size == 4
                risk = "high_penalty"
                if road["penalty"] == 100:
                    risk = "low_penalty"
                elif road["penalty"] == 500:
                    risk = "mid_penalty"
                road_dict[road["uid"]] = {
                    "difficulty": "hard" if road_size == 4 else "easy",
                    "penalty": risk, 
                    "crosswalk": road["crosswalk"], # {"uid": road.crosswalk.uid, "col": road.crosswalk.col}
                }

            actions_list += list(full_log["actions"])
            obs_list += list(full_log["observations"][:len(full_log["observations"])-1])
            target_road_uid_set = set()
            for i in range(len(full_log["actions"])):
                target_road_uid = full_log["play_infos"][i]["map"]["target_road_uid"]
                road_info_dicts.append(road_dict[target_road_uid])
                if target_road_uid not in target_road_uid_set:
                    road_difficulty = road_dict[target_road_uid]["difficulty"]
                    road_risk = road_dict[target_road_uid]["penalty"]
                    for key in ["total", road_difficulty, road_risk, f"{road_difficulty}_{road_risk}"]:
                        target_road_total_counts[key] += 1
                target_road_uid_set.add(target_road_uid)

                crosswalk_activated = full_log["play_infos"][i]["map"]["activated_crosswalk_uid"] > 0
                crosswalk_activated_list.append(crosswalk_activated)

                crosswalk_visible = crosswalk_activated = full_log["play_infos"][i]["map"]["target_road_crosswalk_visible"]
                crosswalk_visible_list.append(crosswalk_visible)

        assert len(actions_list) == len(obs_list)

        for i in range(len(obs_list)):
            cur_obs = obs_list[i]
            action = actions_list[i]
            road_info_dict = road_info_dicts[i]
            road_difficulty = road_info_dict["difficulty"]
            road_risk = road_info_dict["penalty"]
            keys = ["total", road_difficulty, road_risk, f"{road_difficulty}_{road_risk}"]

            is_cur_safe_tile = cur_obs[5] == 0
            is_cur_danger_tile = cur_obs[5] == 1
            is_down_safe_tile = cur_obs[7] == 0
            is_car_in_up_tile = cur_obs[34] == 1
            is_car_in_down_tile = cur_obs[38] == 1
            is_crosswalk_activated = crosswalk_activated_list[i]

            car_in_same_lane_left_close = cur_obs[89] >= 0.7
            car_in_same_lane_right_close = cur_obs[93] >= 0.7
            car_in_next_lane_left = cur_obs[77] >= 0.1
            car_in_next_lane_right = cur_obs[81] >= 0.1

            crosswalk_to_the_right = cur_obs[25] > 0 and (not is_crosswalk_activated)
            crosswalk_to_the_left = cur_obs[26] > 0 and (not is_crosswalk_activated)

            is_up = action == 1
            is_down = action == 2
            is_sideways = action >= 3
            is_right = action == 3
            is_left = action == 4

            if (crosswalk_to_the_right and is_right) or (crosswalk_to_the_left and is_left):
                for key in keys:
                    move_toward_crosswalk_counts[key] += 1
                    if is_cur_safe_tile:
                        move_toward_crosswalk_safe_zone_counts[key] += 1
                    else:
                        move_toward_crosswalk_danger_zone_counts[key] += 1

            if is_cur_safe_tile and is_sideways:
                if not crosswalk_visible_list[i]:
                    for key in keys:
                        sideway_movements_in_crosswalk_not_visible_safe_zone_counts[key] += 1
                for key in keys:
                    sideway_movements_in_safe_zone_counts[key] += 1
            if is_cur_danger_tile and is_sideways:
                for key in keys:
                    sideway_movements_in_danger_zone_counts[key] += 1

            if is_cur_danger_tile and is_down_safe_tile and is_down:
                for key in keys:
                    move_back_to_safe_zone_counts[key] += 1
            if is_car_in_up_tile and is_up:
                for key in keys:
                    move_up_at_car_counts[key] += 1
            if not is_crosswalk_activated and (car_in_same_lane_left_close or car_in_same_lane_right_close):
                if is_up:
                    for key in keys:
                        move_up_to_evade_same_lane_car_counts[key] += 1
                if is_down:
                    for key in keys:
                        move_down_to_evade_same_lane_car_counts[key] += 1
                if is_sideways:
                    for key in keys:
                        move_sideways_to_evade_same_lane_car_counts[key] += 1
            if not is_crosswalk_activated and (car_in_next_lane_left or car_in_next_lane_right):
                if is_up:
                    for key in keys:
                        move_up_to_lane_with_approching_car_counts[key] += 1

        for dictionary in [
            target_road_total_counts, move_back_to_safe_zone_counts, move_up_at_car_counts, 
            move_up_to_evade_same_lane_car_counts, move_down_to_evade_same_lane_car_counts,
            sideway_movements_in_safe_zone_counts, sideway_movements_in_danger_zone_counts, sideway_movements_in_crosswalk_not_visible_safe_zone_counts,
            move_toward_crosswalk_counts, move_toward_crosswalk_safe_zone_counts, move_toward_crosswalk_danger_zone_counts,
        ]:
            assert dictionary["total"] == dictionary["easy"] + dictionary["hard"]
            assert dictionary["total"] == dictionary["low_penalty"] + dictionary["mid_penalty"] + dictionary["high_penalty"]
            assert dictionary["total"] == dictionary["easy_low_penalty"] + dictionary["easy_mid_penalty"] + dictionary["easy_high_penalty"] + \
                dictionary["hard_low_penalty"] + dictionary["hard_mid_penalty"] + dictionary["hard_high_penalty"]

        def ratio(dictionary, col):
            if target_road_total_counts[col] == 0: return 0
            return dictionary[col] / target_road_total_counts[col]

        def move_up_to_evade_approaching_cars_ratio(col):
            if (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col]) == 0:
                return 0
            return move_up_to_evade_same_lane_car_counts[col] / (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col])

        def move_down_to_evade_approaching_cars_ratio(col):
            if (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col]) == 0:
                return 0
            return move_down_to_evade_same_lane_car_counts[col] / (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col])

        def move_up_over_down_to_evade_approaching_cars_ratio(col):
            if (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col]) == 0:
                return 0
            return (move_up_to_evade_same_lane_car_counts[col] - move_down_to_evade_same_lane_car_counts[col]) / (move_up_to_evade_same_lane_car_counts[col] + move_down_to_evade_same_lane_car_counts[col] + move_sideways_to_evade_same_lane_car_counts[col])

        suffix = "" if session == "full" else f"_session{session}"

        # 도로에서 다시 안전구역으로 되돌아간 경우
        full_dict[f"move_back_to_safe_zone_cnt_total{suffix}"] = move_back_to_safe_zone_counts["total"]
        full_dict[f"move_back_to_safe_zone_cnt_easy{suffix}"] = move_back_to_safe_zone_counts["easy"]
        full_dict[f"move_back_to_safe_zone_cnt_hard{suffix}"] = move_back_to_safe_zone_counts["hard"]
        full_dict[f"move_back_to_safe_zone_cnt_low_penalty{suffix}"] = move_back_to_safe_zone_counts["low_penalty"]
        full_dict[f"move_back_to_safe_zone_cnt_mid_penalty{suffix}"] = move_back_to_safe_zone_counts["mid_penalty"]
        full_dict[f"move_back_to_safe_zone_cnt_high_penalty{suffix}"] = move_back_to_safe_zone_counts["high_penalty"]

        full_dict[f"move_back_to_safe_zone_ratio_total{suffix}"] = ratio(move_back_to_safe_zone_counts, "total")
        full_dict[f"move_back_to_safe_zone_ratio_easy{suffix}"] = ratio(move_back_to_safe_zone_counts, "easy")
        full_dict[f"move_back_to_safe_zone_ratio_hard{suffix}"] = ratio(move_back_to_safe_zone_counts, "hard")
        full_dict[f"move_back_to_safe_zone_ratio_low_penalty{suffix}"] = ratio(move_back_to_safe_zone_counts, "low_penalty")
        full_dict[f"move_back_to_safe_zone_ratio_mid_penalty{suffix}"] = ratio(move_back_to_safe_zone_counts, "mid_penalty")
        full_dict[f"move_back_to_safe_zone_ratio_high_penalty{suffix}"] = ratio(move_back_to_safe_zone_counts, "high_penalty")

        # 바로 앞칸에 아직 자동차가 남아있음에도 충분히 기다리지 못하고 앞으로 이동한 경우
        full_dict[f"move_up_at_car_cnt_total{suffix}"] = move_up_at_car_counts["total"]
        full_dict[f"move_up_at_car_cnt_easy{suffix}"] = move_up_at_car_counts["easy"]
        full_dict[f"move_up_at_car_cnt_hard{suffix}"] = move_up_at_car_counts["hard"]
        full_dict[f"move_up_at_car_cnt_low_penalty{suffix}"] = move_up_at_car_counts["low_penalty"]
        full_dict[f"move_up_at_car_cnt_mid_penalty{suffix}"] = move_up_at_car_counts["mid_penalty"]
        full_dict[f"move_up_at_car_cnt_high_penalty{suffix}"] = move_up_at_car_counts["high_penalty"]

        # 현재 칸으로 옆에서 달려오는 자동차를 피해 앞으로 회피한 횟수
        full_dict[f"move_up_to_evade_same_lane_car_cnt_total{suffix}"] = move_up_to_evade_same_lane_car_counts["total"]
        full_dict[f"move_up_to_evade_same_lane_car_cnt_easy{suffix}"] = move_up_to_evade_same_lane_car_counts["easy"]
        full_dict[f"move_up_to_evade_same_lane_car_cnt_hard{suffix}"] = move_up_to_evade_same_lane_car_counts["hard"]
        full_dict[f"move_up_to_evade_same_lane_car_cnt_low_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["low_penalty"]
        full_dict[f"move_up_to_evade_same_lane_car_cnt_mid_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["mid_penalty"]
        full_dict[f"move_up_to_evade_same_lane_car_cnt_high_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["high_penalty"]

        full_dict[f"move_up_to_evade_same_lane_car_ratio_total{suffix}"] = move_up_to_evade_approaching_cars_ratio("total")
        full_dict[f"move_up_to_evade_same_lane_car_ratio_easy{suffix}"] = move_up_to_evade_approaching_cars_ratio("easy")
        full_dict[f"move_up_to_evade_same_lane_car_ratio_hard{suffix}"] = move_up_to_evade_approaching_cars_ratio("hard")
        full_dict[f"move_up_to_evade_same_lane_car_ratio_low_penalty{suffix}"] = move_up_to_evade_approaching_cars_ratio("low_penalty")
        full_dict[f"move_up_to_evade_same_lane_car_ratio_mid_penalty{suffix}"] = move_up_to_evade_approaching_cars_ratio("mid_penalty")
        full_dict[f"move_up_to_evade_same_lane_car_ratio_high_penalty{suffix}"] = move_up_to_evade_approaching_cars_ratio("high_penalty")

        # 현재 칸으로 옆에서 달려오는 자동차를 피해 뒤로 회피한 횟수
        full_dict[f"move_down_to_evade_same_lane_car_cnt_total{suffix}"] = move_down_to_evade_same_lane_car_counts["total"]
        full_dict[f"move_down_to_evade_same_lane_car_cnt_easy{suffix}"] = move_down_to_evade_same_lane_car_counts["easy"]
        full_dict[f"move_down_to_evade_same_lane_car_cnt_hard{suffix}"] = move_down_to_evade_same_lane_car_counts["hard"]
        full_dict[f"move_down_to_evade_same_lane_car_cnt_low_penalty{suffix}"] = move_down_to_evade_same_lane_car_counts["low_penalty"]
        full_dict[f"move_down_to_evade_same_lane_car_cnt_mid_penalty{suffix}"] = move_down_to_evade_same_lane_car_counts["mid_penalty"]
        full_dict[f"move_down_to_evade_same_lane_car_cnt_high_penalty{suffix}"] = move_down_to_evade_same_lane_car_counts["high_penalty"]

        full_dict[f"move_down_to_evade_same_lane_car_ratio_total{suffix}"] = move_down_to_evade_approaching_cars_ratio("total")
        full_dict[f"move_down_to_evade_same_lane_car_ratio_easy{suffix}"] = move_down_to_evade_approaching_cars_ratio("easy")
        full_dict[f"move_down_to_evade_same_lane_car_ratio_hard{suffix}"] = move_down_to_evade_approaching_cars_ratio("hard")
        full_dict[f"move_down_to_evade_same_lane_car_ratio_low_penalty{suffix}"] = move_down_to_evade_approaching_cars_ratio("low_penalty")
        full_dict[f"move_down_to_evade_same_lane_car_ratio_mid_penalty{suffix}"] = move_down_to_evade_approaching_cars_ratio("mid_penalty")
        full_dict[f"move_down_to_evade_same_lane_car_ratio_high_penalty{suffix}"] = move_down_to_evade_approaching_cars_ratio("high_penalty")

        # 현재 칸으로 옆에서 달려오는 자동차를 피해 뒤보다는 위로 회피한 횟수
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_total{suffix}"] = move_up_to_evade_same_lane_car_counts["total"] - move_down_to_evade_same_lane_car_counts["total"]
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_easy{suffix}"] = move_up_to_evade_same_lane_car_counts["easy"] - move_down_to_evade_same_lane_car_counts["easy"]
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_hard{suffix}"] = move_up_to_evade_same_lane_car_counts["hard"] - move_down_to_evade_same_lane_car_counts["hard"]
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_low_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["low_penalty"] - move_down_to_evade_same_lane_car_counts["low_penalty"]
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_mid_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["mid_penalty"] - move_down_to_evade_same_lane_car_counts["mid_penalty"]
        full_dict[f"move_up_over_down_to_evade_same_lane_car_cnt_high_penalty{suffix}"] = move_up_to_evade_same_lane_car_counts["high_penalty"] - move_down_to_evade_same_lane_car_counts["high_penalty"]

        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_total{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("total")
        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_easy{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("easy")
        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_hard{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("hard")
        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_low_penalty{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("low_penalty")
        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_mid_penalty{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("mid_penalty")
        full_dict[f"move_up_over_down_to_evade_same_lane_car_ratio_high_penalty{suffix}"] = move_up_over_down_to_evade_approaching_cars_ratio("high_penalty")

        # 현재 칸으로 옆에서 달려오는 자동차를 피해 좌우로 회피한 횟수
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_total{suffix}"] = move_sideways_to_evade_same_lane_car_counts["total"]
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_easy{suffix}"] = move_sideways_to_evade_same_lane_car_counts["easy"]
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_hard{suffix}"] = move_sideways_to_evade_same_lane_car_counts["hard"]
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_low_penalty{suffix}"] = move_sideways_to_evade_same_lane_car_counts["low_penalty"]
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_mid_penalty{suffix}"] = move_sideways_to_evade_same_lane_car_counts["mid_penalty"]
        full_dict[f"move_sideways_to_evade_same_lane_car_cnt_high_penalty{suffix}"] = move_sideways_to_evade_same_lane_car_counts["high_penalty"]

        # 바로 앞줄에서 플레이어를 향해 달려오는 자동차가 있음에도 앞으로 이동한 횟수
        full_dict[f"move_up_to_lane_with_approching_car_cnt_total{suffix}"] = move_up_to_lane_with_approching_car_counts["total"]
        full_dict[f"move_up_to_lane_with_approching_car_cnt_easy{suffix}"] = move_up_to_lane_with_approching_car_counts["easy"]
        full_dict[f"move_up_to_lane_with_approching_car_cnt_hard{suffix}"] = move_up_to_lane_with_approching_car_counts["hard"]
        full_dict[f"move_up_to_lane_with_approching_car_cnt_low_penalty{suffix}"] = move_up_to_lane_with_approching_car_counts["low_penalty"]
        full_dict[f"move_up_to_lane_with_approching_car_cnt_mid_penalty{suffix}"] = move_up_to_lane_with_approching_car_counts["mid_penalty"]
        full_dict[f"move_up_to_lane_with_approching_car_cnt_high_penalty{suffix}"] = move_up_to_lane_with_approching_car_counts["high_penalty"]

        # 안전 구역에서 좌우 이동한 횟수
        full_dict[f"sideway_movements_in_safe_zone_cnt_total{suffix}"] = sideway_movements_in_safe_zone_counts["total"]
        full_dict[f"sideway_movements_in_safe_zone_cnt_easy{suffix}"] = sideway_movements_in_safe_zone_counts["easy"]
        full_dict[f"sideway_movements_in_safe_zone_cnt_hard{suffix}"] = sideway_movements_in_safe_zone_counts["hard"]
        full_dict[f"sideway_movements_in_safe_zone_cnt_low_penalty{suffix}"] = sideway_movements_in_safe_zone_counts["low_penalty"]
        full_dict[f"sideway_movements_in_safe_zone_cnt_mid_penalty{suffix}"] = sideway_movements_in_safe_zone_counts["mid_penalty"]
        full_dict[f"sideway_movements_in_safe_zone_cnt_high_penalty{suffix}"] = sideway_movements_in_safe_zone_counts["high_penalty"]

        # 위험 구역에서 좌우 이동한 횟수
        full_dict[f"sideway_movements_in_danger_zone_cnt_total{suffix}"] = sideway_movements_in_danger_zone_counts["total"]
        full_dict[f"sideway_movements_in_danger_zone_cnt_easy{suffix}"] = sideway_movements_in_danger_zone_counts["easy"]
        full_dict[f"sideway_movements_in_danger_zone_cnt_hard{suffix}"] = sideway_movements_in_danger_zone_counts["hard"]
        full_dict[f"sideway_movements_in_danger_zone_cnt_low_penalty{suffix}"] = sideway_movements_in_danger_zone_counts["low_penalty"]
        full_dict[f"sideway_movements_in_danger_zone_cnt_mid_penalty{suffix}"] = sideway_movements_in_danger_zone_counts["mid_penalty"]
        full_dict[f"sideway_movements_in_danger_zone_cnt_high_penalty{suffix}"] = sideway_movements_in_danger_zone_counts["high_penalty"]

        # 안전 구역에서 횡단보도가 안보일 때 좌우 이동한 횟수
        low_penalty = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["low_penalty"]
        high_penalty = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["high_penalty"]
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_total{suffix}"] = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["total"]
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_easy{suffix}"] = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["easy"]
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_hard{suffix}"] = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["hard"]
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_low_penalty{suffix}"] = low_penalty
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_mid_penalty{suffix}"] = sideway_movements_in_crosswalk_not_visible_safe_zone_counts["mid_penalty"]
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_high_penalty{suffix}"] = high_penalty
        full_dict[f"sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_high_to_low_penalty{suffix}"] = (high_penalty - low_penalty) / (high_penalty + low_penalty + 1e-6)

        # move toward crosswalk
        full_dict[f"sideway_movements_toward_visible_crosswalk_total{suffix}"] = move_toward_crosswalk_counts["total"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_easy{suffix}"] = move_toward_crosswalk_counts["easy"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_hard{suffix}"] = move_toward_crosswalk_counts["hard"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_low_penalty{suffix}"] = move_toward_crosswalk_counts["low_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_mid_penalty{suffix}"] = move_toward_crosswalk_counts["mid_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_high_penalty{suffix}"] = move_toward_crosswalk_counts["high_penalty"]

        # move toward crosswalk (safe)
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_total{suffix}"] = move_toward_crosswalk_safe_zone_counts["total"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_easy{suffix}"] = move_toward_crosswalk_safe_zone_counts["easy"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_hard{suffix}"] = move_toward_crosswalk_safe_zone_counts["hard"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_low_penalty{suffix}"] = move_toward_crosswalk_safe_zone_counts["low_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_mid_penalty{suffix}"] = move_toward_crosswalk_safe_zone_counts["mid_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_safe_zone_cnt_high_penalty{suffix}"] = move_toward_crosswalk_safe_zone_counts["high_penalty"]

        # move toward crosswalk (danger)
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_total{suffix}"] = move_toward_crosswalk_danger_zone_counts["total"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_easy{suffix}"] = move_toward_crosswalk_danger_zone_counts["easy"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_hard{suffix}"] = move_toward_crosswalk_danger_zone_counts["hard"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_low_penalty{suffix}"] = move_toward_crosswalk_danger_zone_counts["low_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_mid_penalty{suffix}"] = move_toward_crosswalk_danger_zone_counts["mid_penalty"]
        full_dict[f"sideway_movements_toward_visible_crosswalk_danger_zone_cnt_high_penalty{suffix}"] = move_toward_crosswalk_danger_zone_counts["high_penalty"]

    return full_dict
