import pandas as pd
from util import data_dir, validate_no_op_episodes

from preprocess import load_survey_results_desktop, load_survey_results_web
from preprocess.pedestrian_stats import build_pedestrian_task_results

def build_behavioral_task_results(df, subjIds, web=False):
    behavioral_task_df = load_behavioral_task_results(subjIds, web)
    df = pd.merge(df, behavioral_task_df, on="subj_id", how="inner")
    return df

def load_behavioral_task_results(subjIds, web):
    dfs = []
    for subjId in subjIds:
        DDT_mean_log_k, CRA_mean_alpha = load_behavioral_task_results_ind(subjId, web)
        dfs.append(pd.DataFrame({
            "subj_id": [subjId],
            "DDT_mean_log_k": [DDT_mean_log_k],
            "CRA_mean_alpha": [CRA_mean_alpha],
        }))
    return pd.concat(dfs, ignore_index=True)

def load_behavioral_task_results_ind(subjId, web):
    DDT_mean_log_k, CRA_mean_alpha = None, None
    subj_data_dir = data_dir(subjId, web)
    for file in subj_data_dir.iterdir():
        if file.is_dir(): continue
        if file.name.startswith("DDT"):
            if DDT_mean_log_k is not None:
                raise Exception("multiple DDT results found")
            df = pd.read_csv(file, sep="\t")
            DDT_mean_log_k = df.iloc[-1]["mean_log_k"]
        if file.name.startswith("CRA"):
            if CRA_mean_alpha is not None:
                raise Exception("multiple CRA results found")
            df = pd.read_csv(file, sep="\t")
            CRA_mean_alpha = df.iloc[-1]["mean_alpha"]
    return DDT_mean_log_k, CRA_mean_alpha

def load_and_save_desktop():
    df = load_survey_results_desktop()
    subj_ids = df["subj_id"].to_list()
    if len(subj_ids) == 0: return

    df = build_pedestrian_task_results(df, subj_ids, web=False)
    df = build_behavioral_task_results(df, subj_ids, web=False)
    df.to_csv("data/data_desktop.csv", index=False)

def load_and_save_web(subjIDs = None, verbose=False, save=True, validate=True):
    df = load_survey_results_web(subjIDs) # NOTE: only people who finished the entire experiment has a survey.csv file
    subjIds = df["subj_id"].to_list()
    if len(subjIds) == 0: return
    if validate:
        validate_no_op_episodes(subjIds, web=True, verbose=verbose)
    df = build_pedestrian_task_results(df, subjIds, web=True)
    if save:
        df.to_csv("data/data_web.csv", index=False)

if __name__ == "__main__":
    load_and_save_desktop()
    load_and_save_web(validate=False)
