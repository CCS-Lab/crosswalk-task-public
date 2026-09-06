from pathlib import Path
import pandas as pd

RAW_SUBJECT_ID_COLNAME = "prolific_id"

EXCLUDE_IDS = {
    # rejected - WORST CASES
    "5cce9b46b09ec90015f3ef05", # Low attention/effort: Out of 40 minutes of the main task, pressed nothing for 19 minutes and 30 seconds straight)- did nothing - 19 minutes 30 seconds # noop episode detected - 39 out of 75 episodes [1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 2005, 2006, 2007, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031] # noop for over 10 seconds episode detected - 40 out of 75 episodes [1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 2005, 2006, 2007, 2008, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031]
    "69a9ecf393dd16f99a6a7abf", # pressed nothing for 30 seconds straight for 9 out of 126 stages, and pressed the same button for 10 stages.
    "69c442fac787bfd111f5df6c", # pressed nothing for 30 seconds straight for 8 out of 101 stages, and pressed the same button for 10 stages.
    "69b89d7352b40a4e5843f201", # noop episode detected - 7 out of 130 episodes [1032, 1033, 1034, 1035, 1036, 2063, 2064] # noop for over 10 seconds episode detected - 8 out of 130 episodes [1032, 1033, 1034, 1035, 1036, 1037, 2063, 2064]
    "5a947fb0f05361000171b5a3", # noop episode detected - 9 out of 117 episodes [1032, 1033, 1034, 1039, 1040, 1047, 1048, 2054, 2057] # noop for over 10 seconds episode detected - 10 out of 117 episodes [1032, 1033, 1034, 1039, 1040, 1041, 1047, 1048, 2054, 2057]
    "58adfc6e7cf56d0001f931a2", # noop episode detected - 12 out of 120 episodes [1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 2064] # noop for over 10 seconds episode detected - 17 out of 120 episodes [1021, 1036, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 2004, 2024, 2046, 2064]
    "5c6709ab926a5b0001eb29c1", # noop episode detected - 7 out of 103 episodes [2043, 2044, 2046, 2047, 2048, 2049, 2050] # noop for over 10 seconds episode detected - 15 out of 103 episodes [1004, 1014, 1015, 1038, 2040, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051]
    "6985fe6da973e57559c9fc64", # noop episode detected - 7 out of 95 episodes [1037, 1039, 1040, 2004, 2005, 2006, 2007]   # noop for over 10 seconds episode detected - 14 out of 95 episodes [1010, 1026, 1036, 1037, 1038, 1039, 1040, 1042, 2004, 2005, 2006, 2007, 2023, 2035]      

    # rejected or returned (v2)
    "699632a125fb1bd95501ef4d",
    "69c626b660af6d7fd9dfd041",
    "5c37aa8b832a8400010983b3",
    "69ab36f0e49d1525b5967842",
    "694f50766aba30210e26eb55",
    "5910a5f8748d450001ba3a67",
    "69b8ef7b87c2e000e46e09e6",
    "558b20eefdf99b7dfdbac3c9",
    "5e59feea5bb31220840c12c6",
    "69c70b2ae3a40492f22627e3",
    "63b18d94775ad4aad221103c",
    "675ebd1fb3aef884a42df656",
    "65524ce2443c9df5aba5960c",
    "69c41892ffbcb7d0ac0aa8a8",
    "69b095b0e6122cd784140dbd",
    "69457dc95341cec41fca5695",
    "65c118f1cd19001157ace3a0",
    "615509a589a34f7cb9fb075f",
    "6987bd1fac8e6dd1f6026b85",
    "602fd11faa844cd2d0fd5818",
    "66b4c5d581010580d8d67618",
    "69c132b7ff339dc396efb34f",
    "69b88657023868edd4bcfc60",
    "69cd5c11be43bdb01ac5d0aa",
    "69d13686d7eea723bfa50501",
    "5654175604212700054a12b1",
    "62911ef14fd68295d60ba83b",
    "5a8f13595292b80001235e09",
    "65031ea9cb2723bf741d4151",
    "67e5ebfad8af26b1bb60e460",
    "5edc7aed858a159580341afa",
    "69bd0d02afc8d65e087ddb4b",
    "69d02f08c6b3bec8413e7511",
    "69cf9a6716e5f72c429ba868",
    "675c89cc4e1a6bad637f454f",
    "69b1d6020035e7641df37c19",
    "67be1d86f01089f6bc1677f1",
    "67d38d0b4d7fc9c2517ff079",
    "69ac4a8252f08f66b05aa859",
    "673e0f21862638c66615b402",
    "694434f55e41999a800c4ef6",
    "69c2a2d827fe14230000c13c",
    "67dd4210d92f4c3aac8d944c",
    "5b0b77a9d282ff00015b7531",
    "6824afdfce30efcd2e7274ef",
    "6978ff0b2fd50c60960e62a6",
    "67f015260f679212b3c34caa",
    "66bfe13c60dea5964d491735",
    "63616640b17046286153dad8",
    "63e5e4a0e6325fe0fabe8ee6",
    "5b98940f754b700001f0f15c",
    "69c22c43b7035cbd7c8d0082",
    "65c25e3de8b230c294f2b32c",
    "5b5a822e1ad8270001c4f28d",
    "69a3b07b245e39c0d43f0f81",
    "676334ae578dda6ef33f5886",
    "5b9c2c816f92f30001ecce7a",
    "678a85c749e99bc3baed6452",
    "63ed78d1217e614a3c6b392e",
    "673ec935d1ab67f5acdd49e3",
    "695ffecb21296f4c852e1591",
    "694180d069557e7a626bbbfc",
    "65fccfbedce99b1065ef1768",
    "60f2a2b4932fed5326988e0f",
    "69a9b76bf524448d100fc51b",
    "699a3895d5edd5d66f02b8ef",
    "5f888a84d0200d03067e17ed",
    "694143fe5c84af5a24325e42",
    "69a65cb77a65aab485084932",
    "673378009f99d14c232c1f2b",
    "604bbfc901b85fb02ae92c1a",
    "5a04869ff2e3460001edad2e",
    "69b8563d512b30c54762f06b",
    "67803a05b60a2727e9fa85b4",
    "6987307ee9e55830b1594fd4",
    "69a396424c59826d1a132598",
    "69b32bdf1b7617a66b7b7f2a",
    "69b885c83f7ae2b8458c6624",
    "67e406244bfad9999a051c1e",
    "677e29646b0a2cf5eff174dc",
    "6999c4e4418779b6b1526adc",
    "6753519ac83cda11cee58723",
    "5e6ca4b9a2b73d25c187ae61",
    "67e92dbd9199b830fbecdf2e",
    "6667627ba6d0d6f7b08686ab",
    "695fb8cce5fd1ede2a4b4b45",
    "66b62d378900b27a626cf2ba",
    "62b9c2f1e55fe9f17dfedc5d",
    "69829bc1224b28f0538900d5",
    "69a26ca7254273cdad33b66e",
    "6405feae7d6321bfaaf793e8",
    "5d845d15fbc5970001e821e0",
    "60e37439f7c69f9e2d4466eb",
    "60085c96f16b701cbdced9a4",
    "6985dbc356145f0a109baabf",
    "69c61f04d8d4b54792b0f267",
    "69c6b1dde56fbd1a6166028d",
    "69a9dcac25a062cd320e83a1",
    "559d9916fdf99b1e2b2dc6b8",
    "69bdf08c99e22a5d274eb0cb",
    "69a5a37d45ce9e498e6d383b",
    "69c4f2892a2dc97148b7c7e1",
    "66199b3cd796398a2ba74e05",
    "65490276dc42961a8e1cdec3",
    "66ccbb7d4109382e498e3c82",
    "6054e5f7c1931593649b158f",
    "5e7e0e92b7838b4aca1b6fca",
    "5bec5b873ffbad0001e8b968",
    "698b5a4b6bf6a0eee98c0c53",
    "671bbaf1bec15d847b871b9b",
    "6266b4ea9408fc167507d923",
    "67e97d506e0dfa7ff0b407c5",
    "6718cc3b749c4925a6ac7b03",
    "67dc70145c70ad07748b3f1b",
    "69458f2aa5309848951f9026",
    "67155947b83a6aee15acbc0d",
    "5f81fc9446ee002f681ea5e3",
    "65e1f75ab3efd22e847cfbf0",
    "66bc28e25fcddfb82e557a03",
    "5c1021a547d07800011a23de",
    "6750f7eb7c60e2b2c22aeaea",
    "63f82a4ff43ec144ca3622a8",
    "671165ce01cd9c2699b90dc1",
    "69a7e18d4c24cacdbb47110c",
    "69a7212afab4b044536b0c1f",
    "66c4f3cb14cbfc4db836bd4e",
    "69c5ae123501a091a2bdcabd",
    "67bc755b022467c613763798",
    "69cedec8ab6dfce0b1572b1f",
    "67cb51b23d74b436185a20eb",
    "67dd46602623d4bf857f7fa2",
    "67e40bab848c22a30914c34c",
    "60859f820e5009a67adb3921",
    "66e87276ccc570e1c0d7df24",
    "6999787fd639f1bad3dd0cff",
    "62c5cdfe50000fce3e390be0",
}

def load_survey_results_web(subjIDs = None):
    BASE_DIR = Path(__file__).resolve().parent.parent / "data/web"
    print(BASE_DIR)
    rows = []
    for csv_path in BASE_DIR.glob("**/survey.csv"):
        subj_id = csv_path.parent.parent.parent.name
        if subj_id in EXCLUDE_IDS:
            print(f"excluding {subj_id}")
            continue
        if subjIDs is not None and subj_id not in subjIDs: continue
        df = pd.read_csv(csv_path)
        if len(df) != 1:
            raise ValueError(f"{csv_path} has {len(df)} rows (expected 1)")
        rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    subj_ids = df[RAW_SUBJECT_ID_COLNAME].tolist()
    if len(subj_ids) != len(set(subj_ids)):
        dup_ids = df.loc[
            df[RAW_SUBJECT_ID_COLNAME].duplicated(),
            RAW_SUBJECT_ID_COLNAME
        ].unique()
        raise ValueError(f"duplicate subject data found: {dup_ids}")

    print(f"Total Subjects: {len(df)} ({subj_ids})")
    return pd.DataFrame({
        "subj_id": df[RAW_SUBJECT_ID_COLNAME],
        "age": _calculate_age(df),
        "gender": _gender(df).astype(int), # NOTE: let's not debate about anything regarding this wording
        **_calculate_BIS(df),
        **_calculate_STAI(df),
        **_calculate_DOSPERT(df),
        **_calculate_IGDS(df),
        **_calculate_IAT(df),
    })

def _calculate_age(df):
    response_date = pd.to_datetime(df["end_timestamp"], unit="ms", errors="coerce") # format: unix timestamp (ex. 1769667523994)
    response_year = response_date.dt.year
    response_month = response_date.dt.month

    birth_year = df["birth_year"].astype(int)
    birth_month = df["birth_month"].astype(int)

    age = response_year - birth_year
    age -= (response_month < birth_month).astype(int)
    return age.astype(int)

def _gender(df):
    gender_map = {
        "Female": 1,
        "Male": 2,
        "Prefer not to say": 3,
        "Other": 3,
    }
    return df["gender"].astype(str).map(gender_map)

def _calculate_BIS(df):
    all_questions = [f"BIS_{i+1}" for i in range(30)]
    df[all_questions] = df[all_questions].astype(int)
    BIS_same_response = _same_response(df, all_questions, "BIS")

    # 이소라, 이원혜, 박정수, 김설민, 김종우, 심재현 (2012). 한국판 Barratt Impulsiveness Scale-11-Revised의 신뢰도 및 타당도 연구 ; 일반성인집단을 중심으로
    prev_BIS_1 = df["BIS_1"].to_list()
    reverse_scored_questions = [1, 7, 8, 9, 10, 12, 13, 15, 20, 29, 30]
    for q in reverse_scored_questions:
        _apply_reverse_scoring(df, f"BIS_{q}", min_score = 1, max_score = 4)
    print(f"reversing BIS_1: {prev_BIS_1} => {df['BIS_1'].to_list()}")

    # 허심양, 오주용, & 김지혜. (2012). 한국판 Barratt 충동성 검사-11 의 신뢰도 및 타당도 연구. 한국심리학회지: 일반, 31(3), 769-782.
    motor_impulsivity_questions = [2, 3, 4, 16, 17, 19, 21, 22, 23, 25, 30] # 운동 충동성 (11)
    nonplanning_impulsivity_questions = [1, 7, 8, 10, 12, 13, 14, 15, 18, 27, 29] # 무계획성 충동성 (11)
    attentional_impulsivity_questions = [5, 6, 9, 11, 20, 24, 26, 28] # 인지 충동성 (8)

    BIS_total = df[all_questions].sum(axis=1)
    BIS_motor = df[[f"BIS_{i}" for i in motor_impulsivity_questions]].sum(axis=1)
    BIS_nonplanning = df[[f"BIS_{i}" for i in nonplanning_impulsivity_questions]].sum(axis=1)
    BIS_attentional = df[[f"BIS_{i}" for i in attentional_impulsivity_questions]].sum(axis=1)
    return {
        "BIS":             BIS_total, # 30 ~ 120
        "BIS_motor":       BIS_motor, # 11 ~ 44
        "BIS_nonplanning": BIS_nonplanning, # 11 ~ 44
        "BIS_attentional": BIS_attentional, # 8 ~ 32
        "BIS_same_response": BIS_same_response,
    }

def _calculate_STAI(df):
    stai_s = [f"STAI-S_{i+1}" for i in range(20)]
    stai_t = [f"STAI-T_{i+1}" for i in range(20)]
    all_questions = [f"STAI_{i+1}" for i in range(40)]
    df[all_questions] = df[stai_s + stai_t].astype(int)
    STAI_same_response = _same_response(df, all_questions, "STAI")

    # Seok, Hamid, Mutang, and Ismail (2018). Psychometric Properties of the State-Trait Anxiety Inventory (Form Y) among Malaysian University Students
    # The reversed score items
    # - 1, 2, 5, 8, 10, 11, 15, 16, 19, 20 (State Anxiety Scale)
    # - 21, 23, 26, 27, 30, 33, 34, 36, 39 (Trait Anxiety Scale)
    prev_STAI_1 =  df["STAI_1"].to_list()
    reverse_scored_questions = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30, 33, 34, 36, 39]
    for q in reverse_scored_questions:
        _apply_reverse_scoring(df, f"STAI_{q}", min_score = 1, max_score = 4)
    print(f"reversing STAI_1: {prev_STAI_1} => {df['STAI_1'].to_list()}")

    STAI_total = df[all_questions].sum(axis=1)
    STAI_S = df[[f"STAI_{i + 1}" for i in range(20)]].sum(axis=1)
    STAI_T = df[[f"STAI_{i + 21}" for i in range(20)]].sum(axis=1)
    return {
        "STAI":   STAI_total, # 40 ~ 160
        "STAI_S": STAI_S, # 20 ~ 80
        "STAI_T": STAI_T, # 20 ~ 80
        "STAI_same_response": STAI_same_response,
    }

def _calculate_DOSPERT(df):
    all_questions = [f"DOSPERT_{i+1}" for i in range(30)]
    df[all_questions] = df[all_questions].astype(int)
    DOSPERT_same_response = _same_response(df, all_questions, "DOSPERT")

    # Hong et al (2010). BART와 K-DOSPERT로 알아본 한국인의 위험감수경향
    ethical_questions = [6, 9, 10, 16, 29, 30] # 윤리, ETH (6)
    financial_questions = [3, 4, 8, 12, 14, 18] # 경제, FIN (6)
    health_safety_questions = [5, 15, 17, 20, 23, 26] # 건강/안전, HEA (6)
    recreational_questions = [2, 11, 13, 19, 24, 25] # 오락, REC (6)
    social_questions = [1, 7, 21, 22, 27, 28] # 사회성, SOC (6)
    assert len(set(ethical_questions + financial_questions + health_safety_questions + recreational_questions + social_questions)) == 30

    DOSPERT_total = df[all_questions].sum(axis=1)
    DOSPERT_ethical = df[[f"DOSPERT_{i}" for i in ethical_questions]].sum(axis=1)
    DOSPERT_financial = df[[f"DOSPERT_{i}" for i in financial_questions]].sum(axis=1)
    DOSPERT_health_safety = df[[f"DOSPERT_{i}" for i in health_safety_questions]].sum(axis=1)
    DOSPERT_recreational = df[[f"DOSPERT_{i}" for i in recreational_questions]].sum(axis=1)
    DOSPERT_social = df[[f"DOSPERT_{i}" for i in social_questions]].sum(axis=1)
    return {
        "DOSPERT_raw": DOSPERT_total, # 35 ~ 210
        "DOSPERT":     pomp(DOSPERT_total, 35, 210), # 0 ~ 100
        "DOSPERT_ETH": DOSPERT_ethical, # 7 ~ 42
        "DOSPERT_FIN": DOSPERT_financial, # 7 ~ 42
        "DOSPERT_HEA": DOSPERT_health_safety, # 7 ~ 42
        "DOSPERT_REC": DOSPERT_recreational, # 7 ~ 42
        "DOSPERT_SOC": DOSPERT_social, # 7 ~ 42
        "DOSPERT_same_response": DOSPERT_same_response,
    }

# IGDS27P (long-form Likert version): 6-point Likert scale (1~6) * 27 Qs
def _calculate_IGDS(df):
    all_questions = [f"IGDS_{i+1}" for i in range(27)]
    df[all_questions] = df[all_questions].astype(int)
    IGDS_same_response = _same_response(df, all_questions, "IGDS")

    preoccpuation_questions = [1, 2, 3] # 몰입, PRE (3)
    tolerance_questions = [4, 5, 6] # 관용, TOL (3)
    withdrawal_questions = [7, 8, 9] # 금단, WIT (3)
    persistence_questions = [10, 11, 12]
    escape_questions = [13, 14, 15] # 도피, ESC (3)
    problems_questions = [16, 17, 18] # 문제, PRO (3
    deception_questions = [19, 20, 21] # 기만, DEC (3)
    displacement_questions = [22, 23, 24] # 대체, DIS (3)
    conflict_questions = [25, 26, 27] # 갈등, CON (3)
    assert len(set(preoccpuation_questions + tolerance_questions + withdrawal_questions + persistence_questions + 
                   persistence_questions + escape_questions + problems_questions + deception_questions + displacement_questions
                    + conflict_questions)) == 27
    IGDS_total = df[all_questions].sum(axis=1)
    IGDS_preoccupation = df[[f"IGDS_{i}" for i in preoccpuation_questions]].sum(axis=1)
    IGDS_tolerance = df[[f"IGDS_{i}" for i in tolerance_questions]].sum(axis=1)
    IGDS_withdrawal = df[[f"IGDS_{i}" for i in withdrawal_questions]].sum(axis=1)
    IGDS_persistence = df[[f"IGDS_{i}" for i in persistence_questions]].sum(axis=1)
    IGDS_escape = df[[f"IGDS_{i}" for i in escape_questions]].sum(axis=1)
    IGDS_problems = df[[f"IGDS_{i}" for i in problems_questions]].sum(axis=1)
    IGDS_deception = df[[f"IGDS_{i}" for i in deception_questions]].sum(axis=1)
    IGDS_displacement = df[[f"IGDS_{i}" for i in displacement_questions]].sum(axis=1)
    IGDS_conflict = df[[f"IGDS_{i}" for i in conflict_questions]].sum(axis=1)
    return {
        "IGDS": IGDS_total, # 27 ~ 162
        "IGDS_PRE": IGDS_preoccupation, # 3 ~ 18
        "IGDS_TOL": IGDS_tolerance, # 3 ~ 18
        "IGDS_WIT": IGDS_withdrawal, # 3 ~ 18
        "IGDS_PRS": IGDS_persistence, # 3 ~ 18
        "IGDS_ESC": IGDS_escape, # 3 ~ 18
        "IGDS_PRO": IGDS_problems, # 3 ~ 18
        "IGDS_DEC": IGDS_deception, # 3 ~ 18
        "IGDS_DIS": IGDS_displacement, # 3 ~ 18
        "IGDS_CON": IGDS_conflict, # 3 ~ 18
        "IGDS_same_response": IGDS_same_response,
    }

def _calculate_IAT(df):
    all_questions = [f"IAT_{i+1}" for i in range(20)]
    df[all_questions] = df[all_questions].astype(int)
    IAT_same_response = _same_response(df, all_questions, "IAT")

    IAT_total = df[all_questions].sum(axis=1)
    # number of (1=Not Applicable) responses
    IAT_num_1_cnt = (df[all_questions] == 1).sum(axis=1)

    # 1="Not Applicable" added to 5-point Likert scale: 2~6 => 1~5
    mapping = {1:1, 2:1, 3:2, 4:3, 5:4, 6:5}
    df_revised = df[all_questions].replace(mapping)
    IAT_revised = df_revised.sum(axis=1) # 20 ~ 100
    return {
        "IAT": IAT_total, # 20 ~ 120
        "IAT_not_applicable_cnt": IAT_num_1_cnt, # 0 ~ 20
        "IAT_revised": IAT_revised, # 20 ~ 100
        "IAT_same_response": IAT_same_response,
    }

def _apply_reverse_scoring(df, col, min_score = 1, max_score = 4):
    df[col] = max_score + min_score - df[col]

def _same_response(df, all_questions, questionnaire):
    df_block = df[all_questions]
    unique_count = df_block.nunique(axis=1, dropna=True)
    first_val = df_block.iloc[:, 0]
    mask = (unique_count == 1)
    print(f"{questionnaire}: same response only", df_block[mask])
    return first_val.where(mask, -1)

def pomp(x, min_score, max_score):
    return (x - min_score) / (max_score - min_score) * 100
