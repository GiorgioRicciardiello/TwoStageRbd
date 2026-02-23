"""
======================================================================
Data Integration and Preprocessing Script for Questionnaire Cohorts
======================================================================

Description
-----------
This script processes, harmonizes, and integrates multiple datasets
(Clinic/SHAS, Stanford, VascBrain, and Actigraphy) into a single
unified dataframe (`df_data`). It performs cleaning, variable
standardization, and dataset merging to prepare for downstream
statistical analysis and modeling.

Key Steps
---------
1. Load raw data from multiple sources:
   - Clinic/SHAS questionnaire (`raw_questionnaire`)
   - Stanford demographics (`raw_dem_stanford`)
   - Stanford race/ethnicity matrix (`raw_dem_race_stanford`)
   - Actigraphy data (`raw_actigraphy`)
   - VascBrain dataset (`raw_vasc`)

2. Standardize column names across datasets (snake_case, lowercase).

3. Harmonize variables:
   - Diagnosis → {Control=0, iRBD=1}
   - Gender → {M=1, F=0}
   - Dataset label → {"Clinic", "SHAS", "Stanford", "VASC"}
   - Race and Ethnicity mapped to broad categories + numeric encodings.

4. Handle missing values:
   - Impute BMI, age, gender from Stanford demographics where available.
   - Fill questionnaire answers and harmonize categorical encodings.
   - Assign dataset membership based on `subject_id` prefixes.

5. Aggregate Actigraphy features (TST, WASO, SE, etc.) per subject.

6. Create derived flags:
   - `has_quest` → 1 if subject has complete questionnaire responses.
   - `actig` → 1 if actigraphy measures are available.
   - `vasc_brain` → 1 if subject belongs to VascBrain cohort.

7. Save the final harmonized dataframe to
   `config['data_path']['pp_questionnaire']`.

Outputs
-------
- CSV file containing a single harmonized dataframe with:
  - `subject_id`
  - Demographics (age, gender, race, ethnicity, BMI)
  - Questionnaire responses (Q1–Q5)
  - Actigraphy averages
  - Flags (`has_quest`, `actig`, `vasc_brain`)
  - Clean categorical + numeric encodings for race and ethnicity

Usage
-----
Run this script as a standalone program:

    $ python preprocess_questionnaire.py
======================================================================
"""

import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from config.config import config
import ast
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def visualize_table(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Count the unique pair combinations in a dataframe within the grouped by columns.
    :param df: Input DataFrame
    :param group_by: List of column names to group by
    :return: DataFrame showing counts of unique combinations
    """
    df_copy = df.copy()
    print("Distribution before modification:")

    # Only fill NaN with 'NaN' in object (string) columns to avoid dtype issues
    df_plot_before = df_copy.copy()
    for col in df_plot_before.select_dtypes(include='object'):
        df_plot_before[col] = df_plot_before[col].fillna('NaN')

    grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(name='Counts')

    print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
    print(f'Remaining Rows: {df_copy.shape[0]}')
    return grouped_counts_before


def reshape_race_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converting:
                              ID AXRBD117 AXRBD000  ... AXRBD114 AXRBD115 AXRBD116
        0  White/Asian/Black        W        W  ...        W        W        W
        1   Latino/NonLatino       NL       NL  ..

    To:
                ID         id race ethnicity
            0    AXRBD117    W        NL
            1    AXRBD000    W        NL
            2    AXRBD001    W        NL
            3    AXRBD002    W        NL
            4    AXRBD003    W        NL
            ..        ...  ...       ...
    :param df:
    :return:
    """
    # Make sure the first column (labels) is clean text
    df = df.copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

    # Turn the label column into the index, transpose so IDs become rows
    out = (
        df.set_index(df.columns[0]).T
        .rename(columns={
            "White/Asian/Black": "race",
            "Latino/NonLatino": "ethnicity"
        })
        .reset_index()
        .rename(columns={"index": "id"})
    )

    # Optional: expand codes -> full names (keeps originals if unmapped)
    race_map = {"W": "White", "A": "Asian", "B": "Black", "O": "Other"}
    eth_map = {"NL": "Latino/NonLatino", "L": "Latino"}

    # Assert all observed codes are covered by the mapping keys
    missing_race = set(out["race"].dropna().unique()) - set(race_map.keys())
    missing_eth = set(out["ethnicity"].dropna().unique()) - set(eth_map.keys())
    assert not missing_race, f"Unknown race code(s): {sorted(missing_race)}"
    assert not missing_eth, f"Unknown ethnicity code(s): {sorted(missing_eth)}"

    out["race"] = out["race"].map(race_map).fillna(out["race"])
    out["ethnicity"] = out["ethnicity"].map(eth_map).fillna(out["ethnicity"])

    # Order columns
    return out[["id", "race", "ethnicity"]]

def generate_single_stf_frame(path: pathlib.Path) -> pd.DataFrame:
    """
    Stanford is given in a two sheet, merge them into a single frame
    :param path:
    :return:
    """
    # Standardize column names
    def clean_cols(df):
        df.columns = [
            re.sub(r"[^\w]+", "_", col.strip().lower()).strip("_")
            for col in df.columns
        ]
        return df

    col_extra = {
        1:'Subject ID',
        3:'GENDER',
        4:'AGE',
        5:'Height',
        6:'Weight',
        7:'BMI',
        11:'Patient Type'
    }

    mapper_diag = {'CASE': 1,
                   'CASE*': 1,
                   'CASE-1night': 1,
                   'CONTROL': 0,
                   }

    # read input
    df_cases = pd.read_excel(path, sheet_name='42CASES')
    df_controls = pd.read_excel(path, sheet_name='42CONTROLS')
    # df_extras = pd.read_excel(path, sheet_name='Sheet1',
    #                           header=None).T

    df_cases = clean_cols(df_cases)
    df_controls = clean_cols(df_controls)

    # df_extras = df_extras.rename(columns=col_extra)
    # df_extras = df_extras[list(col_extra.values())]
    # df_extras = clean_cols(df_extras)

    # df_extras['patient_type'] = df_extras['patient_type'].map(mapper_diag)
    df_cases['patient_type'] = df_cases['patient_type'].map(mapper_diag)
    df_controls['patient_type'] = df_controls['patient_type'].map(mapper_diag)
    # df_extras = df_extras.loc[df_extras['patient_type'].isin([0, 1])]

    # Find common columns among all three
    common_cols = (
        df_cases.columns
        .intersection(df_controls.columns)
        # .intersection(df_extras.columns)
        .tolist()
    )
    # print("Common columns:", common_cols)

    # Concatenate on common columns
    df_all = pd.concat(
        [df_cases[common_cols], df_controls[common_cols]],
        # [df_cases[common_cols], df_controls[common_cols], df_extras[common_cols]],
        axis=0,
        ignore_index=True
    )
    # keep only first, drop duplicate
    df_all = df_all.drop_duplicates(subset="subject_id", keep="first").reset_index(drop=True)

    df_all.replace('?', np.nan, inplace=True)
    df_all['age'] = df_all['age'].astype(float)
    df_all['bmi'] = df_all['bmi'].astype(float)

    assert df_all.subject_id.is_unique

    return df_all

# %% Main
if __name__ == "__main__":
    # %% Read the data
    df_data = pd.read_excel(config.get('data_path').get('raw_questionnaire'))
    # df_raw = pd.read_excel(config.get('data_path').get('raw_questionnaire'))
    # df_dem_sinai = pd.read_csv(config.get('data_path').get('raw_dem_sinai'), encoding='latin1')
    #%% Stanford
    df_dem_stf = generate_single_stf_frame(config.get('data_path').get('raw_dem_stanford'))
    df_stf_race = pd.read_excel(config.get('data_path').get('raw_dem_race_stanford'))
    # %% Actigraphy
    df_act = pd.read_csv(config.get('data_path').get('raw_actigraphy'))
    # %% VascBrain
    df_vasc = pd.read_excel(config.get('data_path').get('raw_vasc'))

    # %% format column names
    df_data.columns = [
        re.sub(r'[^\w]+', '_', col.strip().lower()).strip('_')
        for col in df_data.columns
    ]

    df_dem_stf.columns = [
        re.sub(r'[^\w]+', '_', col.strip().lower()).strip('_')
        for col in df_dem_stf.columns
    ]

    df_vasc.columns = [
        re.sub(r'[^\w]+', '_', col.strip().lower()).strip('_')
        for col in df_vasc.columns
    ]

    # %% keep only records with actigraphy data
    # df_quest = df_quest.loc[~df_quest['actigraphy_id'].isna(), :].copy()

    # %%
    df_data.rename(columns={'sex': 'gender',
                           'shas_id': 'study_id',
                             }, inplace=True)

    df_data.replace('N/A ', np.nan, inplace=True)

    df_data['diagnosis'] = df_data['diagnosis'].str.lstrip().str.rstrip()


    # %% map variables
    mappers = {
        'diagnosis': {
            'Control': 0,
            'iRBD': 1,
            'Healthy control': 0,
        },
        'gender': {
            'M': 1,
            'F': 0
        },
        # 'data_set': {
        #     'Gilyadov': 'Clinic',
        #     'SHAS': 'SHAS',
        #     'Clinic - Clean Data': 'Clinic',
        #     'May Clinic Data': 'Clinic',
        #     'April Clinic Data': 'Clinic',
        #     'Stanford': 'Stanford',
        #     'Mimic': 'Clinic'
        # }
    }

    df_data['diagnosis'] = df_data['diagnosis'].map(mappers['diagnosis'])
    df_data['gender'] = df_data['gender'].map(mappers['gender'])
    # df_data['data_set'] = df_data['data_set'].map(mappers['data_set'])
    # TS-001: SHAS
    # TS-002: Stanford
    # TS-003: clinic
    df_data.loc[df_data["master_id"].str.startswith("TS-001"), "data_set"] = "SHAS"
    df_data.loc[df_data["master_id"].str.startswith("TS-002"), "data_set"] = "Stanford"
    df_data.loc[df_data["master_id"].str.startswith("TS-003"), "data_set"] = "Clinic"

    df_dem_stf['gender'] = df_dem_stf['gender'].map(mappers['gender'])
    df_vasc['gender'] = df_vasc['gender'].map(mappers['gender'])


    df_dem_stf = df_dem_stf[['subject_id',  'gender', 'age', 'bmi']]

    df_data['race'].replace({'unknown': np.nan,
                              'patient declined': np.nan},
                             inplace=True)
    df_data = df_data.copy()
    df_data = df_data.rename(columns={'study_id': 'subject_id'})
    # df_data['subject_id'] = df_data['subject_id'].str.replace('MIM', 'CLN')

    df_vasc['diagnosis'] = df_vasc.diagnosis.str.strip()
    df_vasc['diagnosis'] = df_vasc['diagnosis'].replace(mappers['diagnosis'])
    # df_vasc = df_vasc[~df_vasc['diagnosis'].isna()]
    df_vasc.loc[~df_vasc['diagnosis'].isin([0, 1]), 'diagnosis'] = 0

    df_vasc.reset_index(inplace=True, drop=True)
    df_vasc = df_vasc[df_vasc['diagnosis'].isin(list(set(mappers['diagnosis'].values())))]
    mask_numeric = df_vasc["id_lily"].apply(lambda x: not isinstance(x, str))
    df_vasc.loc[mask_numeric, 'id_lily'] = df_vasc.loc[mask_numeric, 'id']

    df_vasc.rename(columns={'rbd_sq_1_yes': 'q1_rbd',
                            'constipation_1_yes': 'q4_constipation',
                            'abnormal_olfaction_subjective': 'q2_smell',
                            'id_lily': 'subject_id'},
                   inplace=True)
    df_vasc = df_vasc.drop(columns=['id',
                                    'participant_id',
                                    # 'x',
                                    'standing_bp',
                                    'abnormal_olfaction_objective',
                                    'supine_bp'])
    df_vasc['subject_id'] = df_vasc['subject_id'].str.replace(r'^(SHAS)', r'\1-', regex=True)
    mapper_quest_vasc = {
        "Don't Know": 0.5,
        "DON'T KNOW": 0.5,
        'no': 0,
        'NO':0,
        'YES':1,
        'yes':1
    }
    df_vasc['q1_rbd'] = df_vasc['q1_rbd'].map(mapper_quest_vasc)
    df_vasc['q2_smell'] = df_vasc['q2_smell'].map(mapper_quest_vasc)
    df_vasc['q4_constipation'] = df_vasc['q4_constipation'].map(mapper_quest_vasc)
    df_vasc["data_set"] = np.where(
        df_vasc["subject_id"].str.startswith("SHAS"),
        "SHAS",
        "VASC"
    )


    data_ids = set(df_data['subject_id'])
    act_ids = set(df_act['subject_id'])

    # vasc subjects to drop = not in data OR not in actigraphy
    drop_subjects = [val for val in df_vasc['subject_id']
                     if val not in data_ids or val not in act_ids]

    df_stf_race = reshape_race_ethnicity(df=df_stf_race)
    df_stf_race = df_stf_race.rename(columns={'id': 'subject_id'})
    
    # make stanford a single row per subject frame
    df_act['subject_id'] = df_act['subject_id'].str.upper()
    df_act['subject_id'] = df_act['subject_id'].str.replace(r'^(SHAS)', r'\1-', regex=True)
    df_act = df_act.rename(columns={'label': 'diagnosis'})

    cols_avg = ['TST', 'WASO', 'SE','T_avg', 'nw_night', 'diagnosis']
    df_act.groupby(by='subject_id')
    df_act_avg = (
        df_act
        .groupby('subject_id')[cols_avg]
        .mean()
        .reset_index()
    )
    df_act_avg['diagnosis'] = df_act_avg['diagnosis'].astype(int)


    assert df_stf_race['subject_id'].is_unique
    assert df_dem_stf['subject_id'].is_unique
    assert df_act_avg['subject_id'].is_unique
    assert df_vasc['subject_id'].is_unique
    assert df_data['subject_id'].is_unique

    # %% Include all the IDs in the main df_data
    id_frames = pd.concat([
        df_data[['subject_id']],  # Clinic/SHAS
        df_stf_race[['subject_id']],  # Stanford
        df_vasc[['subject_id']],  # VASC
        df_act_avg[['subject_id']] #actigraphy subjects
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)

    assert id_frames['subject_id'].is_unique

    # Merge back with df_quest (your "main" dataframe)
    df_data = id_frames.merge(df_data, on="subject_id", how="left")


    # Only fill where dataset is missing
    mask = df_data["data_set"].isna()
    df_data.loc[mask & df_data["subject_id"].str.startswith("AXRBD"), "data_set"] = "Stanford"
    df_data.loc[mask & df_data["subject_id"].str.startswith("SHAS"), "data_set"] = "SHAS"
    df_data.loc[mask & df_data["subject_id"].str.startswith(("CLN", 'MIM')), "data_set"] = "Clinic"
    mask = df_data["data_set"].isna()
    df_data.loc[mask, "data_set"] = "VASC"

    assert df_data['subject_id'].is_unique

    # Counts and percentages side by side
    dataset_counts = df_data['data_set'].value_counts().to_frame('count').assign(
        percent=lambda x: 100 * x['count'] / x['count'].sum()
    )
    print(tabulate(dataset_counts, headers='keys', tablefmt='grid'))



    # %% include stanford demographics
    # format stanford race and include ethnicity
    df_data = df_data.merge(df_stf_race,
                              on='subject_id',
                              how='left',
                              suffixes=('', '_new')
                              )
    df_data['race'] = df_data['race'].fillna(df_data['race_new'])
    df_data['ethnicity'] = df_data['ethnicity'].fillna(df_data['ethnicity_new'])
    df_data.drop(columns=['race_new', 'ethnicity_new'], inplace=True)

    # Fill missing bmi values in df_data with the new ones from df_dem_stf
    df_data = pd.merge(df_data,
                    df_dem_stf[['subject_id',  'gender', 'age', 'bmi']],
                    left_on='subject_id',
                    right_on='subject_id',
                    how='left',
                    suffixes=('', '_new')
                )
    col_new = [col for col in df_data.columns if '_new' in col]
    for col in col_new:
        base_col = col.replace('_new', '')
        df_data[base_col] = df_data[base_col].fillna(df_data[col])
    df_data = df_data.drop(columns=col_new)

    #%% include the actigraphy measures averages
    # get only the subjects that have the questionnaire data
    df_data = pd.merge(left=df_data,
                             right=df_act_avg,
                             left_on='subject_id',
                             right_on='subject_id',
                             how='left',
                            suffixes=('', '_new')
                             )
    col_new = [col for col in df_data.columns if '_new' in col]
    for col in col_new:
        base_col = col.replace('_new', '')
        df_data[base_col] = df_data[base_col].fillna(df_data[col])
    df_data = df_data.drop(columns=col_new)

    # mark which ones have actigraphy
    df_data['actig'] = 0
    df_data.loc[~df_data['TST'].isna(), 'actig'] = 1

    # mark which ones have questionnaire
    col_questionnaire = [col for col in df_data.columns if col.startswith('q')]
    # has_quest = 1 if no NaNs in that row across questionnaire columns, else 0
    df_data["has_quest"] = df_data[col_questionnaire].notna().all(axis=1).astype(int)

    # mark which ones are from vasc
    df_data['vasc_brain'] = 0
    df_data.loc[df_data['subject_id'].isin(df_vasc['subject_id']), 'vasc_brain'] = 1
    df_data.loc[df_data['data_set'] == 'VASC', 'vasc_brain'] = 1

    # %% Include VASC
    df_data = pd.merge(
        left=df_data,
        right=df_vasc[[ 'age', 'bmi', 'gender', 'race',
                        'subject_id',
                        'diagnosis',
                        'q1_rbd','q4_constipation', 'q2_smell']],
        left_on='subject_id',
        right_on='subject_id',
        how='left',
        suffixes=('', '_new')
    )
    col_new = [col for col in df_data.columns if '_new' in col]
    for col in col_new:
        base_col = col.replace('_new', '')
        df_data[base_col] = df_data[base_col].fillna(df_data[col])
    df_data = df_data.drop(columns=col_new)

    # %% Ethnicity mapper
    ethnicity_mapper = {
        # Missing / declined
        "unknown": np.nan,
        "patient declined": np.nan,

        # Non-hispanic
        "non-hispanic": "Non-Hispanic",
        "latino/nonlatino": np.nan,  # ambiguous → set missing

        # Generic latino
        "latin american": "Hispanic",
        "latin american ": "Hispanic",
        "latin american.": "Hispanic",  # just in case
        "latino": "Hispanic",

        # Nationalities → Latino
        "puertorican": "Hispanic",
        "dominican": "Hispanic",
        "colombian": "Hispanic",

        "spaniard": "European",

        "ecuadorian": "Hispanic",
        "mexican": "Hispanic",
        "honduran": "Hispanic",
        "venezuelan": "Hispanic",
    }
    ethnicity_mapper_numeric = {
        'Non-Hispanic': 0,
        'Hispanic': 1,
        "European": 2,
    }

    df_data["ethnicity"] = df_data["ethnicity"].str.strip().str.lower().replace(ethnicity_mapper)
    df_data["ethnicity_num"] = df_data["ethnicity"].replace(ethnicity_mapper_numeric)

    assert df_data["ethnicity"].isna().sum() == df_data["ethnicity_num"].isna().sum()

    # Race mapper
    race_mapper = {
        # Basic categories
        "white": "White",
        "black": "Black or African American",
        "black of african american": "Black or African American",
        "black or african american": "Black or African American",
        "asian": "Asian",
        "other": "Other",
        "unknown": np.nan,

        # Nationalities → map into larger groups
        "jamaican": "Black or African American",
        "west indian": "Black or African American",
        "asian indian": "Asian",
        "japanese": "Asian",
        "korean": "Asian",
        "chinese": "Asian",
        "filipino": "Asian",
        "papua new guinean": "Other",  # could also be Pacific Islander depending on context
        'middle eastern': "Other",  # only 1
        'asian/pacific islander': 'Other'  # only 1
        }

    race_numeric_mapper = {
        "White": 0,
        "Black or African American": 1,
        'Asian': 3,
        # 'asian/pacific islander': 4,
        'asian': 4,
        'Other': 4,
        'middle eastern': 4,
        'Mixed':5,
        }

    set(race_numeric_mapper.keys()) ^ set(race_mapper.values())

    if set(df_data["race"].unique()).isdisjoint(race_numeric_mapper.keys()):
        raise ValueError(f'Mssing numeric mappers in race code')

    df_data["race"] = df_data["race"].str.strip().str.lower().replace(race_mapper)    # Handle multi-race as Mixed
    df_data.loc[df_data["race"].str.contains(",", na=False), "race"] = "Mixed"


    # Apply mapping
    df_data["race_num"] = df_data["race"].replace(race_numeric_mapper)
    assert df_data["race_num"].isna().sum() == df_data["race"].isna().sum()

    # %% sort the columns
    col_sorted = ['subject_id', 'master_id',
                  'data_set',
                  'diagnosis',
                  'age', 'gender', 'bmi', 'race', 'race_num', 'ethnicity', 'ethnicity_num',
                  'other_neuro_sleep_diagnosis',
                  'q1_rbd', 'q2_smell','q4_constipation', 'q5_orthostasis',
                  'TST', 'WASO', 'SE','T_avg', 'nw_night',
                  'actig', 'has_quest', 'vasc_brain']

    df_data = df_data[col_sorted]
    # %% keep records that have only actigraphy or questionnaire, else drop them
    df_data = df_data.loc[~((df_data['actig'] == 0) & (df_data['has_quest'] == 0))]
    df_data.reset_index(inplace=True, drop=True)

    # %% assign the cohort, SASH or Stanford
    df_data['cohort'] = df_data['data_set']

    # df_data['cohort'] = np.nan
    # # SHAS-like cohorts
    # df_data.loc[df_data["subject_id"].str.startswith(("SHAS", "CLN", "MIMI")), "cohort"] = "SHAS"
    # # The rest go to Stanford
    # df_data.loc[df_data["cohort"].isna(), "cohort"] = "Stanford"

    # %% Save into a signle dataframe
    df_data.to_csv(config.get('data_path').get('pp_questionnaire'), index=False)

    # %% Visualzie data
    df_tab = visualize_table(df_data, group_by=['actig', 'has_quest', 'vasc_brain', 'diagnosis', 'cohort'])
    df_tab.Counts.sum()








