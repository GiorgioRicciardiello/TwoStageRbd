

import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from config.config import config, config_actigraphy
import ast
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import sys
import datetime

def matlab_datenum_to_datetime(datenum):
    return datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=datenum%1) - datetime.timedelta(days=366)

def clean_subject_id(x:str) -> str:
    x = x.upper()
    if 'SHAS' in x:
        x = x.replace('SHAS', 'SHAS-')
    return x

# %% Main
if __name__ == "__main__":
    # %% Define the Actigraphy raw datasets:
    df_actig_raw = pd.read_csv(config.get('data_path').get('raw').joinpath(r'nightly_features_raw.csv'))
    df_actig_adrc = pd.read_csv(config_actigraphy.get('raw_adrc'))

    # %% Define the demographics table
    df_quest = pd.read_csv(config.get('data_path').get('pp_questionnaire'))

    # %% Output
    path_output = config.get('data_path').get('pp_actig')

    # %% ==================== Pre-process the raw dataset ====================
    if not 'subject_id' in df_actig_raw.columns:
        df_actig_raw = df_actig_raw.rename(columns={'ID': 'subject_id'})
    if 'ID' in df_actig_raw.columns:
        df_actig_raw = df_actig_raw.drop(columns=['ID'])

    if not 'subject_id' in df_actig_adrc.columns:
        df_actig_adrc = df_actig_adrc.rename(columns={'ID': 'subject_id'})
    if 'ID' in df_actig_adrc.columns:
        df_actig_adrc = df_actig_adrc.drop(columns=['ID'])

    # Include the labels (diagnosis)
    if not 'label' in df_actig_adrc.columns:
        df_actig_adrc = pd.merge(left=df_actig_adrc,
                                 right=df_quest[['subject_id', 'diagnosis']], )
        df_actig_adrc = df_actig_adrc.rename(columns={'diagnosis': 'label'})

    if not 'label' in df_actig_raw.columns:
        df_actig_raw = pd.merge(left=df_actig_raw,
                                 right=df_quest[['subject_id', 'diagnosis']], )
        df_actig_raw = df_actig_raw.rename(columns={'diagnosis': 'label'})

    # Only add if it doesn't exist
    if 'night_seq' not in df_actig_adrc.columns:
        # Convert dates from MATLAB datenum if needed
        df_actig_adrc["Date"] = df_actig_adrc["Date"].apply(matlab_datenum_to_datetime)
        # Sort by subject_id and converted date
        df_actig_adrc = df_actig_adrc.sort_values(by=["subject_id", "Date"])
        # Generate sequential night numbers
        df_actig_adrc["night_seq"] = df_actig_adrc.groupby("subject_id").cumcount() + 1
        df_actig_adrc = df_actig_adrc.drop(columns=["Date"])


    # Only add if it doesn't exist
    if 'night_seq' not in df_actig_raw.columns:
        # Convert dates from MATLAB datenum if needed
        df_actig_raw["Date"] = df_actig_raw["Date"].apply(matlab_datenum_to_datetime)
        # Sort by subject_id and converted date
        df_actig_raw = df_actig_raw.sort_values(by=["subject_id", "Date"])
        # Generate sequential night numbers
        df_actig_raw["night_seq"] = df_actig_raw.groupby("subject_id").cumcount() + 1
        df_actig_raw = df_actig_raw.drop(columns=["Date"])


    cols_unmatch = set(df_actig_raw.columns) ^ set(df_actig_adrc.columns)
    for col in cols_unmatch:
        if col in df_actig_raw.columns:
            df_actig_raw = df_actig_raw.drop(columns=col)

        if col in df_actig_adrc.columns:
            df_actig_adrc = df_actig_adrc.drop(columns=col)
    cols_unmatch = set(df_actig_raw.columns) ^ set(df_actig_adrc.columns)
    assert len(cols_unmatch)==0

    if 'Class' in df_actig_adrc.columns:
        df_actig_adrc.drop(columns=['Class'], inplace=True)

    if 'Class' in df_actig_raw.columns:
        df_actig_raw.drop(columns=['Class'], inplace=True)


    # make single dataset
    df_actig = pd.concat([df_actig_raw, df_actig_adrc], axis=0)

    # format the subject id
    df_actig['subject_id'] = df_actig['subject_id'].apply(clean_subject_id)
    # Define the site
    df_actig = pd.merge(left=df_actig,
             right=df_quest[['subject_id', 'cohort']],
                        on='subject_id')

    if 'site' in df_actig.columns: df_actig = df_actig.drop(columns=['site'])


    # %% ==================== Filter by age ====================
    age_min, age_max = 40, 80
    subjects_age = df_quest.loc[(df_quest['age'].isna()) | ((df_quest['age'] >= age_min) & (df_quest['age'] <= age_max)), 'subject_id' ]

    # subjects that are not in the df_quest should not be part of the study
    df_actig = df_actig.loc[df_actig['subject_id'].isin(subjects_age)]

    # %% ==================== Clean Bad Nights ====================
    total_nights = len(df_actig_raw)
    df_actig_raw_clean = df_actig_raw.copy()
    # --- Rule 1: discard TST <3 h or >12 h ------------------------------
    df_actig["flag_bad_TST"] = (df_actig["TST"] < 3) | (df_actig["TST"] > 12)
    # --- Rule 2: discard low temperature (<27 °C) -----------------------
    df_actig["flag_low_temp"] = df_actig["T_avg"] < 27
    # --- Rule 3: discard non-wear >2 h between 0-6 AM -------------------
    df_actig["flag_nonwear_night"] = df_actig["nw_night"] > 4

    df_actig["good_night"] = ~(df_actig[["flag_bad_TST", "flag_low_temp", "flag_nonwear_night"]].any(axis=1))
    # Count losses per stage
    loss_bad_tst = df_actig["flag_bad_TST"].sum()
    loss_low_temp = df_actig["flag_low_temp"].sum()
    loss_nonwear = df_actig["flag_nonwear_night"].sum()
    loss_any = (~df_actig["good_night"]).sum()
    kept_final = df_actig["good_night"].sum()

    print(f"Total nights: {total_nights}")
    print(f"Lost at bad_TST: {loss_bad_tst}")
    print(f"Lost at low_temp: {loss_low_temp}")
    print(f"Lost at nonwear_night: {loss_nonwear}")
    print(f"Lost by any rule: {loss_any}")
    print(f"Kept after all QC: {kept_final}")
    # keep only the good nights
    df_actig = df_actig[df_actig["good_night"]]
    df_actig = df_actig.drop(columns=["good_night", 'flag_bad_TST','flag_low_temp', 'flag_nonwear_night' ])

    # organize columns
    # Define the desired column order
    first_cols = ['subject_id', 'night_seq', 'cohort', 'label']
    remaining_cols = [col for col in df_actig.columns if col not in first_cols]

    # Reorder the DataFrame
    df_actig = df_actig[first_cols + remaining_cols]

    # %% ==================== Counts ====================
    # count how many records per subjects
    df_counts_raw = (
        df_actig["subject_id"]
        .value_counts()
        .reset_index(name="n_nights")  # count column
        .rename(columns={"index": "subject_id"})  # rename the subject column
        .sort_values(by="n_nights", ascending=False)
    )

    df_grouped = (
        df_actig[['subject_id', 'cohort']]
        .drop_duplicates()
        .groupby('cohort')
        .size()
        .reset_index(name='count')
    )
    print(df_grouped )

    # Replace inf/-inf with NaN before saving
    df_actig = df_actig.replace([np.inf, -np.inf], np.nan)
    # Sanity check for inf/-inf
    numeric_df = df_actig.select_dtypes(include=[np.number])
    if np.isinf(numeric_df.to_numpy()).any():
        print("⚠️ Warning: DataFrame still contains infinity values in numeric columns!")
    else:
        print("✅ No infinity values found in numeric columns.")
    # %% ==================== Save Frame ====================
    df_actig.to_csv(config_actigraphy.get('pp_actig_merged'), index=False)








































