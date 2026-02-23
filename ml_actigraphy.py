# main.py
from config.config import config_actigraphy, config
from typing import List, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
# Local imports from your library
# from library.ml_actigraphy.training import train_nested_cv_xgb_optuna, train_nested_cv_xgb_optuna_parallel_cpu_only, run_nested_cv_with_optuna_parallel
from library.ml_actigraphy.training import  run_nested_cv_with_optuna_parallel
from library.ml_actigraphy.evaluation import plot_roc_with_threshold_cms_grid_avg_subjects, compute_feature_importance, compute_ppv_table
from library.ml_actigraphy.scoring import compute_auc_by_cohort_and_fold

import optuna

APPROVED_FEATURES = [
    "TST", "WASO", "SE", "On", "Off", "AI10", "AI10_w", "AI10_REM", "AI10_REM_w",
    "AI10_NREM", "AI10_NREM_w", "AI30", "AI30_w", "AI30_REM", "AI30_REM_w",
    "AI30_NREM", "AI30_NREM_w", "AI60", "AI60_w", "AI60_REM", "AI60_REM_w",
    "AI60_NREM", "AI60_NREM_w", "TA0.5", "TA0.5_w", "TA0.5_REM", "TA0.5_REM_w",
    "TA0.5_NREM", "TA0.5_NREM_w", "TA1", "TA1_w", "TA1_REM", "TA1_REM_w",
    "TA1_NREM", "TA1_NREM_w", "TA1.5", "TA1.5_w", "TA1.5_REM", "TA1.5_REM_w",
    "TA1.5_NREM", "TA1.5_NREM_w", "SIB0", "SIB0_w", "SIB0_REM", "SIB0_REM_w",
    "SIB0_NREM", "SIB0_NREM_w", "SIB1", "SIB1_w", "SIB1_REM", "SIB1_REM_w",
    "SIB1_NREM", "SIB1_NREM_w", "SIB5", "SIB5_w", "SIB5_REM", "SIB5_REM_w",
    "SIB5_NREM", "SIB5_NREM_w", "LIB60", "LIB60_w", "LIB60_REM", "LIB60_REM_w",
    "LIB60_NREM", "LIB60_NREM_w", "LIB120", "LIB120_w", "LIB120_REM",
    "LIB120_REM_w", "LIB120_NREM", "LIB120_NREM_w", "LIB300", "LIB300_w",
    "LIB300_REM", "LIB300_REM_w", "LIB300_NREM", "LIB300_NREM_w", "MMAS",
    "MMAS_w", "MMAS_REM", "MMAS_REM_w", "MMAS_NREM", "MMAS_NREM_w", "T_avg",
    "T_avg_w", "T_avg_REM", "T_avg_REM_w", "T_avg_NREM", "T_avg_NREM_w",
    "T_std", "T_std_w", "T_std_REM", "T_std_REM_w", "T_std_NREM", "T_std_NREM_w",
    "HP_A_ac", "HP_A_ac_w", "HP_A_ac_REM", "HP_A_ac_REM_w", "HP_A_ac_NREM",
    "HP_A_ac_NREM_w", "HP_M_ac", "HP_M_ac_w", "HP_M_ac_REM", "HP_M_ac_REM_w",
    "HP_M_ac_NREM", "HP_M_ac_NREM_w", "HP_C_ac", "HP_C_ac_w", "HP_C_ac_REM",
    "HP_C_ac_REM_w", "HP_C_ac_NREM", "HP_C_ac_NREM_w"
]

if __name__ == '__main__':
    # %% data and path
    # df = pd.read_csv(config.get('data_path').get('pp_actig'))
    df = pd.read_csv(config_actigraphy.get('pp_actig_merged'))  # new dataset with ADRC
    target = 'label'
    print(f'Dataset fo dimension: {df.shape}')
    # %% Include the demographic features
    # df_quest = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    # df_quest = df_quest.loc[df_quest['actig'] == 1, :]
    #
    # df = pd.merge(left=df,
    #               right=df_quest[['subject_id', 'age', 'bmi', 'gender', 'cohort']],
    #               on='subject_id',
    #               how='left',
    #               )
    #
    # APPROVED_FEATURES = APPROVED_FEATURES + ['age', 'gender', 'bmi']
    #
    # # define the continuous features to normalize
    # feat_cont = {f"{feat}": df[feat].max() for feat in APPROVED_FEATURES}
    # features_to_normalize  = [feat for feat, max_ in feat_cont.items() if max_ > 1]

    # %% Counts for images
    # idx_unique = df['subject_id'].drop_duplicates(keep='first').index
    # df_unique = df.loc[idx_unique, :]
    # df_unique[['label', 'cohort']].value_counts()

    # %% output path
    output_dir = config.get('results_path').get('results').joinpath(f'ml_actig_with_adrc')
    path_metrics = output_dir / 'metrics' / "metrics_outer_folds_ci.csv"
    path_predictions = output_dir / "predictions" / "predictions_outer_folds.csv"
    # df.index = df['subject_id
    random_seed = 42  # Random seed for reproducibility

    n_outer_splits = 10
    n_inner_splits = 5
    n_trials = 250

    model_types = ["xgboost"]
    # --- Preprocessing Pipeline ---
    # preprocessor = ColumnTransformer(
    #     transformers=[("num", SimpleImputer(strategy="median"), APPROVED_FEATURES)],
    #     remainder="drop"
    # )

    optuna_sampler = optuna.samplers.TPESampler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not(path_metrics.exists() and path_predictions.exists()):
        # --- Training ---
        print("Starting nested CV training...")
        (df_metrics_ci,
         df_predictions,
         df_inner_val_records) =  run_nested_cv_with_optuna_parallel(
            df=df,
            target_col=target,
            continuous_cols=None,
            feature_cols=APPROVED_FEATURES,
            col_id='subject_id',  # this will be dropped from the data matrix
            model_types=model_types,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_trials=n_trials,
            pos_weight=True,
            outer_use_es=True,
            outer_es_val_frac=0.15,
            study_sampler=optuna_sampler,
            maximize="spec",
            results_dir=output_dir,
        )
    else:
        # df_metrics = pd.read_csv(output_dir / "metrics_outer_folds.csv")
        df_metrics_ci = pd.read_csv(path_metrics)
        df_predictions = pd.read_csv(path_predictions)
        # df_inner_val_records = pd.read_csv(output_dir / "inner_val_records.csv")



    # %% Visualization
    # 1. report all the models
    plot_roc_with_threshold_cms_grid_avg_subjects(df_predictions=df_predictions,
                                                  df_metrics_ci=df_metrics_ci,
                                                  class_names=  {0: "CN", 1: "iRBD"},
                                                  model_type='xgboost',
                                                  title='Actigraphy Based-Model Subject Level',
                                                  scale = 1,
                                                  figsize=(16, 14),
                                                  output_path=output_dir.joinpath(f'all_optimizations.png'))


    optimization = 'auc'
    model_type = 'xgboost'
    df_metrics_ci_best = df_metrics_ci.loc[(df_metrics_ci['optimization'] == optimization) &
                                           (df_metrics_ci['model_type'] == model_type), :]
    df_predictions_best = df_predictions.loc[(df_predictions['optimization'] == optimization) &
                                             (df_predictions['model_type'] == model_type), :]
    plot_roc_with_threshold_cms_grid_avg_subjects(df_predictions=df_predictions_best,
                                                  df_metrics_ci=df_metrics_ci_best,
                                                  class_names=  {0: "CL", 1: "iRBD"},
                                                  model_type=model_type,
                                                  title='Actigraphy Based-Model Subject Level',
                                                  scale=1.3,
                                                  suptitle_size=12,
                                                  figsize=(16, 5),
                                                  output_path=None,
                                                # output_path = output_dir.joinpath(f'best_model_{optimization}_actigraphy.png')
                                                  )

    ppv_table = compute_ppv_table(df_predictions, df_metrics_ci, model_type=model_type, prevalence=0.015)
    print(ppv_table)

    round(ppv_table.ppv_adj.mean(), 1)
    round(ppv_table.ppv_adj.std(), 1)

    round(ppv_table.ppv.mean(), 1)
    round(ppv_table.ppv.std(), 1)

    # %% Validation metrics separate cohorts
    df_quest = pd.read_csv(config.get('data_path').get('pp_questionnaire'))

    df_predictions_best = df_predictions_best.loc[df_predictions_best['threshold_type'] == 'youden', :]
    # average the subjects so we have one subject observation per fold
    df_predictions_best_avg = (
        df_predictions_best
        .groupby(["subject_id"])
        .agg(y_true=("y_true", "first"),
             outer_fold=('outer_fold', 'first'),
             y_score=("y_score", "mean"),
             y_score_std=("y_score", "std"))
        .reset_index()
    )
    df_predictions_best_avg["y_score_std"] = df_predictions_best_avg["y_score_std"].fillna(0)

    disjoint_subjects = set(df_quest['subject_id']) ^ set(df_predictions_best_avg['subject_id'])
    print(f"{len(disjoint_subjects)} disjoint subjects")

    missing_in_predictions = set(df_quest['subject_id']) - set(df_predictions_best_avg['subject_id'])
    missing_in_questionnaire = set(df_predictions_best_avg['subject_id']) - set(df_quest['subject_id'])

    print(f"Missing in predictions: {len(missing_in_predictions)}")
    print(f"Missing in questionnaire: {len(missing_in_questionnaire)}")


    df_predictions_best_cohort = pd.merge(left=df_predictions_best_avg,
                                          right=df_quest[['subject_id', 'cohort']],
                                          on='subject_id',
                                          how='left')


    assert df_predictions_best_cohort.shape[0] == df_predictions_best.shape[0]

    df_res_cross_cohort, df_summary_cross_cohort = compute_auc_by_cohort_and_fold(df_predictions_best_cohort)

    # %% Model evaluation
    compute_feature_importance(models_dir=output_dir.joinpath('models', 'xgboost'),
                               features_names=APPROVED_FEATURES,
                               objective="youden",
                               top_n=15)

    # %%  Count nights per subject
    nights_per_subject = df.groupby("subject_id").size().reset_index(name="n_nights")
    # Add each subject's label (same for all their rows)
    # (take the first value, since it's consistent per subject)
    labels = df.groupby("subject_id")["label"].first().reset_index()
    # Merge counts with labels
    subject_summary = nights_per_subject.merge(labels, on="subject_id")
    # Compute mean and std per group
    result = (
        subject_summary
        .groupby("label")["n_nights"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    df['label'].value_counts()  # number of nights per group

    # %%
    df_unique = df.drop_duplicates(['subject_id'], keep='first')
    df_unique.label.value_counts()