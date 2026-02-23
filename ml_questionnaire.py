import pathlib

import pandas as pd
import optuna
from config.config import config
from library.ml_questionnaire.training import run_nested_cv_with_optuna_parallel
from library.ml_questionnaire.visualization import plot_roc_with_threshold_cms_grid, compute_feature_importance, compute_ppv_table
from library.ml_questionnaire.scoring import compute_auc_by_cohort_and_fold

# %% Main
if __name__ == "__main__":
    # %% data and path
    df = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    # df[['diagnosis', 'cohort']].value_counts()
    df = df.loc[df['has_quest'] == 1, :]
    target = 'diagnosis'
    # df = df.loc[df.cohort == 'Stanford', :]

    df.reset_index(inplace=True, drop=True)
    # result_dir = config.get('results_path').get('results').joinpath('ml_quest_feat_impt')
    result_dir = config.get('results_path').get('results').joinpath('ml_questionnaire_minerva_adrc_filter')
    result_dir.mkdir(parents=True, exist_ok=True)
    # %% input
    random_seed = 42  # Random seed for reproducibility
    # Cross-validation parameters (for evaluation, optimization, and fine-tuning)
    cv_folds_eval = 10  # Folds for evaluating models
    cv_folds_optimization = 5  # Folds used in cross_val_score during optimization

    # Optuna optimization parameters
    optuna_trials = 250# 250  # Number of trials to run during optimization

    model_types = [
                    "random_forest",
                   "lightgbm",
                    "xgboost_sens",
                   "xgboost",
                   "elastic_net"
    ]

    optuna_sampler = optuna.samplers.TPESampler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    #%% Data splits and features
    # Define the feature columns (using 11 Q's)
    feature_cols = [c for c in df.columns if c.startswith('q')]

    # X_df = df[['subject_id'] + feature_cols]
    # y = df[target]
    # %%
    if not(result_dir.joinpath('df_outer_metrics_records_ci.csv').exists() and
           result_dir.joinpath('predictions_outer_folds.csv').exists()):

        (df_metrics_ci,
         df_predictions,
         df_inner_val_records) =  run_nested_cv_with_optuna_parallel(
            df=df,
            target_col=target,
            continuous_cols=None,
            feature_cols=feature_cols,
            col_id='subject_id',  # this will be dropped from the data matrix
            model_types=model_types,
            n_outer_splits=cv_folds_eval,
            n_inner_splits=cv_folds_optimization,
            n_trials=optuna_trials,
            pos_weight=True,
            study_sampler=optuna_sampler,
            maximize="sens",
            results_dir=result_dir,
        )

    else:
        df_metrics_ci = pd.read_csv(result_dir.joinpath(f'df_outer_metrics_records_ci.csv'))
        df_predictions = pd.read_csv(result_dir.joinpath(f'predictions_outer_folds.csv'))
        # df_inner_val_records = pd.read_csv(result_dir.joinpath(f'metrics_outer_folds.csv'))

    # %%
    result_dir_visualization = result_dir.joinpath('visualization')
    result_dir_visualization.mkdir(parents=True, exist_ok=True)
    # 1. report across all the models
    for model in model_types:
        plot_roc_with_threshold_cms_grid(df_predictions=df_predictions.loc[df_predictions['model_type'] == model],
                                    df_metrics_ci=df_metrics_ci.loc[df_metrics_ci['model_type'] == model],
                                    class_names=  {0: "CN", 1: "iRBD"},
                                    model_type=model,
                                    title=f'{model.capitalize().replace("_", " ")} Questionnaire Based-Model Subject Level',
                                    scale=1.1,
                                    figsize=(16, 14),
                                    output_path=result_dir_visualization.joinpath(f'roc_cm_model_{model}.png'))


    # 2. Report the one with highest metric
    optimization = 'youden'
    model_type = 'xgboost'
    df_metrics_ci_best = df_metrics_ci.loc[(df_metrics_ci['optimization'] == optimization) &
                                           (df_metrics_ci['model_type'] == model_type), :]
    df_predictions_best = df_predictions.loc[(df_predictions['optimization'] == optimization) &
                                             (df_predictions['model_type'] == model_type), :]
    plot_roc_with_threshold_cms_grid(df_predictions=df_predictions_best,
                                df_metrics_ci=df_metrics_ci_best,
                                class_names=  {0: "CL", 1: "iRBD"},
                                model_type='xgboost',
                                title='Questionnaire Based-Model',
                                scale=1.2,
                                 figsize=(16, 5),
                                # output_path=result_dir.joinpath(f'best_model_{model_type}_{optimization}.png'),
                                 output_path=result_dir_visualization.joinpath(f'single_roc_cm_model_{model_types}_{optimization}.png'))

    ppv_table = compute_ppv_table(df_predictions, df_metrics_ci, model_type="xgboost", prevalence=0.015)
    print(ppv_table)

    round(ppv_table.ppv_adj.mean(), 1)
    round(ppv_table.ppv_adj.std(), 1)

    round(ppv_table.ppv.mean(), 1)
    round(ppv_table.ppv.std(), 1)


    # %% Validation metrics separate cohorts
    df_predictions_best_cohort = pd.merge(left=df_predictions_best,
                                          right=df[['subject_id', 'cohort']],
                                          on='subject_id',
                                          how='left')
    df_predictions_best_cohort = df_predictions_best_cohort.loc[df_predictions_best_cohort['threshold_type'] == 'youden', :]

    df_res_cross_cohort, df_summary_cross_cohort = compute_auc_by_cohort_and_fold(df_predictions_best_cohort)

    # %% Evalaute xgboost
    compute_feature_importance(models_dir=result_dir.joinpath('folds'),
                               features_names=feature_cols,
                               objective="youden",
                               top_n=15)



    #
    from tabulate import  tabulate

    df_counts = (
        df.loc[df.age > 80, ['diagnosis', 'cohort', 'has_quest', 'actig']]
        .value_counts()
        .reset_index(name="count")  # move multi-index into columns
        .sort_values("count", ascending=False)  # optional: sort by frequency
    )

    # Pretty table
    print(
        tabulate(
            df_counts,
            headers="keys",  # automatically picks column names
            tablefmt="psql",
            showindex=False  # cleaner, don’t show the DataFrame index
        )
    )


    # %% supplementary table best model
    df_best_models = pd.DataFrame()
    for model in model_types:
        df_model = df_metrics_ci.loc[df_metrics_ci['model_type'] == model, :]
        df_model_best_auc = df_model.loc[df_model['auc_score'] == df_model['auc_score'].max(),
        ['model_type', 'threshold', 'auc_score_ci', 'sensitivity_ci', 'specificity_ci']]
        df_best_models = pd.concat([df_best_models, df_model_best_auc])













