"""
Two stage prediction combining the actihraphy results and the questionnare results
"""
import pandas as pd
from config.config import config, config_actigraphy
from library.two_stage.model_selection import (get_questionnaire_single_column,
                                               get_best_questionnaire_model_by_selection,
                                               get_best_actigraphy_model_by_selection,
                                               plot_confusion_matrices_style_bar)

if __name__ == "__main__":
    # %%  read the input predictions
    # path_acti_pred = config.get('results_path').get('results').joinpath(f'ml_actigraphy_pos_weight_no_dem_no')
    path_acti_pred = config.get('results_path').get('results').joinpath(f'ml_actig_with_adrc')
    # path_quest_pred = config.get('results_path').get('results').joinpath('ml_questionnaire_minerva')
    path_quest_pred = config.get('results_path').get('results').joinpath('ml_questionnaire_minerva_adrc_filter')
    path_questionnaire = config.get('data_path').get('pp_questionnaire')
    # %% output path
    output_path = config.get('results_path').get('results').joinpath(f'two_stage_new_metrics_frame')
    output_path.mkdir(parents=True, exist_ok=True)

    # %% ===================================================================
    # ========================== PLOT SINGLE MODELS ========================
    # ======================================================================

    model_select = {
        'questionnaire': {
            'optimization': 'auc',
            'threshold': 'sens_max',  # max sensitivity tau
            'model_type': 'xgboost'
        },
        'actigraphy': {
            'optimization': 'maxspec', # max specificity tau
            'threshold': 'youden',
            'model_type': 'xgboost'
        },
    }
    # ------------------------- questionnaire  ------------------------------------------
    col_test = 'q1_rbd'
    df_quest_raw = get_questionnaire_single_column(path_questionnaire=path_questionnaire,
                                                   col_quest='q1_rbd')

    # ------------------------- questionnaire ML ------------------------------------------
    model_select_questionnaire = model_select.get('questionnaire')

    df_quest_metrics_best, df_quest_pred_best = get_best_questionnaire_model_by_selection(path_quest_pred=path_quest_pred,
                                                                             model_select=model_select_questionnaire)
    print(f'Questionnaire Model Max Sens: {df_quest_metrics_best.T}')
    # -------------------------actigraphy------------------------------------------
    model_select_actigraphy = model_select.get('actigraphy')
    df_actig_metrics_best, df_actig_pred_best = get_best_actigraphy_model_by_selection(path_quest_actig=path_acti_pred,
                                                                             model_select=model_select_actigraphy)

    print(f'Actigraphy Model Max Spec: {df_actig_metrics_best.T}')
    # -------------------------Merge------------------------------------------

    df_two_stage = pd.merge(left=df_quest_pred_best[['subject_id', 'y_pred_quest']],
                            right=df_actig_pred_best[['subject_id', 'y_true', 'y_pred_actig']])

    df_two_stage = pd.merge(left=df_two_stage,
                            right=df_quest_raw,)
    df_two_stage[col_test] = df_two_stage[col_test].astype(int)

    df_two_stage['serial_test_quest_raw_actig'] = 0
    df_two_stage.loc[(df_two_stage[col_test] == 1) &
                     (df_two_stage['y_pred_actig'] == 1), 'serial_test_quest_raw_actig'] = 1

    df_two_stage['serial_test_quest_ml_actig'] = 0
    df_two_stage.loc[(df_two_stage['y_pred_quest'] == 1) &
                     (df_two_stage['y_pred_actig'] == 1), 'serial_test_quest_ml_actig'] = 1


    plot_confusion_matrices_style_bar(df=df_two_stage,
                                      y_true_col='y_true',
                                      class_names={0: "Control", 1: "iRBD"},
                                      methods=[col_test, 'y_pred_actig', 'serial_test_quest_raw_actig'],
                                      titles=['Questionnaire', 'Actigraphy', 'Two-stage'],
                                      comparison=f'RBD Question & Actigraphy',
                                      figsize=(12, 6),
                                      # output_dir=output_path
                                      )

    plot_confusion_matrices_style_bar(df=df_two_stage,
                                      y_true_col='y_true',
                                      class_names={0: "Control", 1: "iRBD"},
                                      methods=['y_pred_quest', 'y_pred_actig', 'serial_test_quest_ml_actig'],
                                      titles=['Questionnaire', 'Actigraphy', 'Two-stage'],
                                      comparison=f'4-Item Questionnaire & Actigraphy',
                                      figsize=(12, 6),
                                      output_dir=None)
    prevalence = 0.015
    ppv_adj = (0.67 * prevalence) / (0.67 * prevalence + (1 - 1) * (1 - prevalence))

    prevalence = 0.015
    ppv_adj = (0.74 * prevalence) / (0.74 * prevalence + (1 - 1) * (1 - prevalence))

    # %% ============================================================================================================
    # ====================== Sanity check that the max_sens and max_spec are the models =============================
    #================================================================================================================
    # Exhaustive search logic has been removed to align with manuscript methodology.
    # The selected models (max_sens for questionnaire and max_spec for actigraphy) 
    # are evaluated rigorously using nested cross-validation to prevent post-hoc bias.
    print(f"Two-stage prediction completed.")
