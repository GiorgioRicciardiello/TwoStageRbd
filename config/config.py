"""
Author: Giorgio Ricciardiello
        giocrm@stanford.edu
configurations parameters for the paths
"""
import pathlib

# import shutil

# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]
# Define raw data path
data_path = root_path.joinpath('data')
# data paths
data_raw_path = data_path.joinpath('raw')
data_pp_path = data_path.joinpath('pp')
data_ukbb_path = data_path.joinpath('ukbb')
# results path
data_res = root_path.joinpath('results')




# Construct the config dictionary with nested templates
config = {
    'root_path': root_path,
    'data_path': {
        'data': data_path,
        # raw files
        'raw': data_raw_path,
        # 'raw_questionnaire': data_raw_path.joinpath('shas_clinic_stanford_4Qs.xlsx'),
        # 'raw_questionnaire': data_raw_path.joinpath('FinalQuestionnaireData.csv'),
        # %% Sinai
        'raw_questionnaire': data_raw_path.joinpath('FinalQuestionnaireData.xlsx'),
        # 'raw_dem_sinai': data_raw_path.joinpath('DemographicsAugus5.csv'),
        # %% Stf
        'raw_dem_stanford': data_raw_path.joinpath('AxDemographicsStanford.xlsx'),
        'raw_dem_race_stanford': data_raw_path.joinpath('ax_stanford_race_ethnicity_fields.xlsx'),
        # %% Acrigraphy
        'raw_actigraphy': data_raw_path.joinpath('nightly_features_qc3.csv'),
        # 'raw_act_features': data_raw_path.joinpath('nightly_features_qc3.csv'),
        # %% VASCBrain
         'raw_vasc': data_raw_path.joinpath('vascbrain_records.xlsx'),
        # pre-process files
        'pp_stf_adrc': data_pp_path.joinpath('nightly_features_adrc.csv'),
        # 'pp_questionnaire': data_pp_path.joinpath('pp_questionnaire.csv'),
        'pp_questionnaire': data_pp_path.joinpath('pp_questionnaire_manual_edits.csv'),
        'pp_actig': data_pp_path.joinpath('pp_nightly_features_qc3.csv'),

        'pp_dem_only_act': data_pp_path.joinpath('pp_dem_only_act.csv'),  # demographis of subjects only in the actig data
    },
    # results path
    'results_path': {
        'results': data_res,
        'full_cross_val': data_res.joinpath('full_cross_val'),
        'table_one': data_res.joinpath('table_one'),
        # predictions of the actigraphy mode results
        'predictions_actigraphy': data_res.joinpath('ml_actigraphy', 'xgboost_nightly_features_nested_cv_predictions.csv'),
        'predictions_questionnaire': data_res.joinpath('ml_questionnaire', 'predictions_outer_folds.csv'),
        'two_stage': data_res.joinpath('two_stage'),
    }
}


config_actigraphy = {
    'data': data_raw_path.joinpath('nightly_features_qc3.csv'),
    'pp_actig': config.get('data_path').get('pp_actig'),
    'pp_actig_adrc': config.get('data_path').get('pp_stf_adrc'),
    'raw_adrc': data_raw_path.joinpath('nightly_features_stanford_ADRC_subjects_78.csv'),
    'raw_actigraphy_adrc_wlabels': data_raw_path.joinpath('nightly_features_stanford_ADRC_subjects_78_with_labels.csv'),
    'results_dir': data_res.joinpath('actigraphy_cv'),

    'pp_actig_merged': data_pp_path.joinpath('nightly_features_qc3_merged.csv'),
}



























