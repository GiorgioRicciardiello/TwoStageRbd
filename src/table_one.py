"""
Compute the table with hypothesis testing
"""
import pathlib

import pandas as pd
from config.config import config
from library.epidemiology.table_one import MakeTableOne
from library.epidemiology.hypothesis_testing import stats_test
from typing import Dict, List, Optional
from tabulate import tabulate


def compute_table_comparison_wrapper(df: pd.DataFrame,
                             strata_cohort: str,
                             strata_diagnosis: str,
                             columns: Dict[str, List[str]],
                             strata_study: Optional[str] = None,
                             table_name: str = None,
                             output_path: pathlib.Path = None) -> pd.DataFrame:
    """
    Compute descriptive tables and hypothesis testing for:
    1) Entire cohort distribution,
    2) Cohort comparison (if >1 cohort),
    3) Diagnosis comparison (if >1 group).

    Returns a combined DataFrame with descriptive stats + p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    strata_cohort : str
        Column name defining cohort strata (e.g., study site).
    strata_diagnosis : str
        Column name defining diagnosis strata (e.g., case/control).
    columns : dict
        Dictionary with keys {'categorical': [...], 'continuous': [...]}
    table_name : str, optional
        Filename for saving output.
    output_path : pathlib.Path, optional
        Directory to save the output.
    """

    # ---------- Inner helpers ----------
    # def _align(df_other):
    #     """Reindex to match the base variable list"""
    #     if df_other.empty:
    #         return pd.DataFrame(index=variables).reset_index().rename(columns={"index": "variable"})
    #     return df_other.set_index("variable").reindex(variables).reset_index()

    def _compute_hypothesis_test(df: pd.DataFrame,
                                 categorical_var: List[str],
                                 continuous_var: List[str],
                                 strata: str) -> pd.DataFrame:
        """Compute hypothesis tests (binary, continuous, ordinal) and return p-values + method."""
        # Binary vs ordinal split
        binary_var = [col for col in categorical_var if set(df[col].dropna().unique()) <= {0, 1}]
        ordinal_var = [col for col in categorical_var if col not in binary_var]
        # Run stats tests via dispatcher
        df_stats_bin = stats_test(data=df, columns=binary_var,
                                  var_type='binary', strata_col=strata, SHOW=True)
        df_stats_cont = stats_test(data=df, columns=continuous_var,
                                   var_type='continuous', strata_col=strata, SHOW=True)
        df_stats_ord = stats_test(data=df, columns=ordinal_var,
                                  var_type='ordinal', strata_col=strata, SHOW=True)

        # Subset to keep only relevant info
        df_stats_bin = df_stats_bin[['Variable', 'p-value formatted', 'Stat Method']]
        df_stats_cont = df_stats_cont[['Variable', 'p-value formatted', 'Stat Method']]
        df_stats_ord = df_stats_ord[['Variable', 'p-value formatted', 'Stat Method']].drop_duplicates()

        return pd.concat([df_stats_bin, df_stats_cont, df_stats_ord], axis=0)

    def _safe_hypothesis_test(df, categorical_var, continuous_var, strata):
        """Wrapper: only run tests if there is more than 1 unique group."""
        if strata not in df.columns or df[strata].nunique() <= 1:
            return pd.DataFrame(columns=["Variable", "p-value formatted", "Stat Method"])
        return _compute_hypothesis_test(df, categorical_var, continuous_var, strata)

    def _safe_make_table(df, continuous_var, categorical_var, strata=None):
        """Wrapper: only stratify if >1 group, otherwise return empty frame."""
        if strata and (strata not in df.columns or df[strata].nunique() <= 1):
            return pd.DataFrame(columns=["variable"])
        tab = MakeTableOne(df,
                           continuous_var=continuous_var,
                           categorical_var=categorical_var,
                           strata=strata)
        t = tab.create_table()
        return tab.group_variables_table(t, show=False)

    # ---------- Pre-processing ----------

    categorical_var = [col for col in columns['categorical'] if col in df.columns]
    continuous_var = [col for col in columns['continuous'] if col in df.columns]

    # ---------- Table 1: Overall cohort ----------
    tab_dist = MakeTableOne(df,
                            continuous_var=continuous_var,
                            categorical_var=categorical_var)
    df_tab_single_dist = tab_dist.create_table()
    df_tab_single_dist = tab_dist.group_variables_table(df_tab_single_dist)

    # ---------- Table 2: Between cohorts ----------
    df_tab_cohort = _safe_make_table(df, continuous_var, categorical_var, strata_cohort)
    if df[strata_cohort].nunique() <= 2:
        df_test_cohort = _safe_hypothesis_test(df, categorical_var, continuous_var, strata_cohort)
        df_tab_cohort_dist_stats = (
            df_tab_cohort.merge(df_test_cohort, left_on="variable", right_on="Variable", how="left")
            .drop(columns=["Variable"], errors="ignore")
        )
        df_tab_cohort_dist_stats.rename(columns={'p-value formatted': f'p-value (Cohort)'}, inplace=True)
    else:
        # do not compute p values and stat test when the cohorts are  > 2
        df_tab_cohort_dist_stats = df_tab_cohort

    # ---------- Table 3: Between diagnosis ----------
    df_tab_diag = _safe_make_table(df, continuous_var, categorical_var, strata_diagnosis)
    df_test_diag = _safe_hypothesis_test(df, categorical_var, continuous_var, strata_diagnosis)
    df_tab_diag_dist_stats = (
        df_tab_diag.merge(df_test_diag, left_on="variable", right_on="Variable", how="left")
        .drop(columns=["Variable"], errors="ignore")
    )
    df_tab_diag_dist_stats.rename(columns={'p-value formatted': f'p-value (Diagnosis)'}, inplace=True)

    compute_study_test = False
    # ---------- Table 3: Between Actigraphy ----------
    if strata_study and df[strata_study].nunique() > 1:
        compute_study_test = True
        df_tab_study = _safe_make_table(df, continuous_var, categorical_var, strata_study)
        # df_test_study = _safe_hypothesis_test(df, categorical_var, continuous_var, strata_study)
        #
        # df_tab_study_dist_stats = (
        #     df_tab_study.merge(df_test_study, left_on="variable", right_on="Variable", how="left")
        #     .drop(columns=["Variable"], errors="ignore")
        # )
        # # df_tab_study_dist_stats = _align(df_tab_study_dist_stats)
        # df_tab_study_dist_stats = df_tab_study_dist_stats
        # df_tab_study_dist_stats.rename(columns={'p-value formatted': f'p-value (Actigraphy)'}, inplace=True)


    # ---------- Align and combine ----------
    # variables = df_tab_single_dist["variable"]

    tables_to_concat = [
        # _align(df_tab_single_dist),
        # _align(df_tab_diag_dist_stats),
        # _align(df_tab_cohort_dist_stats),
        # df_tab_single_dist,
        df_tab_diag_dist_stats,
        df_tab_cohort_dist_stats,
    ]

    if compute_study_test:
        tables_to_concat.append(df_tab_study)
        # tables_to_concat.append(df_tab_study_dist_stats)

    # Final concatenated table
    big_table = pd.concat(tables_to_concat, axis=1)

    # clean the table for the publication
    # big_table.rename(columns={'p-value formatted': 'p-value'}, inplace=True)
    # Keep only the first occurrence of 'variable' and 'p-value' if duplicated
    cols = big_table.columns
    mask = ~((cols.duplicated()) & (cols.isin(["variable", "Stat Method"])))
    big_table = big_table.loc[:, mask]

    # identify the rows in 'variable' column that are questionnaire items (start with 'q')
    mask_q = big_table["variable"].astype(str).str.startswith("q")
    q_indices = big_table.index[mask_q]
    mapping = {'0': "No", '1': "Do not know", '2': "Yes"}

    for idx in q_indices:
        # look at the next 3 rows after the question
        next_rows = big_table.loc[idx + 1:idx + 3, "variable"]
        # replace if numeric 0/1/2
        big_table.loc[idx + 1:idx + 3, "variable"] = next_rows.replace(mapping)

    # big_table = big_table.loc[big_table['variable'] != '0']

    # Move "Stat Method" column to the end
    if "Stat Method" in big_table.columns:
        cols = [c for c in big_table.columns if c != "Stat Method"] + ["Stat Method"]
        big_table = big_table[cols]


    # mask_yes = big_table['variable'].astype(str).str.startswith("1")
    # yes_indices = big_table.index[mask_yes]
    # for idx in yes_indices:
    #

    # # sawp the no, do not know and yes rows to yes, do not know
    # big_table["variable"] = pd.Categorical(
    #     big_table["variable"],
    #     categories=['Yes', 'Do not know', 'No'],  # desired order
    #     ordered=True
    # )
    # big_table = big_table.sort_values(["variable"], kind="stable").reset_index(drop=True)
    #

    def reorder_batches(df):
        # desired order for answer rows
        order = ["Yes", "Do not know", "No"]

        rows = []
        i = 0
        while i < len(df):
            if df.loc[i, "variable"] == "No" and i + 2 < len(df):
                batch = df.iloc[i:i + 3]  # No, Do not know, Yes block
                # Reorder the batch
                batch = batch.set_index("variable").loc[order].reset_index()
                rows.append(batch)
                i += 3
            else:
                rows.append(df.iloc[[i]])
                i += 1

        return pd.concat(rows, ignore_index=True)

    # usage
    try:
        big_table = reorder_batches(df=big_table)

    except KeyError:
        # If any label missing, just keep original
        pass

    mapper_var = {
        'Count': 'Count',
        'age': 'Age',
        'bmi': 'BMI',
        'q1_rbd': 'Q1 RBD',
        'No': '   No',
        'Do not know': '   Do not know',
        'Yes': '   Yes',
        'q2_smell': 'Q2 Smell',
        'q4_constipation': 'Q3 Constipation',
        'q5_orthostasis': 'Q4 Orthostasis',
        'gender': 'Gender',
        # '1': '   1',
        'actig': 'Actigraphy',
        # '1'
        'race_num': 'Race',

    }
    big_table['variable'] = big_table['variable'].replace(mapper_var)

    # ----------- Case specifci for our project
    race_numeric_mapper = {
        "White": 0,
        "Black or African American": 1,
        'Asian': 3,
        'Other': 4,
        'Mixed':5,
        }
    race_numeric_mapper_inv = {str(val): key for key, val in race_numeric_mapper.items()}
    idx_race = big_table.loc[big_table['variable'] == 'Race',].index
    if not idx_race is None:
        idx_race = int(idx_race[0]) + 1
        for race_row in range(idx_race, len(race_numeric_mapper)+idx_race):
            row_val =  big_table.loc[race_row, 'variable']
            new_val = race_numeric_mapper_inv.get(row_val)
            print(f'{row_val} -> {new_val}')
            if not new_val is None:
                big_table.loc[race_row, 'variable'] = f'   {new_val}'

    big_table.replace("nan±nan(0)", "-", inplace=True)
    big_table.fillna("-", inplace=True)


    # ---------- Output ----------
    print(tabulate(big_table,
                   headers=[*big_table.columns],
                   showindex=False,
                   tablefmt="fancy_grid"))

    if output_path and table_name:
        big_table.to_excel(output_path.joinpath(f'{table_name}.xlsx'), index=False)

    return big_table




def compute_table_comparison(df: pd.DataFrame,
                             stratas: Dict[str, str],
                             columns: Dict[str, List[str]],
                             table_name: str = None,
                             output_path: pathlib.Path = None) -> pd.DataFrame:
    """
    Compute descriptive tables and hypothesis testing for:
    1) Entire cohort distribution,
    2) Any number of additional strata (e.g., cohort, diagnosis, study).

    Returns a combined DataFrame with descriptive stats + p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratas : dict
        Dictionary mapping a *label* (e.g., "Cohort", "Diagnosis")
        to the column name in df. Example:
            {"Cohort": "cohort_col",
             "Diagnosis": "dx_col",
             "Study": "study_site"}
    columns : dict
        Dictionary with keys {'categorical': [...], 'continuous': [...]}
    table_name : str, optional
        Filename for saving output.
    output_path : pathlib.Path, optional
        Directory to save the output.
    """

    # ---------- Inner helpers ----------

    def _align(df_other, variables):
        """Reindex to match the base variable list"""
        if df_other.empty:
            return pd.DataFrame(index=variables).reset_index().rename(columns={"index": "variable"})
        return df_other.set_index("variable").reindex(variables).reset_index()

    def _compute_hypothesis_test(df: pd.DataFrame,
                                 categorical_var: List[str],
                                 continuous_var: List[str],
                                 strata: str) -> pd.DataFrame:
        """Compute hypothesis tests (binary, continuous, ordinal) and return p-values + method."""
        # Binary vs ordinal split
        binary_var = [col for col in categorical_var if set(df[col].dropna().unique()) <= {0, 1}]
        ordinal_var = [col for col in categorical_var if col not in binary_var]

        # Run stats tests via dispatcher
        df_stats_bin = stats_test(data=df, columns=binary_var,
                                  var_type='binary', strata_col=strata, SHOW=True)
        df_stats_cont = stats_test(data=df, columns=continuous_var,
                                   var_type='continuous', strata_col=strata, SHOW=True)
        df_stats_ord = stats_test(data=df, columns=ordinal_var,
                                  var_type='ordinal', strata_col=strata, SHOW=True)

        # Subset to keep only relevant info
        df_stats_bin = df_stats_bin[['Variable', 'p-value formatted', 'Stat Method']]
        df_stats_cont = df_stats_cont[['Variable', 'p-value formatted', 'Stat Method']]
        df_stats_ord = df_stats_ord[['Variable', 'p-value formatted', 'Stat Method']].drop_duplicates()

        return pd.concat([df_stats_bin, df_stats_cont, df_stats_ord], axis=0)

    def _safe_hypothesis_test(df, categorical_var, continuous_var, strata):
        """Wrapper: only run tests if there is more than 1 unique group."""
        if strata not in df.columns or df[strata].nunique() <= 1:
            return pd.DataFrame(columns=["Variable", "p-value formatted", "Stat Method"])
        return _compute_hypothesis_test(df, categorical_var, continuous_var, strata)

    def _safe_make_table(df, continuous_var, categorical_var, strata=None):
        """Wrapper: only stratify if >1 group, otherwise return empty frame."""
        if strata and (strata not in df.columns or df[strata].nunique() <= 1):
            return pd.DataFrame(columns=["variable"])
        tab = MakeTableOne(df,
                           continuous_var=continuous_var,
                           categorical_var=categorical_var,
                           strata=strata)
        t = tab.create_table()
        return tab.group_variables_table(t, show=False)

    # ---------- Pre-processing ----------

    categorical_var = [col for col in columns['categorical'] if col in df.columns]
    continuous_var = [col for col in columns['continuous'] if col in df.columns]

    # ---------- Table 1: Overall cohort ----------
    tab_dist = MakeTableOne(df,
                            continuous_var=continuous_var,
                            categorical_var=categorical_var)
    df_tab_single_dist = tab_dist.create_table()
    df_tab_single_dist = tab_dist.group_variables_table(df_tab_single_dist)

    # Variables reference
    variables = df_tab_single_dist["variable"]

    # ---------- Loop through all strata ----------
    strata_results = []
    for strata_label, strata_col in stratas.items():
        df_tab_strata = _safe_make_table(df, continuous_var, categorical_var, strata_col)
        df_test_strata = _safe_hypothesis_test(df, categorical_var, continuous_var, strata_col)

        df_tab_strata_stats = (
            df_tab_strata.merge(df_test_strata,
                                left_on="variable",
                                right_on="Variable",
                                how="left")
            .drop(columns=["Variable"], errors="ignore")
        )
        # df_tab_strata_stats = _align(df_tab_strata_stats, variables)

        # Rename columns to avoid collisions
        df_tab_strata_stats = df_tab_strata_stats.add_suffix(f"_{strata_label.lower()}")
        df_tab_strata_stats = df_tab_strata_stats.rename(
            columns={f"variable_{strata_label.lower()}": "variable"}
        )

        strata_results.append(df_tab_strata_stats)

    # ---------- Combine ----------
    big_table = pd.concat(
        [df_tab_single_dist] + strata_results,
        axis=1
    )
    # ---------- Output ----------
    print(tabulate(big_table,
                   headers=[*big_table.columns],
                   showindex=False,
                   tablefmt="fancy_grid"))

    if output_path and table_name:
        big_table.to_excel(output_path.joinpath(f"{table_name}.xlsx"), index=False)

    return big_table



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


# %% Main
if __name__ == "__main__":
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    strata_diagnosis = 'diagnosis'
    strata_cohort = 'cohort'
    strata_study = 'actig'
    strata_study_lbl = strata_study + '_lbl'
    # %% Cohort selection (questionnaire only, questionnaire and actigrapht)
    # cohort = 'quest_actigraphy'  # 'quest_actigraphy' #  'quest_actigraphy'
    # if cohort == 'quest_actigraphy':
    #     df_data = df_data[df_data['actig'] == 1]
    #     print(f'Selecting only quest + actig {df_data.shape[0]}')
    # else:
    #     print(f'Selecting all questionnaires data {df_data.shape[0]}')

    df_data.loc[df_data[strata_study] == 0, strata_diagnosis].value_counts()

    visualize_table(df=df_data, group_by=[strata_cohort, 'has_quest', 'actig'])

    # %%
    import pandas as pd
    from pathlib import Path

    # Input and output paths
    input_path = Path(r"C:\Users\giorg\Downloads\ActigraphySG_ImportTemplate_2025-03-11_descriptions (1).csv")
    output_path = Path(r"C:\Users\giorg\Downloads\Actigraphy spreadsheet proper format wide.csv")

    # Read with correct separator
    df = pd.read_csv(input_path, sep=";")

    # First row = description, second row = values
    descriptions = df.iloc[0]
    values = df.iloc[1]

    # Build long-format dataframe
    tidy = pd.DataFrame({
        "variable": df.columns,
        "description": descriptions.values,
        "values": values.values
    })

    # Save as proper CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Converted REDCap file saved to: {output_path}")


    # %% output_path
    output_path = config.get('results_path').get('table_one')
    output_path.mkdir(parents=True, exist_ok=True)
    # output_path = output_path.joinpath(f'{cohort}')

    # if not output_path.is_dir():
    #     raise NotADirectoryError(f"{output_path} is not a directory")
    # if any(output_path.iterdir()):  # iterdir() yields contents; 'any()' means not empty
    #     raise OSError(f"Directory {output_path} is not empty. Do you want to continue?")
    # %% Count for image 1 of the paper
    # How many questionnaires and actigraphy we have per cohorts
    df_data.loc[df_data['has_quest'] == 1, 'cohort'].value_counts()
    # how many cases and controls per cohort
    df_counts = df_data[['cohort', 'diagnosis']].value_counts().reset_index(name='count')
    df_counts_sorted = df_counts.sort_values(by=['cohort', 'diagnosis'])
    print(df_counts_sorted)
    # how many with questonnaire
    df_data.loc[df_data['has_quest'] == 1, 'diagnosis'].value_counts()
    df_data.loc[df_data['actig'] == 1, 'diagnosis'].value_counts()

    df_two_stages = df_data.loc[(df_data['actig'] == 1) & (df_data['has_quest'] == 1), 'diagnosis'].value_counts()


    # %% label the groups
    mapper_diagnosis = {
        0: 'Control',
        1: 'iRBD'
    }

    df_data[strata_diagnosis] =  df_data[strata_diagnosis].map(mapper_diagnosis)

    mapper_study = {
        0: 'No',
        1: 'Yes'
    }
    df_data[strata_study_lbl] = df_data[strata_study].map(mapper_study)

    # mapper_study = {
    #     0: 'No',
    #     1: 'Yes'
    # }

    # mapper_cohort = {
    #     'SHAS': 0,
    #     'Stanford': 1
    # }
    # df_data[strata_cohort] = df_data[strata_cohort].map(mapper_cohort)

    # Define the mapping
    mapper_responses = {
        0.0: 0,  # No
        0.5: 1,  # Do not know
        1.0: 2  # Yes
    }

    col_questions = [c for c in df_data.columns if c.startswith('q')]
    df_data[col_questions] = df_data[col_questions].replace(mapper_responses)

    # %% columns to evaluate
    col_questions = [c for c in df_data.columns if c.startswith('q')]
    continuous_var = [c for c in ['age', 'bmi', 'TST', 'WASO', 'SE', 'T_avg', 'nw_night',] if c in df_data.columns]
    categorical_var = col_questions
    categorical_var += [c for c in ['race_num', 'gender', strata_study] if c in df_data.columns]
    columns = {
        'categorical': categorical_var,
        'continuous': continuous_var,
    }
    len(categorical_var) + len(continuous_var)
    # %% compute table comparison and stat test
    stratas = {
        "Cohort": strata_cohort,
        "Diagnosis": strata_diagnosis,
        "Study": strata_study_lbl,
    }

    # big_table = compute_table_comparison(
    #     df=df_data,
    #     stratas=stratas,
    #     columns=columns,
    #     table_name="table_one",
    #     output_path=output_path
    # )



    # %% seprate tablessplit the dataframes for each table
    # QUEST = YES
    df_quest_yes_actig_yes = df_data[(df_data.has_quest == 1) & (df_data.actig == 1)].copy()
    df_quest_yes_actig_no = df_data[(df_data.has_quest == 1) & (df_data.actig == 0)].copy()
    df_quest = df_data[(df_data.has_quest == 1)].copy()

    # QUEST = NO
    df_quest_no_actig_yes =  df_data[(df_data.has_quest == 0) & (df_data.actig == 1)].copy()
    df_act = df_data[(df_data.actig == 1)].copy()




    df_study_comparison = df_data.copy()

    # Table of all the data
    compute_table_comparison_wrapper(df=df_data.copy(),
                                    strata_cohort=strata_cohort,
                                    strata_diagnosis=strata_diagnosis,
                                    strata_study=strata_study,
                                    columns=columns,
                                    table_name='AllData',
                                    output_path=output_path)

    # Table Questionnaire Model
    compute_table_comparison_wrapper(df=df_quest.copy(),
                             strata_cohort=strata_cohort,
                             strata_diagnosis=strata_diagnosis,
                             # strata_study=strata_study,
                             columns=columns,
                             table_name='QuestYesActigYesOrNo',
                             output_path=output_path)

    # Table Actigraphy Model
    compute_table_comparison_wrapper(df=df_act.copy(),
                             strata_cohort=strata_cohort,
                             strata_diagnosis=strata_diagnosis,
                             strata_study=strata_study,
                             columns=columns,
                             table_name='ActigYesQuestYesOrNo',
                             output_path=output_path)


    # all subjects with questionnaire
    compute_table_comparison_wrapper(df=df_quest_yes_actig_no.copy(),
                             strata_cohort=strata_cohort,
                             strata_diagnosis=strata_diagnosis,
                            strata_study=strata_study,
                             columns=columns,
                             table_name='QuestYesActigNo',
                             output_path=output_path)

    # all subjects with questionnaire and actigraphy
    compute_table_comparison_wrapper(df=df_quest_yes_actig_yes.copy(),
                             strata_cohort=strata_cohort,
                             strata_diagnosis=strata_diagnosis,
                             strata_study=strata_study,
                             columns=columns,
                             table_name='QuestYesActigYes',
                             output_path=output_path)

    # all subjects with actigraphy no matter their questionnaire status

    compute_table_comparison_wrapper(df=df_quest_no_actig_yes.copy(),
                             strata_cohort=strata_cohort,
                             strata_diagnosis=strata_diagnosis,
                             strata_study=strata_study,
                             columns=columns,
                             table_name='QuestNoActigYes',
                             output_path=output_path)





    # %% Extra notes
    # diagnosis
    for cohort in df_data[strata_cohort].unique():
        df_counts = (
            df_data.loc[df_data[strata_cohort] == cohort, "diagnosis"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "diagnosis", "diagnosis": "count"})
        )
        df_counts.columns = ["diagnosis", "count"]  # enforce correct names

        total = df_data.loc[df_data[strata_cohort] == cohort].shape[0]
        df_counts["percentage"] = (df_counts["count"] * 100 / total).round(3)

        print(f"\nCohort: {cohort}")
        print(df_counts)

    # questionnaire
    for cohort in df_data[strata_cohort].unique():
        df_counts = (
            df_data.loc[df_data[strata_cohort] == cohort, "has_quest"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Has Quest", "diagnosis": "count"})
        )

        df_counts.columns = ["Has Quest", "count"]  # enforce correct names

        total = df_data.loc[df_data[strata_cohort] == cohort].shape[0]
        df_counts["percentage"] = (df_counts["count"] * 100 / total).round(3)

        print(f"\nCohort: {cohort}")
        print(df_counts)


    for val in [14, 47,45, 1]:
        print(f'{val} : {round((val * 100) / 107, 2)}')

    df_data.loc[df_data[strata_diagnosis] == 0, 'gender'].value_counts()
    print(f'{val} : {round((124 * 100) / 237, 2)}')

    for cohort in df_data[strata_diagnosis].unique():
        df_counts = (
            df_data.loc[df_data[strata_cohort] == cohort, "gender"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "gender", "gender": "count"})
        )
        df_counts.columns = ["gender", "count"]  # enforce correct names

        total = df_data.loc[df_data[strata_cohort] == cohort].shape[0]
        df_counts["percentage"] = (df_counts["count"] * 100 / total).round(3)

        print(f"\nCohort: {cohort}")
        print(df_counts)

    # Count cases/controls within each cohort
    df_counts = (
        df_quest.groupby([strata_cohort, "diagnosis"])
        .size()
        .reset_index(name="count")
    )

    # Total per cohort
    df_counts["total"] = df_counts.groupby(strata_cohort)["count"].transform("sum")

    # Percentages per cohort
    df_counts["percentage"] = (df_counts["count"] / df_counts["total"] * 100).round(2)

    print(df_counts)

    round(209 * 100 / 307)

    # %%Two Step Group vs actig only -> VascBrain

    df_tw_stage = df_quest_yes_actig_yes.copy()
    df_act_only = df_quest_no_actig_yes.copy()
    df_tw_stage[strata_cohort] = 'Two Stage'
    df_act_only[strata_cohort] = 'Actigprahy Only'
    columns.get('categorical').append(strata_diagnosis)

    common_ids = set(df_tw_stage.subject_id) & set(df_act_only.subject_id)
    len(common_ids)

    compute_table_comparison_wrapper(df=pd.concat([df_tw_stage, df_act_only]),
                                    strata_cohort=strata_cohort,
                                    strata_diagnosis=strata_diagnosis,
                                    strata_study=strata_study,
                                    columns=columns,
                                    table_name='TwoStageVsAcrOnly',
                                    output_path=output_path)

    df_quest_only = df_quest_yes_actig_no.copy()
    df_quest_only[strata_cohort] = 'Questionnaire Only'

    common_ids = set(df_tw_stage.subject_id) & set(df_quest_only.subject_id)


    compute_table_comparison_wrapper(df=pd.concat([df_tw_stage, df_quest_only]),
                                    strata_cohort=strata_cohort,
                                    strata_diagnosis=strata_diagnosis,
                                    strata_study=strata_study,
                                    columns=columns,
                                    table_name='TwoStageVsQuestOnly',
                                    output_path=output_path)

    # df_data.loc[df_data[strata_cohort] == 'Stanford', 'gender'].value_counts()
    # df_data.loc[df_data[strata_cohort] == 'Clinic', 'gender'].value_counts()
    #
    #
    # for cohort in df_data[strata_cohort].unique():
    #     cohort_n = df_data.loc[df_data[strata_cohort] == cohort, 'gender'].shape[0]
    #     count = df_data.loc[df_data[strata_cohort] == cohort, 'gender'].sum()
    #     percent = (round(count * 100 / cohort_n), 2)
    #     print(f'\nCohort: {cohort} : {count} / {percent}%')
    #












