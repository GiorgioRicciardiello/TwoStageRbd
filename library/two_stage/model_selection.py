"""
SELECT: ML QUESTIONNAIRE

Select the best model given the stages specificaitons as the models are intetented to be selected as
a series of diagnostic tests

    stages = {
        'first_stage': {
            'sens': 90,
            'spc': 60
        },
        'second_stage': {
            'sens':60,
            'spc': 90
        }
    }

"""


import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from matplotlib import colors, patches
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score
)
import matplotlib.colors as mcolors



#%% ------------------------- questionnaire  ------------------------------------------
def get_questionnaire_single_column(path_questionnaire: Path, col_quest: str = 'q1_rbd') -> pd.DataFrame:
    df = pd.read_csv(path_questionnaire)
    df = df.loc[df['has_quest'] == 1, :]
    return df[['subject_id', col_quest]]

#%% ------------------------- questionnaire ML ------------------------------------------

def get_best_questionnaire_model_by_selection(path_quest_pred: Path, model_select: Dict[str, str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    if model_select == None:
        model_select = {
            'optimization': 'auc',
            'threshold': 'youden',
            'model_type': 'xgboost'
        }
    # read the metrics and the predictions
    df_metrics_quest = pd.read_csv(path_quest_pred.joinpath('df_outer_metrics_records_ci.csv'))
    df_predictions_quest = pd.read_csv(path_quest_pred.joinpath('predictions_outer_folds.csv'))

    valid_opt = [*df_metrics_quest['optimization'].unique()]
    valid_thr = [*df_metrics_quest['threshold'].unique()]

    if not model_select.get('optimization') in valid_opt:
        raise ValueError(f'Invalid optimization selection: {model_select.get("optimization")}\n\tValid optimization selection: {valid_opt}')

    if not model_select.get('threshold') in valid_thr:
        raise ValueError(f'Invalid threshold selection: { model_select.get("threshold")}\n\tValid optimization selection: {valid_thr}')

    # get the predictions
    df_select_metrics = df_metrics_quest.loc[
        (df_metrics_quest['optimization'] == model_select.get('optimization')) &
        (df_metrics_quest['threshold'] == model_select.get('threshold')) &
        (df_metrics_quest['model_type'] == model_select.get('model_type'))]
    assert df_select_metrics.shape[0] == 1

    df_selected_pred = df_predictions_quest.loc[
        (df_predictions_quest['model_type'] == model_select.get('model_type')) &
        (df_predictions_quest['optimization'] == model_select.get('optimization')) &
        (df_predictions_quest['threshold_type'] == model_select.get('threshold'))]

    assert df_selected_pred.shape[0] == df_selected_pred['subject_id'].nunique()

    # clean column names
    df_selected_pred = df_selected_pred.loc[:, ['subject_id', 'y_true', 'y_score', 'threshold_value', 'y_pred']]
    df_selected_pred = df_selected_pred.rename(columns={'y_pred': 'y_pred_quest',
                                     'threshold_value': 'thr_opt_quest'})
    return df_select_metrics, df_selected_pred


def get_most_sensitive_questionnaire(path_quest_pred: Path,) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    df_metrics_quest = pd.read_csv(path_quest_pred.joinpath('df_outer_metrics_records_ci.csv'))
    df_predictions_quest = pd.read_csv(path_quest_pred.joinpath('predictions_outer_folds.csv'))

    # df_metrics_quest = df_metrics_quest.loc[(df_metrics_quest['model_type'] == 'xgboost') &
    #                                         (df_metrics_quest['optimization'] == 'youden')]
    df_select_metrics = df_metrics_quest.loc[df_metrics_quest['sensitivity'] == df_metrics_quest['sensitivity'].max()]
    # get the most specific among the selection
    df_select_metrics = df_select_metrics.loc[df_select_metrics['specificity'] == df_select_metrics['specificity'].max()]
    assert df_select_metrics.shape[0] == 1

    df_selected_pred = df_predictions_quest.loc[
        (df_predictions_quest['model_type'] == df_select_metrics['model_type'].values[0]) &
        (df_predictions_quest['optimization'] == df_select_metrics['optimization'].values[0]) &
        (df_predictions_quest['threshold_type'] == df_select_metrics['threshold'].values[0])]

    assert df_selected_pred.shape[0] == df_selected_pred['subject_id'].nunique()

    # clean column names
    df_selected_pred = df_selected_pred.loc[:, ['subject_id', 'y_true', 'y_score', 'threshold_value', 'y_pred']]
    df_selected_pred = df_selected_pred.rename(columns={'y_pred': 'y_pred_quest',
                                     'threshold_value': 'thr_opt_quest'})
    return df_select_metrics, df_selected_pred


#%% ------------------------- Actigraphy ML ------------------------------------------

def get_best_actigraphy_model_by_selection(path_quest_actig: Path,
                                           model_select: Dict[str, str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    if model_select == None:
        model_select = {
            'optimization': 'auc',
            'threshold': 'youden',
            'model_type': 'xgboost'
        }
    # read the metrics and the predictions
    df_metrics_actig = pd.read_csv(path_quest_actig.joinpath('metrics', 'metrics_outer_folds_ci.csv'))
    df_predictions_actig = pd.read_csv(path_quest_actig.joinpath('predictions', 'predictions_outer_folds.csv'))

    # get the predictions
    df_select_metrics = df_metrics_actig.loc[
        (df_metrics_actig['optimization'] == model_select.get('optimization')) &
        (df_metrics_actig['threshold'] == model_select.get('threshold')) &
        (df_metrics_actig['model_type'] == model_select.get('model_type'))]
    assert df_select_metrics.shape[0] == 1

    df_selected_pred = df_predictions_actig.loc[
        (df_predictions_actig['model_type'] == model_select.get('model_type')) &
        (df_predictions_actig['optimization'] == model_select.get('optimization')) &
        (df_predictions_actig['threshold_type'] == model_select.get('threshold'))]

    # we need to average the nights per subject
    df_subj = (
        df_selected_pred
        .groupby(["outer_fold", "optimization", "subject_id"])
        .agg(y_true=("y_true", "first"),
             y_score=("y_score", "mean"),
             threshold_value=('threshold_value', 'mean'))
        .reset_index()
    )


    assert df_subj.shape[0] == df_subj['subject_id'].nunique()

    #
    df_subj = df_subj.loc[:, ['subject_id', 'y_true', 'y_score', 'threshold_value']]
    df_subj['y_pred_actig'] = (df_subj['y_score'] >=df_select_metrics['threshold_value'].values[0]).astype(int)
    # clean column names

    df_subj = df_subj.rename(columns={'threshold_value': 'thr_opt_quest'})
    # compute the predictions form the average threshold
    return df_select_metrics, df_subj



def get_best_specific_actigraphy(path_quest_actig: Path,
                                           model_select: Dict[str, str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    if model_select == None:
        model_select = {
            'optimization': 'auc',
            'threshold': 'youden',
            'model_type': 'xgboost'
        }
    # read the metrics and the predictions
    df_metrics_actig = pd.read_csv(path_quest_actig.joinpath('df_outer_metrics_records_ci.csv'))
    df_predictions_actig = pd.read_csv(path_quest_actig.joinpath('predictions_outer_folds.csv'))


    df_select_metrics = df_metrics_actig.loc[df_metrics_actig['specificity'] == df_metrics_actig['specificity'].max()]
    # get the most specific among the selection
    df_select_metrics = df_select_metrics.loc[df_select_metrics['sensitivity'] == df_select_metrics['sensitivity'].max()]
    assert df_select_metrics.shape[0] == 1

    df_selected_pred = df_predictions_actig.loc[
        (df_predictions_actig['model_type'] == df_select_metrics['model_type'].values[0]) &
        (df_predictions_actig['optimization'] == df_select_metrics['optimization'].values[0]) &
        (df_predictions_actig['threshold_type'] == df_select_metrics['threshold'].values[0])]

    # we need to average the nights per subject
    df_subj = (
        df_selected_pred
        .groupby(["outer_fold", "optimization", "subject_id"])
        .agg(y_true=("y_true", "first"),
             y_score=("y_score", "mean"),
             threshold_value=('threshold_value', 'mean'))
        .reset_index()
    )
    assert df_subj.shape[0] == df_subj['subject_id'].nunique()

    df_subj = df_subj.loc[:, ['subject_id', 'y_true', 'y_score', 'threshold_value']]
    df_subj['y_pred_actig'] = (df_subj['y_score'] >=df_select_metrics['threshold_value'].values[0]).astype(int)
    # clean column names

    df_subj = df_subj.rename(columns={'threshold_value': 'thr_opt_quest'})
    # compute the predictions form the average threshold
    return df_select_metrics, df_subj




# %% --------------------------------------- Visualization -------------------------


def plot_confusion_matrices_style_bar(
    df,
    y_true_col: str = 'y_true',
    class_names: Dict[int, str] = None,
    comparison: str = None,
    methods: List[str] = None,
    titles: List[str] = None,
    output_dir: Path = None,
    figsize: Tuple[float, float] = None,
    theme: str = "whitegrid",
    rc_style: dict = None,
    show_plot: bool = True,
):

    sns.set_theme(style=theme)
    default_rc = {
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 16
    }
    plt.rcParams.update(default_rc if rc_style is None else rc_style)

    def get_metrics(y_true, y_pred):
        cmat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cmat.ravel()
        sens = recall_score(y_true, y_pred) * 100
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = precision_score(y_true, y_pred) * 100
        acc = accuracy_score(y_true, y_pred) * 100
        return cmat, sens, spec, prec, acc

    if class_names is None:
        class_names = ['Control', 'iRBD']
    elif isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    if figsize is None:
        figsize = (14, 9)

    if methods is None:
        methods = ['y_pred_quest', 'y_pred_acitg', 'serial_test']
    if titles is None:
        titles = ['Questionnaire', 'Actigraphy', 'Serial Test']

    fig, axes = plt.subplots(
        2, len(methods),
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    cm_axes, bar_axes = axes[0], axes[1]

    y_true = df[y_true_col]
    n_cases, n_controls = sum(y_true == 1), sum(y_true == 0)

    metric_colors = {
        "Se": "#66c2a5",  # teal
        "Sp": "#fc8d62",  # orange
        "Pr": "#8da0cb",  # purple
        "Acc":  "#e78ac3"   # pink
    }
    metric_order = [*metric_colors.keys()]
    colorset = sns.color_palette("Pastel1", len(methods))
    red_, blue_, green_ = colorset[0], colorset[1], colorset[2]
    colorset[0] = blue_
    colorset[1] = green_
    colorset[2] = red_

    metrics_stages = {}
    for col, (ax, bar_ax, method, title) in enumerate(zip(cm_axes, bar_axes, methods, titles)):
        y_pred = df[method]
        cmat, sens, spec, prec, acc = get_metrics(y_true, y_pred)
        cm_percent = cmat.astype("float") / cmat.sum(axis=1)[:, np.newaxis] * 100

        labels = np.array([
            [f"{cmat[i,j]}\n({cm_percent[i,j]:.1f}%)" for j in range(cmat.shape[1])]
            for i in range(cmat.shape[0])
        ])
        cmap = sns.light_palette(colorset[col], as_cmap=True)

        sns.heatmap(
            cmat, annot=labels, fmt="",
            cmap=cmap, ax=ax, cbar=False,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)) + 0.5)
        ax.set_xticklabels(class_names)
        ax.set_yticks(np.arange(len(class_names)) + 0.5)
        ax.set_yticklabels(class_names)
        ax.set_title(title, fontweight="bold")

        # === Metric bars ===
        bar_ax.set_xlim(0, 100)
        bar_ax.set_ylim(0, 1)
        bar_ax.axis("off")
        # bar_ax.grid(alpha=0.7, axis='x')

        metrics = {"Se": sens, "Sp": spec, "Pr": prec, "Acc": acc}
        bar_height, spacing = 0.18, 0.22
        metrics_stages[method] = metrics
        for i, name in enumerate(metric_order):
            val = metrics[name]
            y = 1 - (i+1) * spacing

            bar_ax.add_patch(patches.Rectangle((0, y), 100, bar_height,
                                               # facecolor="whitesmoke",
                                               facecolor="#f0f0f0",
                                               edgecolor="lightgray",
                                               lw=0.5))

            bar_ax.add_patch(patches.Rectangle((0, y), val, bar_height,
                                               facecolor=metric_colors[name],
                                               edgecolor="none"))

            # === Add dashed gridlines across all bar plots ===
            for x in [80, 85, 90, 95, 100]:
                bar_ax.axvline(x, color="lightgray", linestyle="--", lw=0.8, zorder=0)

            # Value text at right end
            bar_ax.text(val + 1, y + bar_height/2, f"{val:.1f}%",
                        va="center", ha="left", fontsize=10, color="black")

            # Show metric labels only on first (leftmost) column
            if col == 0:
                bar_ax.text(-5, y + bar_height/2, name,
                            va="center", ha="right", fontsize=11, weight="bold")



    subtitle = f"{comparison}\n" if comparison else ""
    big_title = subtitle + f"Confusion Matrices (n={len(df)} | Cases={n_cases}, Controls={n_controls})"
    fig.suptitle(big_title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)

    if output_dir:
        file_name = "cm_two_stage.png" if comparison is None else f"cm_two_stage_{comparison}.png"
        plt.savefig(output_dir.joinpath(file_name), dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return metrics_stages




# %% Wrapper exhaustive two stage
#  These functons are to use in a series
def run_exhaustive_two_stage(
    path_questionnaire: Path,
    path_quest_pred: Path,
    path_acti_pred: Path,
    col_test: str = "q1_rbd",
    output_dir: Optional[Path] = None,
    plt_show: bool = False,
) -> pd.DataFrame:
    """
    Exhaustively run all combinations of questionnaire and actigraphy models,
    merge predictions into two-stage tests, plot confusion matrices,
    and collect stage metrics in a summary table.

    Usage:

        df_results = run_exhaustive_two_stage(
        path_questionnaire=path_questionnaire,
        path_quest_pred=path_quest_pred,
        path_acti_pred=path_acti_pred,
        col_test="q1_rbd",  # raw questionnaire column
        output_dir=None  # where to save the plots
    )

    Returns
    -------
    pd.DataFrame with metrics for all combinations
    """

    # --------------------- valid options ---------------------
    quest_opts = ["auc", "maxsens", "youden"]
    quest_thrs = ["0p5", "sens_max", "youden"]

    actig_opts = ["auc", "maxspec", "youden"]
    actig_thrs = ["0p5", "spec_max", "youden"]

    results = []

    # --------------------- loop over all combinations ---------------------
    for q_opt in quest_opts:
        for q_thr in quest_thrs:
            model_select_questionnaire = {
                "optimization": q_opt,
                "threshold": q_thr,
                "model_type": "xgboost"  # 'xgboost'
            }
            df_quest_metrics, df_quest_pred = get_best_questionnaire_model_by_selection(
                path_quest_pred=path_quest_pred,
                model_select=model_select_questionnaire
            )

            for a_opt in actig_opts:
                for a_thr in actig_thrs:
                    model_select_actigraphy = {
                        "optimization": a_opt,
                        "threshold": a_thr,
                        "model_type": "xgboost"
                    }
                    df_actig_metrics, df_actig_pred = get_best_actigraphy_model_by_selection(
                        path_quest_actig=path_acti_pred,
                        model_select=model_select_actigraphy
                    )

                    # ------------------- merge -------------------
                    df_quest_raw = get_questionnaire_single_column(
                        path_questionnaire=path_questionnaire,
                        col_quest=col_test
                    )

                    df_two_stage = pd.merge(
                        df_quest_pred[["subject_id", "y_pred_quest"]],
                        df_actig_pred[["subject_id", "y_true", "y_pred_actig"]],
                        on="subject_id"
                    )
                    df_two_stage = pd.merge(df_two_stage, df_quest_raw, on="subject_id")
                    df_two_stage[col_test] = df_two_stage[col_test].astype(int)

                    df_two_stage["serial_test_quest_raw_actig"] = (
                        (df_two_stage[col_test] == 1) & (df_two_stage["y_pred_actig"] == 1)
                    ).astype(int)

                    df_two_stage["serial_test_quest_ml_actig"] = (
                        (df_two_stage["y_pred_quest"] == 1) & (df_two_stage["y_pred_actig"] == 1)
                    ).astype(int)

                    # ------------------- plot -------------------
                    comparison = f"Quest[{q_opt}|{q_thr}] & Actig[{a_opt}|{a_thr}]"

                    metrics_raw = plot_confusion_matrices_style_bar(
                        df=df_two_stage,
                        y_true_col="y_true",
                        class_names={0: "Control", 1: "iRBD"},
                        methods=[col_test, "y_pred_actig", "serial_test_quest_raw_actig"],
                        titles=["Questionnaire Raw", "Actigraphy", "Two-Stage Raw"],
                        comparison=comparison + " (Raw)",
                        figsize=(12, 6),
                        output_dir=output_dir,
                        show_plot=plt_show,
                    )

                    metrics_ml = plot_confusion_matrices_style_bar(
                        df=df_two_stage,
                        y_true_col="y_true",
                        class_names={0: "Control", 1: "iRBD"},
                        methods=["y_pred_quest", "y_pred_actig", "serial_test_quest_ml_actig"],
                        titles=["Questionnaire ML", "Actigraphy", "Two-Stage ML"],
                        comparison=comparison + " (ML)",
                        figsize=(12, 6),
                        output_dir=output_dir,
                        show_plot=plt_show,
                    )

                    # ------------------- record results -------------------
                    results.append({
                        "quest_opt": q_opt,
                        "quest_thr": q_thr,
                        "actig_opt": a_opt,
                        "actig_thr": a_thr,
                        "metrics_raw": metrics_raw,
                        "metrics_ml": metrics_ml
                    })

    # flatten results into a table
    records = []
    for r in results:
        for stage, metrics in {"raw": r["metrics_raw"], "ml": r["metrics_ml"]}.items():
            for method, vals in metrics.items():
                records.append({
                    "quest_opt": r["quest_opt"],
                    "quest_thr": r["quest_thr"],
                    "actig_opt": r["actig_opt"],
                    "actig_thr": r["actig_thr"],
                    "stage": stage,
                    "method": method,
                    **vals
                })

    df_results = pd.DataFrame(records)
    return df_results




def select_best_ml_and_compare_raw(df_results: pd.DataFrame, min_spec: float = 96) -> pd.DataFrame:
    """
    Select the best ML row with specificity >= min_spec (highest sensitivity among those),
    then fetch the matching Raw row with the same actigraphy configuration.

    Returns
    -------
    pd.DataFrame with 2 rows: [ML row, Raw row]
    """
    # 1. filter ML stage
    df_ml = df_results[df_results["stage"] == "ml"]

    # 2. keep only rows with specificity >= min_spec
    df_ml_highspec = df_ml[df_ml["Sp"] >= min_spec]

    if df_ml_highspec.empty:
        raise ValueError(f"No ML models found with specificity >= {min_spec}")

    # 3. pick row with max sensitivity
    best_ml_row = df_ml_highspec.loc[df_ml_highspec["Se"].idxmax()]

    # 4. find matching RAW row with same actigraphy config
    df_raw = df_results[
        (df_results["stage"] == "raw") &
        (df_results["actig_opt"] == best_ml_row["actig_opt"]) &
        (df_results["actig_thr"] == best_ml_row["actig_thr"])
        ]

    if df_raw.empty:
        raise ValueError("No matching RAW configuration found for the best ML row")

    # pick the serial_test_quest_raw_actig row if multiple exist
    best_raw_row = df_raw[df_raw["method"] == "serial_test_quest_raw_actig"].iloc[0]

    # 5. return both for comparison
    return pd.DataFrame([best_ml_row, best_raw_row])





def plot_best_ml_vs_raw(
        df_results: pd.DataFrame,
        path_questionnaire: Path,
        path_quest_pred: Path,
        path_acti_pred: Path,
        col_test: str = "q1_rbd",
        min_spec: float = 96,
        output_dir: Optional[Path] = None
):
    """
    Find best ML (Se highest given Sp >= min_spec),
    get matching Raw configuration, regenerate predictions,
    and plot confusion matrices for comparison.
    """
    # --- step 1: get ML + RAW rows
    comparison = select_best_ml_and_compare_raw(df_results, min_spec=min_spec)
    best_ml_row = comparison[comparison["stage"] == "ml"].iloc[0]
    best_raw_row = comparison[comparison["stage"] == "raw"].iloc[0]

    # --- step 2: rebuild questionnaire + actigraphy predictions ---
    model_select_questionnaire = {
        "optimization": best_ml_row["quest_opt"],
        "threshold": best_ml_row["quest_thr"],
        "model_type": "xgboost"
    }
    model_select_actigraphy = {
        "optimization": best_ml_row["actig_opt"],
        "threshold": best_ml_row["actig_thr"],
        "model_type": "xgboost"
    }

    # questionnaire ML
    _, df_quest_pred = get_best_questionnaire_model_by_selection(
        path_quest_pred=path_quest_pred,
        model_select=model_select_questionnaire
    )

    # actigraphy ML
    _, df_actig_pred = get_best_actigraphy_model_by_selection(
        path_quest_actig=path_acti_pred,
        model_select=model_select_actigraphy
    )

    # raw questionnaire
    df_quest_raw = get_questionnaire_single_column(
        path_questionnaire=path_questionnaire,
        col_quest=col_test
    )

    # --- step 3: merge ---
    df_two_stage = pd.merge(
        df_quest_pred[["subject_id", "y_pred_quest"]],
        df_actig_pred[["subject_id", "y_true", "y_pred_actig"]],
        on="subject_id"
    )
    df_two_stage = pd.merge(df_two_stage, df_quest_raw, on="subject_id")
    df_two_stage[col_test] = df_two_stage[col_test].astype(int)

    df_two_stage["serial_test_quest_raw_actig"] = (
            (df_two_stage[col_test] == 1) & (df_two_stage["y_pred_actig"] == 1)
    ).astype(int)

    df_two_stage["serial_test_quest_ml_actig"] = (
            (df_two_stage["y_pred_quest"] == 1) & (df_two_stage["y_pred_actig"] == 1)
    ).astype(int)

    # --- step 4: plot raw vs ML ---
    # comparison_name = (
    #     f"Quest[{best_ml_row['quest_opt']}|{best_ml_row['quest_thr']}] + "
    #     f"Actig[{best_ml_row['actig_opt']}|{best_ml_row['actig_thr']}]"
    # )

    # raw plot
    plot_confusion_matrices_style_bar(
        df=df_two_stage,
        y_true_col="y_true",
        class_names={0: "Control", 1: "iRBD"},
        methods=[col_test, "y_pred_actig", "serial_test_quest_raw_actig"],
        titles=["RBD Symptoms", "Actigraphy", "Two-stage"],
        comparison="RBD Symptoms and Actigraphy",
        figsize=(12, 6),
        output_dir=output_dir
    )

    # ml plot
    plot_confusion_matrices_style_bar(
        df=df_two_stage,
        y_true_col="y_true",
        class_names={0: "Control", 1: "iRBD"},
        methods=["y_pred_quest", "y_pred_actig", "serial_test_quest_ml_actig"],
        titles=["Four-Items ML", "Actigraphy", "Two-stage"],
        comparison="Four-Item ML and Actigraphy",
        figsize=(12, 6),
        output_dir=output_dir
    )

    return comparison














