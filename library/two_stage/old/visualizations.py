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


def plot_confusion_matrices(df,
                            y_true_col='y_true',
                            class_names=None,
                            output_dir:Path=None):
    """
    Plots confusion matrices for questionnaire, actigraphy, and serial test predictions.

    Parameters:
    - df: DataFrame with columns ['y_true', 'y_pred_quest', 'y_pred_acitg', 'serial_test']
    - y_true_col: name of the column with ground truth labels
    """

    def get_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = recall_score(y_true, y_pred) * 100
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = precision_score(y_true, y_pred) * 100
        acc = accuracy_score(y_true, y_pred) * 100
        return cm, sens, spec, prec, acc

    if class_names is None:
        class_names = ['Control', 'iRBD']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = ['y_pred_quest', 'y_pred_acitg', 'serial_test']
    titles = ['Questionnaire', 'Actigraphy', 'Serial Test']

    y_true = df[y_true_col]
    n_cases = sum(y_true == 1)
    n_controls = sum(y_true == 0)

    for ax, method, title in zip(axes, methods, titles):
        y_pred = df[method]
        cm, sens, spec, prec, acc = get_metrics(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[0], class_names[1]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([class_names[0], class_names[1]])

        ax.set_title(
            f"{title}\nSens: {sens:.1f}%, Spec: {spec:.1f}%, Prec: {prec:.1f}%, Acc: {acc:.1f}%",
            fontsize=12
        )

    plt.suptitle(
        f"Confusion Matrices (n={len(df)} | Cases={n_cases}, Controls={n_controls})",
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_dir:
        plt.savefig(output_dir.joinpath(f'cm_two_stage.png'), dpi=300)

    plt.show()



def plot_confusion_matrices_style(
    df,
    y_true_col: str = 'y_true',
    class_names: List[str] = None,
    comparison: str = None,
    methods: List[str] = None,
    titles: List[str] = None,
    output_dir: Path = None,
    figsize: Tuple[float, float] = None,
    theme: str = "whitegrid",
    rc_style: dict = None
):
    """
    Plots confusion matrices with consistent theme/style and metrics.

    Parameters
    ----------
    df : DataFrame
        Must include ground truth and prediction columns.
    y_true_col : str
        Column with ground truth labels.
    class_names : list
        Class names for axes.
    comparison : str
        Optional subtitle for the figure.
    methods : list
        Prediction columns to plot.
    titles : list
        Titles for each confusion matrix subplot.
    output_dir : Path
        Optional path to save figure.
    figsize : tuple
        Figure size.
    theme : str
        Seaborn theme (default: "whitegrid").
    rc_style : dict
        Matplotlib rcParams overrides.
    """

    # === Apply theme and style globally ===
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
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = recall_score(y_true, y_pred) * 100
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = precision_score(y_true, y_pred) * 100
        acc = accuracy_score(y_true, y_pred) * 100
        return cm, sens, spec, prec, acc

    if class_names is None:
        class_names = ['Control', 'iRBD']
    if figsize is None:
        figsize = (12, 8)

    if methods is None:
        methods = ['y_pred_quest', 'y_pred_acitg', 'serial_test']
    if titles is None:
        titles = ['Questionnaire', 'Actigraphy', 'Serial Test']

    if class_names is None:
        class_names = ['Control', 'iRBD']
    elif isinstance(class_names, dict):
        # sort by class index to keep order stable
        class_names = [class_names[i] for i in sorted(class_names.keys())]


    fig, axes = plt.subplots(1, len(methods), figsize=figsize)
    if len(methods) == 1:
        axes = [axes]

    y_true = df[y_true_col]
    n_cases = sum(y_true == 1)
    n_controls = sum(y_true == 0)

    for ax, method, title in zip(axes, methods, titles):
        y_pred = df[method]
        cm, sens, spec, prec, acc = get_metrics(y_true, y_pred)
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Annotate with counts + percentages in **bold**
        labels = np.array([
            [f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 11, "weight": "bold"}
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(class_names)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(class_names)
        ax.set_title(title, fontweight="bold")

        # Metrics under each matrix
        ax.text(
            0.5, -0.25,
            f"Sensitivity: {sens:.1f}%\nSpecificity: {spec:.1f}% \nPrecision: {prec:.1f}% \nAccuracy: {acc:.1f}%",
            fontsize=16,
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    # === Subtitle above the main title ===
    subtitle = f"{comparison}\n" if comparison else ""
    big_title = (subtitle +
        f"Confusion Matrices (n={len(df)} | Cases={n_cases}, Controls={n_controls})")

    fig.suptitle(big_title, fontsize=16, fontweight="bold", y=1.0001)
    plt.tight_layout(rect=[0,  0.1, 1, 0.92])
    plt.tight_layout()

    if output_dir:
        if comparison:
            file_name = "cm_two_stage_" + re.sub(r"\W+", "_", comparison) + ".png"
        else:
            file_name = "cm_two_stage.png"
        plt.savefig(output_dir.joinpath(file_name), dpi=300, bbox_inches="tight")

    plt.show()



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
    rc_style: dict = None
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

    colorset = sns.color_palette("Pastel1", len(methods))

    fig, axes = plt.subplots(
        2, len(methods),
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    cm_axes, bar_axes = axes[0], axes[1]

    y_true = df[y_true_col]
    n_cases, n_controls = sum(y_true == 1), sum(y_true == 0)

    metric_colors = {
        "Sens": "#66c2a5",  # teal
        "Spec": "#fc8d62",  # orange
        "Prec": "#8da0cb",  # purple
        "Acc":  "#e78ac3"   # pink
    }
    metric_order = ["Sens", "Spec", "Prec", "Acc"]

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

        metrics = {"Sens": sens, "Spec": spec, "Prec": prec, "Acc": acc}
        bar_height, spacing = 0.18, 0.22

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

    plt.show()


# %%  plot single models


def _plot_minimalist_roc(
    y_true,
    y_score,
    model: str,
    thr_name: str,
    color: str,
    out_path: Path,
    auc_val: float,
    sens_val: float, spec_val: float,
    thr_val: float,
    n_boot:int = 1000,
    auc_ci: str | None = None,
    sens_ci: str | None = None,
    spec_ci: str | None = None,
):
    """Save minimalist ROC curve + threshold marker and export metrics to CSV."""

    if not color.startswith("#"):
        color = f"#{color}"

    mean_fpr = np.linspace(0, 1, len(y_true))
    fpr_all, tpr_all, thr_all = roc_curve(y_true, y_score)

    # mean curve (bootstrap or folds optional)
    tprs = []
    for _ in range(n_boot):  # bootstrapped tprs
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        fpr_i, tpr_i, _ = roc_curve(y_true[idx], y_score[idx])
        tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    # threshold marker
    idx = np.argmin(np.abs(thr_all - thr_val))
    fpr_thr, tpr_thr = fpr_all[idx], tpr_all[idx]
    tpr_thr_interp = np.interp(fpr_thr, mean_fpr, mean_tpr)

    # save metrics row
    metrics_dict = {
        "model": model,
        "threshold_type": thr_name,
        "thr_val": round(thr_val, 3),
        "auc": round(auc_val, 3),
        "sensitivity": round(sens_val, 3),
        "specificity": round(spec_val, 3),
        "fpr_thr": round(fpr_thr, 3),
        "tpr_thr": round(tpr_thr_interp, 3),
    }

    # plotting
    rgb = np.array(mcolors.to_rgb(color))
    light_rgb = np.clip(rgb + 0.3, 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_alpha(0)

    ax.plot(mean_fpr, mean_tpr, color=color, lw=2)
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color=light_rgb, alpha=0.5)

    # threshold marker
    ax.scatter(fpr_thr, tpr_thr_interp, color=color, s=70,
               edgecolors="k", zorder=3)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.3)

    # light axes
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", which="both", labelsize=8, labelcolor=(0, 0, 0, 0.5))
    for spine in ax.spines.values():
        spine.set_alpha(0.5)
    ax.grid(alpha=.7)

    if out_path:
        # output paths
        mini_path = out_path.joinpath("roc_curves")
        mini_path.mkdir(parents=True, exist_ok=True)
        fname_img = mini_path / f"roc_{model}_{thr_name.replace('*', 'opt')}_thr{thr_val:.2f}.png"

        plt.savefig(fname_img, dpi=300, bbox_inches="tight", transparent=True)
        plt.close()

        # CSV append
        df_metrics = pd.DataFrame([metrics_dict])
        existing_csvs = list(mini_path.glob("roc_metrics*.csv"))
        if existing_csvs:
            df_metrics.to_csv(existing_csvs[0], mode="a", header=False, index=False)
        else:
            df_metrics.to_csv(mini_path / "roc_metrics.csv", index=False)



def plot_single_screening(
        df_predictions: pd.DataFrame,
        subject_col: str = "subject_id",
        results_dir: Path | None = None,
        class_names: dict[int, str] = {0: "Control", 1: "Case"},
        thresholds:Dict[str, float] | None = None,
        figsize: Tuple[int, int] = (10, 5),
        font_size_title: int = 14,
        font_size_big_title: int = 18,
        font_size_label: int = 12,
        font_size_legend: int = 10,
        font_size_cm: int = 12,
        modality: str = 'Screening Modality'):
    """
    Aggregate predictions per subject, compute metrics, and plot ROC + CM with CIs.
    """


    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _bootstrap_ci(metric_fn, y_true, y_score, thr=None, n_boot=2000, alpha=0.95):
        """Bootstrap CI for a metric."""
        rng = np.random.default_rng(42)
        vals = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, ys = y_true[idx], y_score[idx]
            if thr is not None:
                vals.append(metric_fn(yt, ys, thr))
            else:
                vals.append(metric_fn(yt, ys))
        low, high = np.percentile(vals, [(1 - alpha) / 2 * 100, (1 + (alpha)) / 2 * 100])
        return np.mean(vals), (low, high)

    def _sens(y_true, y_score, thr):  # sensitivity
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn + 1e-12)

    def _spec(y_true, y_score, thr):  # specificity
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp + 1e-12)


    # ----------------------------------------------------
    # Aggregate per subject
    # ----------------------------------------------------
    n_unique = df_predictions[subject_col].nunique()

    if n_unique == df_predictions.shape[0]:
        df_avg = df_predictions
        do_boot = True
    else:
        # df_avg, df_metrics_avg = _aggregate_avg(df_predictions)

        do_boot = False
        df_avg = (
            df_predictions
            .groupby(subject_col, as_index=False)
            .agg({
                "y_true": "first",
                "y_pred": "mean",
                "thr_opt": "first",
            })
        )
    if thresholds is None:
        # get the standard and the optimal
        thresholds = {
            "standard": 0.5,
            "opt": df_avg["thr_opt"].iloc[0],
        }

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    fig, axes = plt.subplots(1, len(thresholds) * 2, figsize=figsize)

    for j, (thr_name, thr_val) in enumerate(thresholds.items()):
        ax_roc, ax_cm = axes[j * 2], axes[j * 2 + 1]
        y_true, y_score = df_avg["y_true"].values, df_avg["y_pred"].values

        # ------------------- ROC with CI -------------------
        # if do_boot:
        mean_auc, (auc_lo, auc_hi) = _bootstrap_ci(
            lambda yt, ys: roc_auc_score(yt, ys), y_true, y_score
        )
        # else:
        #     mean_auc, (auc_lo, auc_hi) = _compute_ci()

        fpr, tpr, thr = roc_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, color="blue", lw=2)

        # Shade std/CI by bootstrap sampling of ROC points
        mean_fpr = np.linspace(0, 1, 200)
        bootstrapped_tprs = []
        rng = np.random.default_rng(42)
        for _ in range(500):
            idx = rng.integers(0, len(y_true), len(y_true))
            fpr_i, tpr_i, _ = roc_curve(y_true[idx], y_score[idx])
            bootstrapped_tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))
        bootstrapped_tprs = np.array(bootstrapped_tprs)
        tpr_mean = bootstrapped_tprs.mean(axis=0)
        tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
        tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
        ax_roc.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

        # ------------------- Metrics with CI -------------------
        mean_sens, (sens_lo, sens_hi) = _bootstrap_ci(_sens, y_true, y_score, thr=thr_val)
        mean_spec, (spec_lo, spec_hi) = _bootstrap_ci(_spec, y_true, y_score, thr=thr_val)

        # scatter threshold point
        idx_thr = np.argmin(np.abs(thr - thr_val))
        ax_roc.scatter(fpr[idx_thr], tpr[idx_thr], color="salmon", s=80, edgecolors="k", zorder=3)

        # legend text
        legend_text = (
            f"AUC = {mean_auc:.2f} ({auc_lo:.2f}–{auc_hi:.2f})\n"
            f"Se = {mean_sens * 100:.1f}% ({sens_lo * 100:.1f}–{sens_hi * 100:.1f}%)\n"
            f"Sp = {mean_spec * 100:.1f}% ({spec_lo * 100:.1f}–{spec_hi * 100:.1f}%)\n"
            f"τ = {thr_val:.2f}"
        )
        thr_name = thr_name.replace('standard', 'τ').replace('opt', 'τ*')
        ax_roc.legend([legend_text], fontsize=font_size_legend, loc="lower right")
        ax_roc.set_title(f"ROC | {thr_name} ", fontsize=font_size_title)
        ax_roc.set_xlabel("False Positive Rate", fontsize=font_size_label)
        ax_roc.set_ylabel("True Positive Rate", fontsize=font_size_label)
        ax_roc.grid(True, linestyle="--", alpha=0.7)

        # minimalist ROC export
        if results_dir:
            _plot_minimalist_roc(
                y_true=y_true,
                y_score=y_score,
                model=modality.replace(" ", "_"),
                thr_name=thr_name,
                color="#A8D5BA",  # or pass from palette
                out_path=results_dir,
                auc_val=mean_auc,
                sens_val=mean_sens,
                spec_val=mean_spec,
                thr_val=thr_val,
            )

        # ------------------- Confusion Matrix -------------------
        y_bin = (y_score >= thr_val).astype(int)
        cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(
            cm, annot=[[f"{v}\n({p:.1f}%)" for v, p in zip(r, pr)] for r, pr in zip(cm, cm_pct)],
            fmt="",
            annot_kws={"size": font_size_cm},
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=[class_names[0], class_names[1]],
            yticklabels=[class_names[0], class_names[1]]
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"CM | {thr_name}", fontsize=font_size_title)

    n_cases, n_controls = sum(df_avg.y_true == 1), sum(df_avg.y_true == 0)
    fig.suptitle(f"Diagnostic Performance {modality}\n "
                 f"(n={len(df_avg)} | {class_names.get(1)}={n_cases}, {class_names.get(0)}={n_controls})",
                 fontsize=font_size_big_title, y=1.001)
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir.joinpath(f"ROC_{modality}.png"), dpi=300)
    plt.show()




def plot_for_diagram_figure_one(
    df_predictions: pd.DataFrame,
    subject_col: str = "subject_id",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    thr_col: str = "thr_spec_max",
    class_names: dict[int, str] = {0: "Control", 1: "Case"},
    figsize: Tuple[int, int] = (5, 5),
    color: str = "#1f77b4",
    n_boot: int = 1000,
    results_dir: Path | None = None,
):
    """
    Aggregate predictions per subject/fold, compute metrics, and plot ROC with CI.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Must contain subject_id, y_true, y_pred, thr_spec_max.
    subject_col : str
        Column with subject ID or fold ID (for aggregation).
    y_true_col, y_pred_col, thr_col : str
        Column names for ground truth, predicted probs, threshold.
    class_names : dict
        Label mapping for confusion matrix.
    figsize : tuple
        Size of the ROC figure.
    color : str
        Base color for ROC curve and CI shading.
    n_boot : int
        Number of bootstrap samples for CI.
    results_dir : Path
        If provided, saves plot to this directory.
    """

    # ----------------------------------------------------
    # Aggregate per subject
    # ----------------------------------------------------
    if df_predictions[subject_col].nunique() == df_predictions.shape[0]:
        df_avg = df_predictions.copy()
    else:
        print(f'Computing average across {len(df_predictions)} subjects...')
        df_avg = (
            df_predictions.groupby(subject_col, as_index=False)
            .agg({
                y_true_col: "first",
                y_pred_col: "mean",
                thr_col: "first",
            })
        )

    y_true = df_avg[y_true_col].values
    y_score = df_avg[y_pred_col].values
    thr_spec = df_avg[thr_col].iloc[0]

    # ----------------------------------------------------
    # Bootstrap helper
    # ----------------------------------------------------
    def _bootstrap_ci(metric_fn, thr=None):
        rng = np.random.default_rng(42)
        vals = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, ys = y_true[idx], y_score[idx]
            vals.append(metric_fn(yt, ys, thr) if thr is not None else metric_fn(yt, ys))
        low, high = np.percentile(vals, [2.5, 97.5])
        return np.mean(vals), (low, high)

    def _sens(yt, ys, thr):
        y_bin = (ys >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, y_bin).ravel()
        return tp / (tp + fn + 1e-12)

    def _spec(yt, ys, thr):
        y_bin = (ys >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, y_bin).ravel()
        return tn / (tn + fp + 1e-12)

    # ----------------------------------------------------
    # ROC + CI
    # ----------------------------------------------------
    mean_auc, (auc_lo, auc_hi) = _bootstrap_ci(lambda yt, ys: roc_auc_score(yt, ys))

    fpr, tpr, thr = roc_curve(y_true, y_score)
    mean_fpr = np.linspace(0, 1, 200)

    bootstrapped_tprs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        fpr_i, tpr_i, _ = roc_curve(y_true[idx], y_score[idx])
        bootstrapped_tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))
    bootstrapped_tprs = np.array(bootstrapped_tprs)

    tpr_mean = bootstrapped_tprs.mean(axis=0)
    tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {mean_auc:.3f}")
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

    # Threshold metrics
    sens, (sens_lo, sens_hi) = _bootstrap_ci(_sens, thr=thr_spec)
    spec, (spec_lo, spec_hi) = _bootstrap_ci(_spec, thr=thr_spec)

    idx_thr = np.argmin(np.abs(thr - thr_spec))
    ax.scatter(fpr[idx_thr], tpr[idx_thr], color="orange", s=70, edgecolors="k", zorder=3)

    legend_text = (
        f"AUC = {mean_auc:.3f} ({auc_lo:.3f}–{auc_hi:.3f})\n"
        f"Se = {sens*100:.1f}% ({sens_lo*100:.1f}–{sens_hi*100:.1f}%)\n"
        f"Sp = {spec*100:.1f}% ({spec_lo*100:.1f}–{spec_hi*100:.1f}%)"
    )
    ax.legend([legend_text], fontsize=10, loc="lower right", frameon=False)

    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("ROC Curve with CI", fontsize=14)

    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir.joinpath("roc_subject_level.png"), dpi=300, transparent=True)
    plt.show()

    return {
        "AUC": (mean_auc, auc_lo, auc_hi),
        "Sensitivity": (sens, sens_lo, sens_hi),
        "Specificity": (spec, spec_lo, spec_hi),
    }


