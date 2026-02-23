import numpy as np
from typing import Callable, Optional, List, Tuple
import pathlib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import re
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path
from matplotlib import colors


def _plot_minimalist_roc(
        fpr: np.ndarray,
        tpr: np.ndarray,
        model: str,
        opt: str,
        color: str,
        out_path: Path,
):
    """
    Minimalist roc curve using the FPR and TPR, useful for figure 1 of the paper
    :param fpr:
    :param tpr:
    :param model:
    :param opt:
    :param color:
    :param out_path:
    :return:
    """

    import matplotlib.colors as mcolors

    # --- colors ---
    rgb = np.array(mcolors.to_rgb(color))
    light_rgb = np.clip(rgb + 0.3, 0, 1)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_alpha(0)

    ax.plot(fpr, tpr, color=color, lw=2, )

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
        out_path = out_path.joinpath("roc_curves")
        out_path.mkdir(parents=True, exist_ok=True)
        fname = out_path / f"roc_{model}_{opt}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
        # plt.show()
    plt.close()
    # return fname


def plot_roc_with_threshold_cms_grid_avg_subjects(
        df_predictions: pd.DataFrame,
        df_metrics_ci: pd.DataFrame,
        model_type: str,
        class_names: dict[int, str] = None,
        output_path: Path = None,
        pastel_palette: str = "Pastel1",
        title: str = None,
        figsize: Tuple[int, int] = (16, 8),
        # --- font controls from input ---
        title_size=18,  # subplot titles (ROC + CM)
        axis_label_size = 13,  # ROC axis labels
        tick_size = 11,  # ROC ticks
        legend_size = 11,  # ROC legend
        cm_font_size = 12,  # CM counts + %
        suptitle_size = 15,  # whole figure
        show:bool = True,
        scale:float = 1,
):
    def _draw_confusion_matrix(ax, cm, color, title,
                               class_names={0: "Neg", 1: "Pos"},
                               font_size_cm=12,
                               fontsize_ticks=20,
                               font_size_label=12,
                               font_size_title=12,
                               plt_xlabel:bool = True,
                               plt_ylabel:bool = True,):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cmap = sns.light_palette(color, as_cmap=True)
        im = ax.imshow(cm, cmap=cmap)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val, pct = cm[r, c], cm_pct[r, c]
                # bg_color = im.cmap(im.norm(cm[r, c]))  # color by count
                bg_color = im.cmap(im.norm(pct))  # color by percentage
                im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)  # for percentage
                brightness = colors.rgb_to_hsv(bg_color[:3])[2]
                text_color = "black" if brightness > 0.5 else "white"
                ax.text(c, r, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=font_size_cm, color=text_color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[0], class_names[1]], fontsize=fontsize_ticks)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([class_names[0], class_names[1]], fontsize=fontsize_ticks)
        if plt_xlabel:
            ax.set_xlabel("Predicted", fontsize=font_size_label)
        if plt_ylabel:
            ax.set_ylabel("True", fontsize=font_size_label)
        ax.set_title(title, fontsize=font_size_title, )  # fontweight="bold"
        ax.grid(alpha=0)


    if class_names is None:
        class_names = {0: "Neg", 1: "Pos"}

    formal_threshold_name = {
        '0p5': r'$\tau$',
        'youden': r'$\tau^{*}$',
        'spec_max': r'$\tau_{\text{sp}}$',
        'sens_max': r'$\tau_{\text{se}}$',
    }


    title_size = int(title_size * scale)
    axis_label_size = int(axis_label_size * scale)
    tick_size = int(tick_size * scale)
    legend_size = int(legend_size * scale)
    cm_font_size = int(cm_font_size * scale)
    suptitle_size = int(suptitle_size * scale)

    # average the subjects so we have one subject observation per fold
    df_subj = (
        df_predictions[df_predictions["model_type"] == model_type]
        .groupby(["outer_fold", "optimization", "subject_id"])
        .agg(y_true=("y_true", "first"),
             y_score=("y_score", "mean"),
             y_score_std=("y_score", "std"))
        .reset_index()
    )
    df_subj["y_score_std"] = df_subj["y_score_std"].fillna(0)

    counts = df_subj.groupby("y_true")["subject_id"].nunique()
    n_controls = counts.get(0, 0)
    n_cases = counts.get(1, 0)

    optimizations = df_metrics_ci["optimization"].unique()
    thresholds = df_metrics_ci["threshold"].unique()
    colorset = sns.color_palette(pastel_palette, len(thresholds))

    n_rows = len(optimizations)
    n_cols = 1 + len(thresholds)

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"width_ratios": [2] + [1] * (n_cols - 1)}  # make first col double wide

    )

    for i, opt in enumerate(optimizations):

        row_center = 1 - ((i + 0.5) / n_rows)
        fig.text(
            x=0.01,  # left margin position
            y=row_center,
            s=opt.title(),
            va='center',
            ha='left',
            fontsize=axis_label_size,
            rotation='vertical',
            fontweight='bold'
        )

        df_opt = df_subj[df_subj["optimization"] == opt]
        y_true = df_opt["y_true"].values
        y_score = df_opt["y_score"].values
        y_std = df_opt["y_score_std"].values

        # ROC plot
        ax_roc = axes[i, 0]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        _plot_minimalist_roc(fpr=fpr,
                             tpr=tpr,
                             model='xgboost',
                             opt=opt,
                             color="#A9C9FF",
                             out_path=None
                             )

        tpr_lower = np.clip(tpr - np.mean(y_std), 0, 1)
        tpr_upper = np.clip(tpr + np.mean(y_std), 0, 1)

        ax_roc.plot(fpr, tpr, color="black", lw=2, label="ROC")
        ax_roc.fill_between(fpr, tpr_lower, tpr_upper, color="gray", alpha=0.2, label="±1 std")
        ax_roc.fill_between(fpr, tpr_lower, tpr_upper, color="gray", alpha=0.2, label="±1 std")
        # ax_roc.plot([0, 1], [0, 1], "--", color="lightgray")

        ax_roc.set_xlabel("False Positive Rate", fontsize=axis_label_size)
        ax_roc.set_ylabel(f"\n\n\nTrue Positive Rate", fontsize=axis_label_size)
        # ax_roc.set_ylabel(f"\True Positive Rate", fontsize=axis_label_size)
        ax_roc.tick_params(axis="both", labelsize=tick_size)
        ax_roc.grid(True, alpha=0.7)

        metrics_cell = df_metrics_ci.loc[
            (df_metrics_ci["optimization"] == opt)
            , ['prc_score_ci', 'auc_score_ci']
        ].drop_duplicates(keep='first')

        proc = metrics_cell.prc_score_ci.values[0]
        auc = metrics_cell.auc_score_ci.values[0]

        # auc_score = roc_auc_score(y_true, y_score)
        # prc_score = average_precision_score(y_true, y_score)
        # auc = auc_score
        # proc = prc_score

        title_roc = f'\nPR: {proc}\nAUC: {auc}'
        # title_roc = f'AUC: {auc}'

        ax_roc.set_title(title_roc, fontsize=suptitle_size)  # fontweight="bold" )
        handles, labels = [], []
        for j, (thr, c) in enumerate(zip(thresholds, colorset), start=1):
            ax_cm = axes[i, j]

            thr_val = df_metrics_ci.loc[
                (df_metrics_ci["optimization"] == opt) &
                (df_metrics_ci["threshold"] == thr),
                "threshold_value"
            ].values[0]

            metrics_cell = df_metrics_ci.loc[
                (df_metrics_ci["optimization"] == opt) &
                (df_metrics_ci["threshold"] == thr),
                :
            ]
            # se = metrics_cell['sensitivity'].values[0]
            # sp = metrics_cell['specificity'].values[0]

            # metrics at the subject level
            y_pred = (y_score >= thr_val).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp = cm.ravel()

            se = round((tp / (tp + fn) if (tp + fn) > 0 else 0.0)*100, 3)  # Recall
            sp = round((tn / (tn + fp) if (tn + fp) > 0 else 0.0)*100, 3)
            title_cm = f"{formal_threshold_name[thr]} = {thr_val:.2f}"
            
            _draw_confusion_matrix(
                ax=ax_cm,
                cm=cm,
                color=c,
                class_names=class_names,
                title=title_cm,
                font_size_cm=cm_font_size,
                font_size_label=axis_label_size,
                font_size_title=suptitle_size,
                fontsize_ticks=axis_label_size,
                plt_xlabel=True if j==2 else False,
                plt_ylabel=False,
            )
            fpr_thr = (y_pred[y_true == 0].sum() / (y_true == 0).sum())
            tpr_thr = (y_pred[y_true == 1].sum() / (y_true == 1).sum()
                       if (y_true == 1).sum() > 0 else 0)
            h = ax_roc.scatter(fpr_thr, tpr_thr, color=c, s=150, edgecolor="k")
            handles.append(h)
            # labels.append(f"τ={thr_val:.3f} ")
            labels.append(f"{formal_threshold_name[thr]}={thr_val:.2f} ")

        ax_roc.legend(handles, labels, fontsize=legend_size, loc="lower right", frameon=False)

    if title:
        # title = title + f" | Model: {model_type} | Controls: {n_controls} | {class_names[1]}: {n_cases}"
        title = title + f". Controls: {n_controls} | {class_names[1]}: {n_cases}"

    else:
        title = f"Model: {model_type} | Controls: {n_controls} | {class_names[1]}: {n_cases}"

    # --- title close to figure ---
    fig.suptitle(
        title,
        fontsize=title_size,
        # fontweight="bold",
        y=1.001)

    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.15, hspace=0.25)
    # plt.tight_layout(pad=0)
    plt.tight_layout(pad=0.7, w_pad=1.5, h_pad=1.5)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()



def compute_ppv_table(
        df_predictions: pd.DataFrame,
        df_metrics_ci: pd.DataFrame,
        model_type: str,
        prevalence: float = 0.015
) -> pd.DataFrame:
    """
    Compute PPV, adjusted PPV (at fixed prevalence), Sensitivity, Specificity
    for each optimization and threshold.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Predictions including columns: ["model_type", "outer_fold", "optimization",
                                        "subject_id", "y_true", "y_score"].
    df_metrics_ci : pd.DataFrame
        Metrics table with thresholds info, including ["optimization", "threshold", "threshold_value"].
    model_type : str
        Which model to evaluate.
    prevalence : float
        Prevalence to adjust PPV (default = 0.015 → 1.5%).

    Returns
    -------
    pd.DataFrame
        Table with model, optimization, tau, ppv, ppv_adj, sensitivity, specificity.
    """

    # Keep only unique subject-level scores
    df_scores = (
        df_predictions[df_predictions["model_type"] == model_type]
        .drop_duplicates(subset=["outer_fold", "optimization", "subject_id"])
    )

    results = []

    for opt in df_metrics_ci["optimization"].unique():
        df_opt = df_scores[df_scores["optimization"] == opt]
        y_true = df_opt["y_true"].values
        y_score = df_opt["y_score"].values

        for thr in df_metrics_ci["threshold"].unique():
            thr_val = df_metrics_ci.loc[
                (df_metrics_ci["optimization"] == opt) &
                (df_metrics_ci["threshold"] == thr),
                "threshold_value"
            ].values
            if len(thr_val) == 0 or np.isnan(thr_val[0]):
                thr_val = [0.5]  # fallback default

            thr_val = thr_val[0]
            y_pred = (y_score >= thr_val).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # Sensitivity and Specificity
            se = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            # Raw PPV
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

            # Adjusted PPV for fixed prevalence
            # PPV_adj = (Se * Prev) / (Se * Prev + (1-Sp) * (1-Prev))
            if not np.isnan(se) and not np.isnan(sp):
                ppv_adj = (se * prevalence) / (se * prevalence + (1 - sp) * (1 - prevalence))
            else:
                ppv_adj = np.nan

            results.append({
                "model": model_type,
                "optimization": opt,
                "tau": thr,
                "tau_value": thr_val,
                "sensitivity": round(se * 100, 2),
                "specificity": round(sp * 100, 2),
                "ppv": round(ppv * 100, 2) if ppv == ppv else np.nan,
                "ppv_adj": round(ppv_adj * 100, 2) if ppv_adj == ppv_adj else np.nan,
            })

    return pd.DataFrame(results)


# %% Feature importance


def compute_feature_importance(models_dir: pathlib.Path,
                            features_names:List[str],
                               objective: str,
                               top_n: int = 15,):
    """
    Compute and plot feature importance across folds
    for XGBoost models saved as 'fold_{k}_{objective}_model.pkl'.

    Parameters
    ----------
    models_dir : pathlib.Path
        Path to directory with models.
    objective : str
        One of ['auc', 'spec', 'youden'].
    top_n : int
        Number of top features to plot (ranked by gain).
    """

    # filter model files
    # model_files = sorted(models_dir.glob(f"fold*_{objective}_model.pkl"))
    model_files = sorted(models_dir.glob(f"outer*_{objective}.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No models found for objective '{objective}' in {models_dir}")

    excel_path = models_dir.joinpath(f"feature_importance_{objective}.xlsx")
    plot_path = models_dir.joinpath(f"feature_importance_{objective}.png")

    if features_names:
        features_names = [re.sub(r'_', ' ', fet).title() for fet in features_names]
    if excel_path.exists():
        # --- Load precomputed importance ---
        df_importance = pd.read_excel(excel_path, index_col=0, header=[0, 1])
    else:
        importance_records = []
        # --- Load models and extract importance ---
        for model_file in model_files:
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            # unwrap booster if needed
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
            else:
                booster = model
            if booster.feature_names is None:
                booster.feature_names = features_names

            for importance_type in ["weight", "gain", "cover"]:
                imp = booster.get_score(importance_type=importance_type)
                for feat, val in imp.items():
                    importance_records.append({
                        "feature": feat,
                        "importance_type": importance_type,
                        "value": val,
                        "model": model_file.stem
                    })

        df = pd.DataFrame(importance_records)

        # Pivot to feature × importance_type × folds
        df_pivot = df.pivot_table(
            index=["feature", "importance_type"],
            columns="model",
            values="value"
        )

        # Compute mean ± std across folds
        df_stats = df_pivot.apply([pd.Series.mean, pd.Series.std], axis=1).reset_index()
        df_stats = df_stats.rename(columns={"mean": "mean", "std": "std"})

        # Reshape for plotting
        df_importance = df_stats.pivot(
            index="feature",
            columns="importance_type",
            values=["mean", "std"]
        )

        # Save Excel
        df_importance.to_excel(excel_path)
        print(f"Saved feature importance table to {excel_path}")

    # --- Plot ---
    importance_types = ["weight", "gain", "cover"]
    titles = {
        "weight": "Frequency (Weight)",
        "gain": "Split Gain (Gain)",
        "cover": "Coverage (Cover)"
    }

    sns.set_style("whitegrid")
    colors = sns.color_palette("pastel", n_colors=3)

    # Rank features by gain
    top_features = df_importance["mean"]["gain"].sort_values(ascending=False).head(top_n).index

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for idx, imp_type in enumerate(importance_types):
        ax = axes[idx]
        means = df_importance["mean"][imp_type].loc[top_features]
        stds = df_importance["std"][imp_type].loc[top_features]

        ax.barh(top_features, means, xerr=stds,
                capsize=4, color=colors[idx], edgecolor="black")

        ax.invert_yaxis()
        ax.set_title(titles[imp_type], fontsize=16, weight="bold")
        ax.set_xlabel("Importance (mean ± std)", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.6, axis="x")

    fig.suptitle(f"XGBoost Feature Importance ({objective.capitalize()} Optimization)",
                 fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")





