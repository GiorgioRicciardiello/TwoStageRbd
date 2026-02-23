from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.metrics import confusion_matrix



def generate_ci(
    df_metrics: pd.DataFrame,
    confidence: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean ± CI for selected metrics, grouped by (optimization, threshold_type).
    Confidence intervals are across folds for each combination.
    """

    from scipy import stats

    def _mean_ci_str(series: pd.Series) -> str:
        data = series.dropna().astype(float)
        n = len(data)
        if n <= 1:
            return np.nan

        scale = 100.0 if data.mean() > 1.5 else 1.0
        data_scaled = data / scale

        mean = data_scaled.mean()
        sem = stats.sem(data_scaled)
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        low, high = mean - h, mean + h
        low = max(0.0, low)
        high = min(1.0, high)

        mean_out, low_out, high_out = mean * scale, low * scale, high * scale

        if scale == 1.0:
            return f"{mean_out:.3f} ({low_out:.3f}, {high_out:.3f})"
        else:
            return f"{mean_out:.1f}% ({low_out:.1f}%, {high_out:.1f}%)"

    metrics_ci = [
        col for col in df_metrics.columns
        if any(key in col for key in ("sensitivity", "specificity", "auc", "prc", "ppv", "npv", "youden"))
    ]

    # Compute CI grouped by optimization × threshold_type
    df_ci = (
        df_metrics
        .groupby(["optimization", "threshold_type"])[metrics_ci]
        .agg(lambda x: _mean_ci_str(x))
        .rename(columns={col: f"{col}_ci" for col in metrics_ci})
        .reset_index()
    )

    df_metrics_with_ci = df_metrics.merge(
        df_ci,
        on=["optimization", "threshold_type"],
        how="left"
    )

    return df_metrics_with_ci, df_ci

def compute_metrics(y_true, y_score, thr: float) -> Dict[str, float]:
    """Compute Sens, Spec, PPV, NPV, AUCs at threshold thr."""
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    ppv = tp / (tp + fp + 1e-12)
    npv = tn / (tn + fn + 1e-12)
    roc_auc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    youden_j = sens + spec - 1

    return {
        "sensitivity": round(sens * 100, 3),  # stored in %
        "specificity": round(spec * 100, 3),  # stored in %
        "auc": roc_auc,  # 0–1
        "prc": pr_auc,  # 0–1
        "ppv": ppv,  # 0–1
        "npv": npv,  # 0–1
        "youden": youden_j,  # 0–1
        'threshold': thr,
    }


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Youden's J statistic threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def _sensitivity_score(y_true, y_score, min_sens: float = 0.6):
    """Find threshold that maximizes specificity given sensitivity ≥ min_sens."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(tpr >= min_sens)[0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.5  # infeasible
    best_idx = valid[np.argmax(1 - fpr[valid])]
    spec = 1 - fpr[best_idx]
    sens = tpr[best_idx]
    thr = thresholds[best_idx]
    return spec, sens, thr




def compute_auc_by_cohort_and_fold(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ROC AUC for each cohort within each validation fold.
    Also reports the number and percentage of samples per cohort in each fold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        ['outer_fold', 'subject_id', 'y_true', 'y_score', 'y_pred', 'cohort']

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['outer_fold', 'cohort', 'n_samples', 'percent_within_fold', 'auc']
    """
    results = []

    for fold, df_fold in df.groupby('outer_fold'):
        total_samples = len(df_fold)
        for cohort, df_cohort in df_fold.groupby('cohort'):
            n_samples = len(df_cohort)
            pct = n_samples / total_samples * 100

            # compute AUC if both classes are present
            if df_cohort['y_true'].nunique() == 2:
                auc = roc_auc_score(df_cohort['y_true'], df_cohort['y_score'])
            else:
                auc = np.nan  # cannot compute AUC with single class

            results.append({
                'outer_fold': fold,
                'cohort': cohort,
                'n_samples': n_samples,
                'percent_within_fold': pct,
                'auc': auc
            })

    df_results = pd.DataFrame(results)
    df_summary = (
        df_results.groupby('cohort', as_index=False)
        .agg(
            mean_auc=('auc', 'mean'),
            sd_auc=('auc', 'std'),
            mean_pct=('percent_within_fold', 'mean'),
            mean_n=('n_samples', 'mean')
        )
    )
    return df_results, df_summary
