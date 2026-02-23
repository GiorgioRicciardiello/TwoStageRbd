import numpy as np
from numpy.ma.core import left_shift
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from numpy import ndarray
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
# =============================================================================
# =============================================================================
# =============================================================================
# Custom scoring and threshold utilities for Youden's J statistic
# =============================================================================
from sklearn.metrics import roc_curve
def youdens_j_statistic(y_true, y_prob):
    # Calculate Youden's J statistic and return (J, threshold)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return j_scores[idx], thresholds[idx]

def youdens_j_score_func(y_true, y_prob):
    # Compute only the maximum Youden's J statistic
    j, _ = youdens_j_statistic(y_true, y_prob)
    return float(j)



def youdens_j_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute Youden's J statistic and the corresponding optimal threshold (tau)
    from continuous prediction scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_score : np.ndarray
        Predicted continuous scores (e.g., probabilities for the positive class).

    Returns
    -------
    J_opt : float
        Maximum Youden's J statistic (sensitivity - false positive rate).
    tau_opt : float
        Threshold corresponding to J_opt.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    J = tpr - fpr
    ix = int(np.argmax(J))
    return float(J[ix]), float(thr[ix])


def metrics_at_threshold(y_true: ndarray,
                         y_score: ndarray,
                         tau: float) -> Dict[str, float]:
    """
    Return dict of sensitivity, specificity, Youden's J statistic,
    confusion matrix, AUC score, and PRC (average precision) score
    at threshold tau.
    """
    # Predictions at given threshold
    y_pred = (y_score >= tau).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity & specificity (proportions)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Youden's J statistic (proportion scale)
    youden_j = sens + spec - 1

    # AUC score (threshold-independent)
    auc_score = roc_auc_score(y_true, y_score)

    # PRC score (average precision)
    prc_score = average_precision_score(y_true, y_score)

    return {
        "thr": float(tau),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": round(sens * 100, 3),   # percentage
        "specificity": round(spec * 100, 3),   # percentage
        "youden_j": round(youden_j, 3),        # proportion
        "auc_score": round(auc_score, 5),
        "prc_score": round(prc_score, 5)
    }


def generate_ci(
    df_metrics: pd.DataFrame,
    confidence: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean ± CI for selected metrics for each (model_type, optimization) pair
    and merge them into df_metrics. Handles metrics that may be in [0,1] or [0,100].

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Metrics of each fold and each model.
        Must contain 'model_type' and 'optimization' columns.
    confidence : float, optional
        Confidence level for the CI (default 0.95).

    Returns
    -------
    df_metrics_with_ci : pd.DataFrame
        Original df_metrics with CI columns merged.
    df_ci : pd.DataFrame
        Table of mean ± CI strings for each (model_type, optimization) pair.
    """
    from scipy import stats
    import numpy as np

    def _mean_ci_str(series: pd.Series) -> str:
        data = series.dropna().astype(float)
        n = len(data)
        if n <= 1:
            return np.nan

        # Detect scale: assume percentage if mean > 1.5 (arbitrary cutoff)
        scale = 100.0 if data.mean() > 1.5 else 1.0
        data_scaled = data / scale

        mean = data_scaled.mean()
        sem = stats.sem(data_scaled)
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        low, high = mean - h, mean + h
        # Clip bounds to [0,1]
        low = max(0.0, low)
        high = min(1.0, high)

        # Format back in original scale
        mean_out, low_out, high_out = mean * scale, low * scale, high * scale

        if scale == 1.0:
            return f"{mean_out:.3f} ({low_out:.3f}, {high_out:.3f})"
        else:  # percentage
            return f"{mean_out:.1f}% ({low_out:.1f}%, {high_out:.1f}%)"

    # Select metric columns of interest
    metrics_ci = [
        col for col in df_metrics.columns
        if any(key in col for key in ("sensitivity", "specificity", "auc", "prc"))
    ]

    # Compute CI table for all (model_type, optimization) pairs
    df_ci = (
        df_metrics
        .groupby(['model_type', 'optimization'])[metrics_ci]
        .agg(lambda x: _mean_ci_str(x))
        .rename(columns={col: f"{col}_ci" for col in metrics_ci})
        .reset_index()
    )

    # Merge CI table back into original metrics
    df_metrics_with_ci = df_metrics.merge(
        df_ci,
        on=["model_type", "optimization"],
        how="left"
    )

    return df_metrics_with_ci, df_ci





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
