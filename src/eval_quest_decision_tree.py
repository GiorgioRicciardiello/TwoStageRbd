#!/usr/bin/env python3
"""
select_questions.py

Compute AUC, sensitivity, specificity for each ordinal questionnaire item (0, 0.5, 1)
and fit a small decision tree to see which questions best separate your binary outcome.
"""
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_text
from config.config import config
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import matplotlib.patches as mpatches
import matplotlib.colors as mc, colorsys
import seaborn as sns

def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Varimax rotation for factor loadings.
    Args:
      Phi   : (p × k) matrix of unrotated loadings
      gamma : Kaiser normalization parameter (1.0 = varimax)
      q     : max number of iterations
      tol   : convergence tolerance
    Returns:
      (p × k) matrix of rotated loadings
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        # Compute the orthogonal rotation update via SVD
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vh
        d_new = s.sum()
        if d != 0 and (d_new - d) < tol:
            break
        d = d_new
    return Phi @ R


def run_pca_varimax(df: pd.DataFrame,
                    question_cols: list[str],
                    n_components: int = 3,
                    standardize: bool = True):
    """
    1) Optionally standardize the question columns.
    2) Fit PCA(n_components) and get unrotated loadings.
    3) Rotate loadings via Varimax.
    4) Compute factor scores for each subject.
    Returns:
      rotated_loadings : pd.DataFrame (questions × components)
      factor_scores    : pd.DataFrame (subjects × components)
      pca              : fitted PCA object
    """
    X = df[question_cols].astype(float).values
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    # 1) PCA fit
    pca = PCA(n_components=n_components)
    scores_unrot = pca.fit_transform(X)
    # unrotated loadings: each column = sqrt(var_i) × component vector
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # 2) Varimax rotation
    rotated = varimax(loadings)

    # 3) Rotated factor scores
    #    project standardized X onto rotated axes
    factor_scores = X @ rotated

    # wrap in DataFrames
    comp_names = [f"F{i+1}" for i in range(n_components)]
    rotated_loadings = pd.DataFrame(
        rotated, index=question_cols, columns=comp_names
    )
    factor_scores = pd.DataFrame(
        factor_scores, columns=comp_names, index=df.index
    )

    return rotated_loadings, factor_scores, pca


def evaluate_questions(df: pd.DataFrame,
                       question_cols: list[str],
                       label_col: str = 'label') -> pd.DataFrame:
    """
    For each question in `question_cols`, treats its values (0,0.5,1) as a score,
    computes AUC, finds threshold maximizing Youden's J, and returns
    sensitivity, specificity, and Youden's J at that threshold.
    """
    y_true = df[label_col].values
    results = []

    for q in question_cols:
        y_score = df[q].values.astype(float)

        # 1) AUC
        auc = roc_auc_score(y_true, y_score)

        # 2) ROC curve → fpr, tpr, thresholds
        fpr, tpr, thr = roc_curve(y_true, y_score)
        spec = 1 - fpr  # specificity

        # 3) Youden's J = sens + spec - 1
        J = tpr + spec - 1
        idx = np.nanargmax(J)
        best_thr = thr[idx]
        best_sens = tpr[idx]
        best_spec = spec[idx]

        results.append({
            'question':       q,
            'auc':            round(auc*100, 4),
            'youden_j':       J[idx],
            'best_threshold': best_thr,
            'sensitivity':    round(best_sens*100, 4),
            'specificity':    round(best_spec*100, 4)
        })

    return (
        pd.DataFrame(results)
          .sort_values('auc', ascending=False)
          .reset_index(drop=True)
    )


def build_decision_tree(df: pd.DataFrame,
                        question_cols: list[str],
                        label_col: str = 'label',
                        max_depth: int = 3) -> tuple[DecisionTreeClassifier, str]:
    """
    Fits a DecisionTreeClassifier(max_depth) on the question columns and returns
    both the fitted model and a human-readable text summary.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(df[question_cols], df[label_col])
    tree_text = export_text(clf, feature_names=question_cols)
    return clf, tree_text






if __name__ == "__main__":
    # --- Input ---
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    label_col = 'diagnosis'
    max_depth=2
    n_components = 2
    standardize = True
    # %% Select only questionnaire samples
    df_data = df_data[(df_data.has_quest == 1)].copy()
    # automatically pick all columns starting with the given prefix
    question_cols = [c for c in df_data.columns if c.startswith('q')]
    if not question_cols:
        raise ValueError(f"No columns start with prefix 'q'")

    # %%
    # -- varimax rotation
    loadings, scores, pca = run_pca_varimax(
        df_data, question_cols,
        n_components=n_components,
        standardize=standardize
    )
    print("\nExplained variance (unrotated):")
    for i, var in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {var:.3f}")

    print("\nRotated loadings (first few items):")
    print(loadings.head().round(3))

    # --- visualization factors and diagnosis
    df_scores = scores.copy()
    df_scores['diagnosis'] = df_data['diagnosis']
    fig, ax = plt.subplots()
    for diag, group in df_scores.groupby('diagnosis'):
        ax.scatter(group['F1'], group['F2'], label=diag)

    ax.set_xlabel('Factor 1 (F1)')
    ax.set_ylabel('Factor 2 (F2)')
    ax.set_title('Scatter of Rotated Factor Scores by Diagnosis')
    ax.legend(title='Diagnosis')
    plt.tight_layout()
    plt.show()

    # --- compute per-question metrics ---
    metrics_df = evaluate_questions(df_data, question_cols, label_col=label_col)
    metrics_df['auc'] = metrics_df['auc'].round(1)
    metrics_df['sensitivity'] = metrics_df['sensitivity'].round(1)
    metrics_df['specificity'] = metrics_df['specificity'].round(1)

    # print and save
    print("\nPer-question performance (sorted by AUC):\n")
    print(metrics_df.to_string(index=False, float_format="%.3f"))
    metrics_df.to_csv("question_metrics.csv", index=False)
    print("\nSaved detailed metrics to question_metrics.csv")

    # --- fit a small decision tree ---
    clf, tree_text = build_decision_tree(
        df_data, question_cols, label_col=label_col, max_depth=max_depth
    )
    print(f"\nDecision tree (max_depth={max_depth}):\n")
    print(tree_text)

    # --- count responses ---
    # Count q1_rbd answers within each diagnosis group
    totals = df_data["diagnosis"].value_counts().to_dict()
    # {0: n_controls, 1: n_cases}

    # supplementary table counting how many stratified for each responses between the cases and controls
    for question in [col for col in df_data.columns if col.startswith('q')]:
        counts = (
            df_data.groupby(["diagnosis", question])
            .size()
            .reset_index(name="count")
        )

        # Percent relative to total cases or controls
        counts["percent"] = counts.apply(
            lambda row: round(100 * row["count"] / totals[row["diagnosis"]], 1),
            axis=1
        )

        counts["report"] = counts["count"].astype(str) + " (" + counts["percent"].astype(str) + "%)"

        print(f"\n--- {question} ---")
        print(counts)




