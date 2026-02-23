import os
import glob
import json
import joblib
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = Path("../results/final_nested_cv_run").resolve()
PKL_PATH: Path = RESULTS_DIR / "best_final_model_fold.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _extract_auc(path: Path) -> float:
    """Extract AUC value from filename if present."""
    try:
        return float(path.stem.split("auc=")[-1])
    except Exception:
        return -np.inf


def load_best_model(df: pd.DataFrame, approved_features: List[str],
                    preprocessor: Any) -> Pipeline:
    """
    Load the best saved model, or refit using best hyperparameters if missing.

    Args:
        df: Input dataframe with features and target.
        approved_features: List of features to use.
        preprocessor: Preprocessing pipeline for features.

    Returns:
        Fitted sklearn Pipeline.
    """
    if PKL_PATH.exists():
        return joblib.load(PKL_PATH)

    fold_models = sorted(RESULTS_DIR.glob("fold_*_final_model.pkl"))
    if fold_models:
        best_path = max(fold_models, key=_extract_auc)
        model = joblib.load(best_path)
        joblib.dump(model, PKL_PATH)
        return model

    # Last resort: refit with best Optuna trial from all folds
    print("❗ No model artefacts found – refitting on the full dataset…")
    X = df[approved_features]
    y = df["label"]

    grid_csvs = sorted(RESULTS_DIR.glob("fold_*_study_results.csv"))
    best_trial = (
        pd.concat(
            pd.read_csv(f).sort_values("value", ascending=False).head(1)
            for f in grid_csvs
        )
        .sort_values("value", ascending=False)
        .iloc[0]
    )
    params = {k.replace("param_", ""): v for k, v in best_trial.items()
              if k.startswith("param_")}

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=800,
            random_state=42,
            **params
        ))
    ])
    model.fit(X, y)
    joblib.dump(model, PKL_PATH)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_features(model: Pipeline, df: pd.DataFrame,
                      approved_features: List[str]) -> None:
    """
    Evaluate feature importance using SHAP (preferred) or gain-based XGBoost.

    Args:
        model: Trained sklearn Pipeline with 'clf' step.
        df: Input dataframe with features.
        approved_features: List of features used in the model.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model.named_steps["clf"])
        X_trans = model.named_steps["prep"].transform(df[approved_features])
        shap_values = explainer.shap_values(X_trans, check_additivity=False)

        shap.summary_plot(
            shap_values, X_trans, feature_names=approved_features,
            max_display=20, show=True
        )
    except (ImportError, AttributeError) as e:
        warnings.warn(
            f"SHAP unavailable ({e}). Falling back to gain-based XGBoost importance."
        )
        booster = model.named_steps["clf"].get_booster()
        imp_df = (
            pd.Series(booster.get_score(importance_type="gain"))
            .sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index(name="gain_importance")
        )
        plt.figure(figsize=(7, 6))
        imp_df.head(20).plot.barh(x="feature", y="gain_importance", legend=False)
        plt.gca().invert_yaxis()
        plt.title("Top-20 Features by Gain Importance")
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def summarise_hyperparameters() -> pd.DataFrame:
    """Summarise best hyperparameters across outer folds."""
    hp_files = sorted(glob.glob(str(RESULTS_DIR / "fold_*_study_results.csv")))
    records: List[Dict[str, Any]] = []

    for csv_file in hp_files:
        df_fold = pd.read_csv(csv_file)
        best_row = df_fold.loc[df_fold["value"].idxmax()]
        params = {k.replace("param_", ""): v
                  for k, v in best_row.items() if k.startswith("param_")}
        params["best_auc"] = best_row["value"]
        params["fold"] = Path(csv_file).stem.split("_")[1]
        records.append(params)

    return pd.DataFrame(records).sort_values(
        "best_auc", ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP AUC CI
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray,
                  n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for AUC.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities for positive class.
        n_boot: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        (lower_ci, upper_ci) tuple for 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example usage (requires df, approved_features, preprocessor defined earlier)
    best_model = load_best_model(df, approved_features, preprocessor)
    print(f"✅ Best model ready @ {PKL_PATH}")

    evaluate_features(best_model, df, approved_features)

    hp_summary = summarise_hyperparameters()
    print(hp_summary)
