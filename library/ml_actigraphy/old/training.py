"""
Training for the machine learning actigraphy model
"""
from library.ml_actigraphy.scoring import compute_metrics, generate_ci
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split, BaseCrossValidator
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from typing import Sequence, Tuple, Union, Optional, List, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import joblib
from library.ml_questionnaire.scoring import (youdens_j_from_scores,
                                              metrics_at_threshold)
from library.ml_questionnaire.scoring import generate_ci
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from collections.abc import Iterable
from typing import Callable
import pickle
from scipy import stats

# ---------------------- Helper Functions ----------------------

def save_outer_cv_folds(
    results_dir: Path,
    outer_cv: BaseCrossValidator,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Save/load patient IDs for each outer CV fold."""
    folds_pickle_path = results_dir / "folds_patient_ids.pkl"

    if folds_pickle_path.exists():
        with open(folds_pickle_path, "rb") as f:
            return pickle.load(f)

    folds_dict = {}
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(x, y, groups), start=1):
        folds_dict[fold_num] = {
            "train": pd.DataFrame({
                "index": x.iloc[train_idx].index,
                "subject_id": groups.iloc[train_idx].values,
                "fold": fold_num
            }),
            "validation": pd.DataFrame({
                "index": x.iloc[test_idx].index,
                "subject_id": groups.iloc[test_idx].values,
                "fold": fold_num
            })
        }

    with open(folds_pickle_path, "wb") as f:
        pickle.dump(folds_dict, f)
    return folds_dict


def _group_or_stratified_split(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """GroupShuffleSplit fallback to stratified if groups fail."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, stop_idx = next(gss.split(x, y, groups))
        return (
            x.iloc[train_idx],
            x.iloc[stop_idx],
            y.iloc[train_idx],
            y.iloc[stop_idx],
        )
    except ValueError:
        return train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=random_state
        )


def _fit_with_early_stopping(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: BaseEstimator,
    params: dict
) -> Tuple[xgb.XGBClassifier, BaseEstimator]:
    """
    Fit XGBClassifier with preprocessing + early stopping.
    Returns both the trained model and the fitted preprocessor.
    """
    fitted_preprocessor = clone(preprocessor).fit(x_train)
    x_train_t = fitted_preprocessor.transform(x_train)
    x_val_t = fitted_preprocessor.transform(x_val)

    clf = xgb.XGBClassifier(**params)
    clf.fit(
        x_train_t,
        y_train,
        eval_set=[(x_val_t, y_val)],
        verbose=False
    )
    return clf, fitted_preprocessor

def _get_device():
    try:
        dtrain = xgb.DMatrix([[0],[1]], label=[0,1])
        xgb.train({"tree_method": "hist", "device": "cuda"}, dtrain, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"



# %%
# ---------------------------------------------------
# Threshold finder
# ---------------------------------------------------
def _best_sens_or_spec_thr(y_true,
                           y_score,
                           min_sens: float = 0.7,
                           min_spec: float = 0.6,
                           maximize: str = "spec") -> Tuple[float, float, float]:
    """
    Find the best threshold depending on objective.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities for positive class.
    min_sens : float, default=0.7
        Minimum sensitivity required if maximizing specificity.
    min_spec : float, default=0.6
        Minimum specificity required if maximizing sensitivity.
    maximize : {"spec", "sens"}, default="spec"
        Which metric to maximize:
        - "spec": maximize specificity subject to min sensitivity
        - "sens": maximize sensitivity subject to min specificity

    Returns
    -------
    best_spec : float
    best_sens : float
    best_thr : float
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    sens = tpr

    if maximize == "spec":
        valid = np.where(sens >= min_sens)[0]
        if len(valid) == 0:
            return 0.0, 0.0, 0.5
        best_idx = valid[np.argmax(spec[valid])]
    elif maximize == "sens":
        valid = np.where(spec >= min_spec)[0]
        if len(valid) == 0:
            return 0.0, 0.0, 0.5
        best_idx = valid[np.argmax(sens[valid])]
    else:
        raise ValueError("maximize must be either 'spec' or 'sens'")

    return spec[best_idx], sens[best_idx], thresholds[best_idx]



# ---------------------------------------------------
# Main training function
# ---------------------------------------------------

import gc
import torch
torch.cuda.empty_cache()


def train_nested_cv_xgb_optuna(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str,
    results_dir: Path,
    preprocessor: BaseEstimator,
    n_jobs: int = -1,
    random_seed: int = 42,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    n_trials: int = 50,
    n_estimators: int = 5000,
    eval_metric: str = "auc",
    early_stopping_rounds: int = 100,
    min_sens: float = 0.6,
    min_spec: float = 0.6,
    maximize: str = "spec",   # "spec" or "sens"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def _group_or_stratified_split(x, y, groups, test_size, random_state):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        try:
            tr_idx, va_idx = next(gss.split(x, y, groups))
            return x.iloc[tr_idx], x.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]
        except ValueError:
            return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)

    def _youdens_j_from_scores(y_true, y_score):
        fpr, tpr, thr = roc_curve(y_true, y_score)
        spec = 1 - fpr
        j_stats = tpr + spec - 1
        best_idx = np.argmax(j_stats)
        return j_stats[best_idx], thr[best_idx]

    def _format_metrics_row(label, m):
        """Helper function for formatting a row of metrics in print"""
        return f"{label:<8} Se/Sp = {m['sensitivity']:.1f}/{m['specificity']:.1f}% | AUC={m['auc']:.3f}"


    results_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = results_dir.joinpath("folds")
    fold_dir.mkdir(parents=True, exist_ok=True)
    thr_standard = 0.5
    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    all_fold_predictions, all_fold_metrics = [], []
    outer_cv = GroupKFold(n_splits=n_outer_splits)
    DEVICE = _get_device(); print(DEVICE)
    # ---------- Inner Optuna Objective ----------
    def _objective(trial, X_outer_train, y_outer_train, g_outer_train, fold_val_records):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "binary:logistic",
            "eval_metric": eval_metric,
            "tree_method": "hist" if DEVICE == "cpu" else "hist",
            "device": 'cuda' if DEVICE == "hist" else None,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "n_jobs": n_jobs,
            "random_state": random_seed,
            "verbosity": 0,
        }

        aucs, youdens, sens_spec = [], [], []
        inner_cv = GroupKFold(n_splits=n_inner_splits)

        for tr_i, va_i in inner_cv.split(X_outer_train, y_outer_train, g_outer_train):
            X_tr, y_tr, g_tr = X_outer_train.iloc[tr_i], y_outer_train.iloc[tr_i], g_outer_train.iloc[tr_i]
            X_va, y_va = X_outer_train.iloc[va_i], y_outer_train.iloc[va_i]

            X_fit, X_stop, y_fit, y_stop = _group_or_stratified_split(X_tr, y_tr, g_tr, 0.15, random_seed)

            clf, fitted_preproc = _fit_with_early_stopping(
                X_fit, y_fit, X_stop, y_stop, clone(preprocessor), params
            )
            preds = clf.predict_proba(fitted_preproc.transform(X_va))[:, 1]

            # store inner validation preds for later threshold selection
            fold_val_records.append(pd.DataFrame({
                "y_val_true": y_va.values,
                "y_val_score": preds,
            }))

            aucs.append(roc_auc_score(y_va, preds))
            j, _ = _youdens_j_from_scores(y_va, preds)
            youdens.append(j)

            spec, sens, _ = _best_sens_or_spec_thr(y_va, preds,
                                                   min_sens=min_sens,
                                                   min_spec=min_spec,
                                                   maximize=maximize)
            sens_spec.append(sens if maximize == "sens" else spec)

        return (np.mean(aucs), np.mean(youdens), np.mean(sens_spec))

    # ---------- Outer CV ----------
    for outer_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_outer_train, y_outer_train, g_outer_train = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
        X_outer_test, y_outer_test, g_outer_test = X.iloc[te_idx], y.iloc[te_idx], groups.iloc[te_idx]

        print(f"\n--- Outer Fold {outer_idx}/{n_outer_splits} | Train shape={X_outer_train.shape} ---")

        # -------------- Inner CV Optuna from the outer train--------------
        fold_val_records = []
        study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
        study.optimize(lambda trial: _objective(trial=trial,
                                                X_outer_train=X_outer_train,
                                                y_outer_train=y_outer_train,
                                                g_outer_train=g_outer_train,
                                                fold_val_records=fold_val_records),
                       n_trials=n_trials, show_progress_bar=True)

        # concatenate inner validation predictions
        fold_val_df = pd.concat(fold_val_records, ignore_index=True)

        # Get thresholds from inner CV only
        _, youden_thr = _youdens_j_from_scores(fold_val_df["y_val_true"], fold_val_df["y_val_score"])
        _, _, best_thr = _best_sens_or_spec_thr(fold_val_df["y_val_true"], fold_val_df["y_val_score"],
                                                min_sens=min_sens,
                                                min_spec=min_spec,
                                                maximize=maximize)

        # Best trials per objective
        best_trials = {
            "auc": max(study.best_trials, key=lambda t: t.values[0]),
            "youden": max(study.best_trials, key=lambda t: t.values[1]),
            "sens": max(study.best_trials, key=lambda t: t.values[2]),
        }

        for opt_type, trial in best_trials.items():
            params = trial.params
            print(f"[Fold {outer_idx}] Best {opt_type} | Values={trial.values} | Params={params}")

            # --------------  Retrain final model on full outer train --------------
            X_fit, X_stop, y_fit, y_stop = _group_or_stratified_split(
                X_outer_train, y_outer_train, g_outer_train, 0.15, random_seed
            )
            final_clf, final_preproc = _fit_with_early_stopping(
                X_fit, y_fit, X_stop, y_stop, clone(preprocessor),
                {**params, "objective": "binary:logistic", "eval_metric": eval_metric,
                 "tree_method": "hist", "device": "cuda", "n_estimators": n_estimators,
                 "early_stopping_rounds": early_stopping_rounds, "n_jobs": n_jobs,
                 "random_state": random_seed}
            )
            # ---------------- Predictions in the validation/test set --------------
            preds = final_clf.predict_proba(final_preproc.transform(X_outer_test))[:, 1]

            # Apply inner-derived thresholds on outer test preds
            metrics_standard = compute_metrics(y_outer_test, preds, thr_standard)
            metrics_standard.update({
                "model_type": "xgboost",  # actual model name
                "threshold_type": "standard",
                "fold": outer_idx,
                "optimization": opt_type
            })


            metrics_youden = compute_metrics(y_outer_test, preds, youden_thr)
            metrics_youden.update({
                "model_type": "xgboost",
                "threshold_type": "youden_j",
                "fold": outer_idx,
                "optimization": opt_type
            })


            metrics_custom = compute_metrics(y_outer_test, preds, best_thr)
            metrics_custom.update({
                "model_type": "xgboost",
                "threshold_type": f"{maximize}_max",
                "fold": outer_idx,
                "optimization": opt_type
            })

            all_fold_metrics.extend([metrics_standard, metrics_custom, metrics_youden])

            print(f"[Fold {outer_idx} | {opt_type}]")
            print("  " + _format_metrics_row("Standard", metrics_standard))
            print("  " + _format_metrics_row("Youden", metrics_youden))
            print("  " + _format_metrics_row("Custom", metrics_custom))
            print("-" * 50)  # separator between folds

            fold_df = pd.DataFrame({
                group_col: g_outer_test.values,

                "y_true": y_outer_test.values,
                "y_pred": preds,

                f"y_pred_{maximize}_max": (preds >= best_thr).astype(int),
                "y_pred_youden": (preds >= youden_thr).astype(int),
                "y_pred_standard": (preds >= thr_standard).astype(int),

                f"thr_{maximize}_max": best_thr,
                "thr_youden": youden_thr,
                "thr_standard": thr_standard,
                "fold": outer_idx,
                "optimization": opt_type,
            })
            all_fold_predictions.append(fold_df)

            joblib.dump(final_clf, fold_dir / f"fold_{outer_idx}_{opt_type}_model.pkl")
            joblib.dump(final_preproc, fold_dir / f"fold_{outer_idx}_{opt_type}_preprocessor.pkl")


            # ---------------- GPU cleanup ----------------
            del final_clf, final_preproc, preds
            gc.collect()
            torch.cuda.empty_cache()

            # ----------------------------------------------
            print(f"[Fold {outer_idx} | {opt_type}] Done and cleaned GPU memory")


    # Final results
    df_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    df_metrics = pd.DataFrame(all_fold_metrics)

    # ci and sort metrics columns
    df_metrics, _ = generate_ci(df_metrics=df_metrics)

    # sort metrics columns for better visualization
    cols_sens = [col for col in df_metrics.columns if col.startswith('sensitivity')]
    cols_spec = [col for col in df_metrics.columns if col.startswith('specificity')]
    cols_sens_spec = sorted(cols_sens + cols_spec)

    # include threshold_type in sorting
    df_metrics.sort_values(
        by=['model_type', 'threshold_type', 'optimization', 'fold'],
        ascending=[True, True, True, True],
        inplace=True
    )

    sort_keys = ['fold', 'model_type', 'threshold_type', 'optimization']
    other_cols = [col for col in df_metrics.columns
                  if col not in sort_keys + cols_sens_spec]

    df_metrics = df_metrics[
        sort_keys + cols_sens_spec + other_cols
        ]


    df_predictions.to_csv(results_dir / "predictions_actig.csv", index=False)
    df_metrics.to_csv(results_dir / "metrics_sctg.csv", index=False)

    return df_metrics, df_predictions





from joblib import Parallel, delayed

def train_nested_cv_xgb_optuna_parallel_cpu_only(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str,
    results_dir: Path,
    preprocessor: BaseEstimator,
    n_jobs: int = -1,
    n_jobs_outer: int = 1,   # <--- NEW: controls parallelization across folds
    random_seed: int = 42,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    n_trials: int = 50,
    n_estimators: int = 5000,
    eval_metric: str = "auc",
    early_stopping_rounds: int = 100,
    min_sens: float = 0.6,
    min_spec: float = 0.6,
    maximize: str = "spec",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def _youdens_j_from_scores(y_true, y_score):
        fpr, tpr, thr = roc_curve(y_true, y_score)
        spec = 1 - fpr
        j_stats = tpr + spec - 1
        best_idx = np.argmax(j_stats)
        return j_stats[best_idx], thr[best_idx]

    def _format_metrics_row(label, m):
        return f"{label:<8} Se/Sp = {m['sensitivity']:.1f}/{m['specificity']:.1f}% | AUC={m['auc']:.3f}"

    results_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = results_dir.joinpath("folds")
    fold_dir.mkdir(parents=True, exist_ok=True)
    thr_standard = 0.5
    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    outer_cv = GroupKFold(n_splits=n_outer_splits)

    # ---------- Inner Optuna Objective ----------
    def _objective(trial, X_outer_train, y_outer_train, g_outer_train, fold_val_records):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "binary:logistic",
            "eval_metric": eval_metric,
            "tree_method": "hist",   # <--- force CPU
            "device": "cpu",         # <--- force CPU
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "n_jobs": n_jobs,
            "random_state": random_seed,
            "verbosity": 0,
        }

        aucs, youdens, sens_spec = [], [], []
        inner_cv = GroupKFold(n_splits=n_inner_splits)

        for tr_i, va_i in inner_cv.split(X_outer_train, y_outer_train, g_outer_train):
            X_tr, y_tr, g_tr = X_outer_train.iloc[tr_i], y_outer_train.iloc[tr_i], g_outer_train.iloc[tr_i]
            X_va, y_va = X_outer_train.iloc[va_i], y_outer_train.iloc[va_i]

            X_fit, X_stop, y_fit, y_stop = _group_or_stratified_split(X_tr, y_tr, g_tr, 0.15, random_seed)

            clf, fitted_preproc = _fit_with_early_stopping(
                X_fit, y_fit, X_stop, y_stop, clone(preprocessor), params
            )
            preds = clf.predict_proba(fitted_preproc.transform(X_va))[:, 1]

            # store inner validation preds for later threshold selection
            fold_val_records.append(pd.DataFrame({
                "y_val_true": y_va.values,
                "y_val_score": preds,
            }))

            aucs.append(roc_auc_score(y_va, preds))
            j, _ = _youdens_j_from_scores(y_va, preds)
            youdens.append(j)

            spec, sens, _ = _best_sens_or_spec_thr(y_va, preds,
                                                   min_sens=min_sens,
                                                   min_spec=min_spec,
                                                   maximize=maximize)
            sens_spec.append(sens if maximize == "sens" else spec)

        return (np.mean(aucs), np.mean(youdens), np.mean(sens_spec))

    # ---------- Outer fold runner ----------
    def _run_outer_fold(outer_idx, tr_idx, te_idx):
        X_outer_train, y_outer_train, g_outer_train = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
        X_outer_test, y_outer_test, g_outer_test = X.iloc[te_idx], y.iloc[te_idx], groups.iloc[te_idx]

        print(f"\n--- Outer Fold {outer_idx}/{n_outer_splits} | Train shape={X_outer_train.shape} ---")

        fold_val_records = []
        study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
        study.optimize(lambda trial: _objective(trial=trial,
                                                X_outer_train=X_outer_train,
                                                y_outer_train=y_outer_train,
                                                g_outer_train=g_outer_train,
                                                fold_val_records=fold_val_records),
                       n_trials=n_trials, show_progress_bar=False)

        fold_val_df = pd.concat(fold_val_records, ignore_index=True)
        _, youden_thr = _youdens_j_from_scores(fold_val_df["y_val_true"], fold_val_df["y_val_score"])
        _, _, best_thr = _best_sens_or_spec_thr(fold_val_df["y_val_true"], fold_val_df["y_val_score"],
                                                min_sens=min_sens, min_spec=min_spec, maximize=maximize)

        # Best trials per objective
        best_trials = {
            "auc": max(study.best_trials, key=lambda t: t.values[0]),
            "youden": max(study.best_trials, key=lambda t: t.values[1]),
            f"{maximize}": max(study.best_trials, key=lambda t: t.values[2]),
        }

        fold_metrics, fold_predictions = [], []
        for opt_type, trial in best_trials.items():
            params = {
                **trial.params,
                "objective": "binary:logistic",
                "eval_metric": eval_metric,
                "tree_method": "hist",
                "device": "cpu",
                "n_estimators": n_estimators,
                "early_stopping_rounds": early_stopping_rounds,
                "n_jobs": n_jobs,
                "random_state": random_seed,
            }

            X_fit, X_stop, y_fit, y_stop = _group_or_stratified_split(
                X_outer_train, y_outer_train, g_outer_train, 0.15, random_seed
            )
            final_clf, final_preproc = _fit_with_early_stopping(
                X_fit, y_fit, X_stop, y_stop, clone(preprocessor), params
            )
            preds = final_clf.predict_proba(final_preproc.transform(X_outer_test))[:, 1]

            # Metrics
            metrics_standard = compute_metrics(y_outer_test, preds, 0.5)
            metrics_standard.update(dict(model_type="xgboost", threshold_type="standard",
                                         fold=outer_idx, optimization=opt_type))

            metrics_youden = compute_metrics(y_outer_test, preds, youden_thr)
            metrics_youden.update(dict(model_type="xgboost", threshold_type="youden_j",
                                       fold=outer_idx, optimization=opt_type))

            metrics_custom = compute_metrics(y_outer_test, preds, best_thr)
            metrics_custom.update(dict(model_type="xgboost", threshold_type=f"{maximize}_max",
                                       fold=outer_idx, optimization=opt_type))

            fold_metrics.extend([metrics_standard, metrics_custom, metrics_youden])

            fold_df = pd.DataFrame({
                group_col: g_outer_test.values,
                "y_true": y_outer_test.values,
                "y_pred": preds,
                f"y_pred_{maximize}_max": (preds >= best_thr).astype(int),
                "y_pred_youden": (preds >= youden_thr).astype(int),
                "y_pred_standard": (preds >= 0.5).astype(int),
                f"thr_{maximize}_max": best_thr,
                "thr_youden": youden_thr,
                "thr_standard": 0.5,
                "fold": outer_idx,
                "optimization": opt_type,
                'model_type': "xgboost",
            })
            fold_predictions.append(fold_df)

            joblib.dump(final_clf, fold_dir / f"fold_{outer_idx}_{opt_type}_model.pkl")
            joblib.dump(final_preproc, fold_dir / f"fold_{outer_idx}_{opt_type}_preprocessor.pkl")

        return fold_predictions, fold_metrics

    # ---------- Run folds in parallel ----------
    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(_run_outer_fold)(outer_idx, tr_idx, te_idx)
        for outer_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups), start=1)
    )

    # Collect results
    all_fold_predictions, all_fold_metrics = [], []
    for preds_list, metrics_list in results:
        all_fold_predictions.extend(preds_list)
        all_fold_metrics.extend(metrics_list)

    # Final results
    df_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    df_metrics = pd.DataFrame(all_fold_metrics)
    df_metrics, _ = generate_ci(df_metrics=df_metrics)

    # Sort
    cols_sens = [col for col in df_metrics.columns if col.startswith('sensitivity')]
    cols_spec = [col for col in df_metrics.columns if col.startswith('specificity')]
    cols_sens_spec = sorted(cols_sens + cols_spec)

    df_metrics.sort_values(by=['model_type', 'threshold_type', 'optimization', 'fold'],
                           ascending=[True, True, True, True], inplace=True)

    sort_keys = ['fold', 'model_type', 'threshold_type', 'optimization']
    other_cols = [c for c in df_metrics.columns if c not in sort_keys + cols_sens_spec]
    df_metrics = df_metrics[sort_keys + cols_sens_spec + other_cols]

    df_predictions.to_csv(results_dir / "predictions_actig.csv", index=False)
    df_metrics.to_csv(results_dir / "metrics_sctg.csv", index=False)
    return df_metrics, df_predictions



# %% update loing format
from library.ml_questionnaire.training import _get_model_and_space

def _pos_weight_ratio(y: np.ndarray) -> float:
    # n_neg / n_pos (guard against division by zero)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


def run_nested_cv_with_optuna_parallel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    col_id: str,                          # <-- subject/group id column
    results_dir: Path,
    continuous_cols: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    random_seed: int = 42,
    procerssor: bool = True,
    n_outer_splits: int = 10,
    n_inner_splits: int = 5,
    n_trials: int = 50,
    study_sampler=None,
    pos_weight: bool = False,
    min_sens: float = 0.6,
    min_spec: float = 0.6,
    maximize: str = "spec",               # "spec" or "sens"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested CV with Optuna hyperparameter tuning, using GROUP-STRATIFIED K-Fold.
    Groups = df[col_id] (e.g., subject_id). Ensures no group appears across train/valid
    within any split (outer or inner).

    Preprocessing policy (unchanged):
      - Impute ALL `feature_cols` that may have NaNs.
      - Standardize ONLY the `continuous_cols` subset.
      - Keep ALL features; preserve feature order.

    Implementation details (unchanged, except folds):
      - Build a template ColumnTransformer; clone & fit per split to avoid leakage.
      - Cache preprocessed inner folds once per outer split for Optuna trials.
      - Compute thresholds from inner OOF predictions for the chosen best trial.
    """
    # -------------------------------
    # imports (kept local for drop-in use)
    # -------------------------------
    import json, pickle
    from copy import deepcopy
    from pathlib import Path as _Path
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    # Prefer StratifiedGroupKFold (sklearn >=1.3). Fall back with a clear error.
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as e:
        raise ImportError(
            "StratifiedGroupKFold is required for group-stratified CV. "
            "Please upgrade scikit-learn (>= 1.3)."
        ) from e

    # -------------------------------
    # utilities
    # -------------------------------
    def _save_or_load_folds(path: _Path, cv, X, y, groups=None, random_seed=42):
        """
        Save folds (train/test indices) if not present; load them otherwise.
        Returns list of (train_idx, test_idx). Uses groups when provided.
        """
        path = _Path(path)
        if path.exists():
            with open(path, "rb") as f:
                folds = pickle.load(f)
            print(f"Loaded folds from {path}")
        else:
            # Try with groups first; if splitter doesn't accept groups, retry without.
            try:
                folds = list(cv.split(X, y, groups=groups))
            except TypeError:
                folds = list(cv.split(X, y))
            with open(path, "wb") as f:
                pickle.dump(folds, f)
            print(f"Saved folds to {path}")
        return folds

    def _assert_no_group_leak(train_idx, valid_idx, ids_array, where: str):
        """Assert that no group appears in both train and valid."""
        tr_g = set(ids_array[train_idx])
        va_g = set(ids_array[valid_idx])
        inter = tr_g.intersection(va_g)
        if inter:
            examples = list(sorted(inter))[:5]
            raise AssertionError(
                f"[{where}] Group leakage detected: {len(inter)} groups appear in both "
                f"train and validation. Examples: {examples}"
            )

    def _assert_no_nans(arr: np.ndarray, where: str):
        n = np.isnan(arr).sum()
        if n > 0:
            raise ValueError(f"NaNs remain after preprocessing in {where}: {n} NaNs")

    def _normalize_cols(cols, all_cols: Iterable[str]) -> Optional[List[str]]:
        if cols is None:
            return None
        if isinstance(cols, (str, np.str_)):
            cols = [cols]
        cols = list(cols)
        missing = set(cols) - set(all_cols)
        if missing:
            raise ValueError(f"Unknown columns in continuous_cols: {sorted(missing)}")
        return cols

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

    def _make_preprocessor(
            all_features: List[str],
            cont_cols: Optional[List[str]],
            scale_strategy: str = "none",  # "none" | "standard" | "minmax" | "robust" | "quantile"
    ) -> ColumnTransformer:
        numeric_cols = cont_cols or []
        other_cols = [c for c in all_features if c not in numeric_cols]

        # choose scaler
        scaler = None
        if scale_strategy == "standard":
            scaler = StandardScaler()
        elif scale_strategy == "minmax":
            scaler = MinMaxScaler()
        elif scale_strategy == "robust":
            scaler = RobustScaler()
        elif scale_strategy == "quantile":
            scaler = QuantileTransformer(output_distribution="normal", subsample=2_000_000, random_state=0)
        elif scale_strategy != "none":
            raise ValueError(f"Unknown scale_strategy: {scale_strategy}")

        transformers = []
        if numeric_cols:
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if scaler is not None:
                steps.append(("scaler", scaler))
            transformers.append(("num", Pipeline(steps), numeric_cols))

        if other_cols:
            transformers.append(("other", SimpleImputer(strategy="most_frequent"), other_cols))

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        try:
            ct.set_output(transform="pandas")
        except Exception:
            pass
        return ct

    def _compute_tau_youden(inner_df: pd.DataFrame, col_true="y_true", col_score="y_score"):
        if inner_df.empty:
            return 0.0, 0.5
        return youdens_j_from_scores(inner_df[col_true].values, inner_df[col_score].values)

    def _extract_params_per_run(df: pd.DataFrame) -> pd.DataFrame:
        thr_order = {"youden": 0, "0p5": 1}
        def _thr_rank(s: str) -> int:
            if s in thr_order: return thr_order[s]
            return 2 if str(s).endswith("_max") else 9

        d = df.copy()
        d["_thr_rank"] = d["threshold"].map(_thr_rank)
        out = (
            d.sort_values(["outer_fold", "model_type", "optimization", "_thr_rank"])
            .drop_duplicates(subset=["outer_fold", "model_type", "optimization"], keep="first")
            [["outer_fold", "model_type", "optimization", "best_params_json"]]
            .reset_index(drop=True)
        )
        return out

    def compute_ci_from_folds_average(
            df_metrics_per_fold: pd.DataFrame,
            group_cols: List[str] = None,
            col_metrics: List[str] = None,
            suffix: str = "_ci"
    ) -> pd.DataFrame:
        if group_cols is None:
            group_cols = ["model_type", "optimization", "threshold"]
        if col_metrics is None:
            col_metrics = ["auc_score", "prc_score", "sensitivity", "specificity"]

        df = df_metrics_per_fold.copy()
        for c in col_metrics:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        grp = df.groupby(group_cols, dropna=False)[col_metrics]
        means = grp.mean()
        counts = grp.count()
        stds = grp.std(ddof=1)

        tcrit = stats.t.ppf(0.975, df=counts - 1)
        sem = stds / np.sqrt(counts)
        half_width = sem * tcrit

        ci_lower = means - half_width
        ci_upper = means + half_width

        def _clamp(v: float, scale: float) -> float:
            if pd.isna(v):
                return v
            lo, hi = (0.0, 100.0) if scale == 100.0 else (0.0, 1.0)
            return min(max(v, lo), hi)

        def _fmt_val(v: float, scale: float) -> str:
            if pd.isna(v):
                return "NA"
            if scale == 100.0:
                return f"{v:.1f}%"
            return f"{v:.3f}"

        import re
        def _extract_first_numeric(series: pd.Series) -> pd.Series:
            return series.apply(
                lambda s: re.search(r'[-+]?\d*\.\d+|\d+', s).group() if isinstance(s, str) and re.search(
                    r'[-+]?\d*\.\d+|\d+', s) else None
            )

        out = means.reset_index()[group_cols].copy()
        idx_iter = means.index
        for m in col_metrics:
            formatted = []
            for idx in idx_iter:
                mean_v = float(means.loc[idx, m])
                low_v = float(ci_lower.loc[idx, m]) if not pd.isna(ci_lower.loc[idx, m]) else np.nan
                high_v = float(ci_upper.loc[idx, m]) if not pd.isna(ci_upper.loc[idx, m]) else np.nan
                n_v = counts.loc[idx, m]
                scale = 100.0 if (not pd.isna(mean_v) and mean_v > 1.5) else 1.0
                if pd.isna(n_v) or n_v <= 1:
                    s = f"{_fmt_val(mean_v, scale)} (NA, NA)"
                else:
                    low_v = _clamp(low_v, scale); high_v = _clamp(high_v, scale)
                    s = f"{_fmt_val(mean_v, scale)} ({_fmt_val(low_v, scale)}, {_fmt_val(high_v, scale)})"
                formatted.append(s)
            out[f"{m}{suffix}"] = formatted

        for m in col_metrics:
            out[m] = _extract_first_numeric(out[m + suffix]).astype(float)
        return out

    def _fit_transform_split(
        X_tr_df: pd.DataFrame,
        y_tr_ser: pd.Series,
        X_va_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if preprocessor_template is None:
            return X_tr_df.to_numpy(), X_va_df.to_numpy(), list(X_tr_df.columns)

        preproc = clone(preprocessor_template)
        X_tr_tx = preproc.fit_transform(X_tr_df, y_tr_ser)
        X_va_tx = preproc.transform(X_va_df)

        if isinstance(X_tr_tx, pd.DataFrame):
            X_tr_df_tx = X_tr_tx.copy(); X_va_df_tx = X_va_tx.copy()
        else:
            try:
                feat_names_all = list(preproc.get_feature_names_out(input_features=X_tr_df.columns))
            except Exception:
                feat_names_all = list(X_tr_df.columns)
            X_tr_df_tx = pd.DataFrame(X_tr_tx, index=X_tr_df.index, columns=feat_names_all)
            X_va_df_tx = pd.DataFrame(X_va_tx, index=X_va_df.index, columns=feat_names_all)

        _assert_no_nans(X_tr_df_tx.to_numpy(), "train")
        _assert_no_nans(X_va_df_tx.to_numpy(), "valid/test")

        missing_in_tx = [c for c in feature_cols if c not in X_tr_df_tx.columns]
        if missing_in_tx:
            raise ValueError(
                f"Transformed features missing expected columns: {missing_in_tx}. "
                f"Available: {list(X_tr_df_tx.columns)}"
            )
        X_tr_df_tx = X_tr_df_tx[feature_cols]
        X_va_df_tx = X_va_df_tx[feature_cols]

        if X_tr_df_tx.shape[1] != expected_out_dim or X_va_df_tx.shape[1] != expected_out_dim:
            raise ValueError(
                f"Unexpected transformed width. Got train {X_tr_df_tx.shape[1]}, "
                f"valid {X_va_df_tx.shape[1]}, expected {expected_out_dim}."
            )
        return X_tr_df_tx.to_numpy(), X_va_df_tx.to_numpy(), list(feature_cols)

    def _run_optuna_with_cache(
        model_type: str,
        X_tr_df: pd.DataFrame, y_tr_ser: pd.Series,
        tr_idx: np.ndarray, outer_idx: int,
        groups_tr: np.ndarray
    ) -> Tuple[optuna.Study, Callable[[Dict[str, Any]], pd.DataFrame]]:
        """
        Inner CV uses StratifiedGroupKFold on the OUTER-TRAIN subset.
        Returns:
          - study: the optuna study after optimization
          - inner_oof_for_trial(params): recomputes concatenated inner OOF predictions for a fixed trial
        """
        # Build inner splitter (group + stratified)
        inner_cv = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)

        # Cache preprocessed inner folds once per outer split
        fold_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for inner_idx, (inner_tr, inner_va) in enumerate(inner_cv.split(X_tr_df, y_tr_ser, groups=groups_tr), start=1):
            _assert_no_group_leak(inner_tr, inner_va, groups_tr, where=f"Inner {inner_idx}/Outer {outer_idx}")

            Xt = X_tr_df.iloc[inner_tr]; Xv = X_tr_df.iloc[inner_va]
            yt = y_tr_ser.iloc[inner_tr]; yv = y_tr_ser.iloc[inner_va]

            Xt_proc, Xv_proc, _ = _fit_transform_split(Xt, yt, Xv)
            fold_cache.append((Xt_proc, Xv_proc, yt.to_numpy(), yv.to_numpy(), inner_va))

        pos_weight_ratio = _pos_weight_ratio(y_tr_ser) if pos_weight else None

        def objective(trial):
            model, _ = _get_model_and_space(model_type, trial, pos_weight_ratio=pos_weight_ratio)
            scores_auc, scores_youden, sens_spec = [], [], []

            for (Xt_proc, Xv_proc, yt_np, yv_np, _) in fold_cache:
                mdl = clone(model).fit(Xt_proc, yt_np)
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(Xv_proc)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(Xv_proc)
                else:
                    y_score = mdl.predict(Xv_proc).astype(float)

                scores_auc.append(roc_auc_score(yv_np, y_score))
                j, _ = youdens_j_from_scores(yv_np, y_score)
                scores_youden.append(j)
                spec, sens, _ = _best_sens_or_spec_thr(yv_np, y_score, min_spec=min_spec)
                sens_spec.append(sens if maximize == "sens" else spec)

            return (np.mean(scores_auc), np.mean(scores_youden), np.mean(sens_spec))

        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=study_sampler
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

        # helper: concatenated inner OOF predictions for a fixed trial
        def inner_oof_for_trial(trial_params: Dict[str, Any]) -> pd.DataFrame:
            model, _ = _get_model_and_space(
                model_type=model_type,
                trial=optuna.trial.FixedTrial(trial_params),
                pos_weight_ratio=pos_weight_ratio
            )
            y_true_all, y_score_all, subj_ids_all = [], [], []
            for (Xt_proc, Xv_proc, yt_np, yv_np, inner_va) in fold_cache:
                mdl = clone(model).fit(Xt_proc, yt_np)
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(Xv_proc)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(Xv_proc)
                else:
                    y_score = mdl.predict(Xv_proc).astype(float)

                y_true_all.append(yv_np.astype(int))
                y_score_all.append(y_score.astype(float))
                subj_ids_all.append(ids_np[tr_idx][inner_va])

            return pd.DataFrame({
                "subject_id": np.concatenate(subj_ids_all),
                "y_true":     np.concatenate(y_true_all),
                "y_score":    np.concatenate(y_score_all),
            })

        # Traceability: store inner OOF for the AUC-best trial
        best_auc_trial = max(study.best_trials, key=lambda t: t.values[0])
        df_trace = inner_oof_for_trial(best_auc_trial.params).rename(
            columns={"y_true": "y_val_true", "y_score": "y_val_score"}
        )
        df_trace["outer_fold"] = outer_idx
        df_trace["model_type"] = model_type
        nonlocal df_inner_val_records
        df_inner_val_records = pd.concat([df_inner_val_records, df_trace], ignore_index=True)

        return study, inner_oof_for_trial

    # -------------------------------
    # setup
    # -------------------------------
    if model_types is None:
        model_types = ["random_forest", "lightgbm", "xgboost", "elastic_net"]

    # Remove id from features if present
    feature_cols = [c for c in feature_cols if c != col_id]

    results_dir.mkdir(parents=True, exist_ok=True)

    # Keep as DataFrame/Series to preserve column names
    X = df[feature_cols]
    y = df[target_col]
    ids_np = df[col_id].to_numpy()        # <-- GROUPS

    # Build preprocessor template
    preprocessor_template = None
    if procerssor:
        cont_cols = _normalize_cols(continuous_cols, X.columns)  # may be None
        preprocessor_template = _make_preprocessor(all_features=feature_cols,
                                                   cont_cols=cont_cols,
                                                   scale_strategy='none')

    expected_out_dim = len(feature_cols)

    # -------------------------------
    # CV splitters (GROUP + STRATIFIED)
    # -------------------------------
    outer_cv = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_seed)
    outer_folds = _save_or_load_folds(results_dir / "outer_cv.pkl", outer_cv, X.to_numpy(), y.to_numpy(), groups=ids_np)

    # -------------------------------
    # outputs
    # -------------------------------
    df_outer_metrics_records = pd.DataFrame()
    df_outer_predictions_records = pd.DataFrame()
    df_inner_val_records = pd.DataFrame()  # traceability: inner-CV predictions

    def _mk_row(threshold_name: str, threshold_value, met_dict: Dict[str, float]):
        return {
            "outer_fold": outer_idx,
            "model_type": model_type,
            "optimization": opt_type,
            "threshold": threshold_name,
            "threshold_value": float(threshold_value),
            "auc_score": met_dict.get("auc_score", np.nan),
            "prc_score": met_dict.get("prc_score", np.nan),
            "sensitivity": met_dict.get("sensitivity", np.nan),
            "specificity": met_dict.get("specificity", np.nan),
            "best_params_json": json.dumps(params),
        }

    # -------------------------------
    # Outer CV
    # -------------------------------
    for model_type in model_types:
        for outer_idx, (tr_idx, te_idx) in enumerate(outer_folds, start=1):
            # Assert no group leakage in OUTER split
            _assert_no_group_leak(tr_idx, te_idx, ids_np, where=f"Outer {outer_idx}")

            print(f"\n=== Outer {outer_idx}/{n_outer_splits} | {model_type}  | Feature Set k={len(feature_cols)} ===")
            X_tr = X.iloc[tr_idx]; X_te = X.iloc[te_idx]
            y_tr = y.iloc[tr_idx]; y_te = y.iloc[te_idx]
            ids_te = ids_np[te_idx]
            groups_tr = ids_np[tr_idx]   # groups restricted to outer-train

            # ---- Optuna on inner CV (with cached preprocessed folds)
            study, inner_oof_for_trial = _run_optuna_with_cache(
                model_type=model_type,
                X_tr_df=X_tr, y_tr_ser=y_tr,
                tr_idx=tr_idx, outer_idx=outer_idx,
                groups_tr=groups_tr
            )

            # pick best trials (one per objective)
            best_trials = {
                "auc": max(study.best_trials, key=lambda t: t.values[0]),
                "youden": max(study.best_trials, key=lambda t: t.values[1]),
                f"max{maximize}": max(study.best_trials, key=lambda t: t.values[2]),
            }

            # ---- Evaluate each chosen trial on the outer test fold
            for n_scoring, (opt_type, trial) in enumerate(best_trials.items(), start=0):
                params = trial.params
                print(f"[Outer {outer_idx}] {model_type} | Best {opt_type}: {trial.values[n_scoring]:2f}")
                print("Params:")
                for k, v in params.items():
                    print(f"  - {k}: {v}")

                # thresholds from inner OOF predictions for THIS trial
                inner_preds_df = inner_oof_for_trial(params)
                youden_score, youden_threshold = _compute_tau_youden(inner_preds_df, "y_true", "y_score")
                _, _, thr_sens_spec_max = _best_sens_or_spec_thr(
                    y_true=inner_preds_df["y_true"],
                    y_score=inner_preds_df["y_score"],
                    min_sens=min_sens,
                    min_spec=min_spec,
                    maximize=maximize,
                )

                # train best model on full outer-train
                best_model, _ = _get_model_and_space(
                    model_type=model_type,
                    trial=optuna.trial.FixedTrial(params),
                    pos_weight_ratio=_pos_weight_ratio(y_tr) if pos_weight else None,
                )
                X_tr_proc, X_te_proc, feature_names = _fit_transform_split(X_tr, y_tr, X_te)
                best_model.fit(X_tr_proc, y_tr.to_numpy())

                # predict on outer test
                if hasattr(best_model, "predict_proba"):
                    y_score_te = best_model.predict_proba(X_te_proc)[:, 1]
                elif hasattr(best_model, "decision_function"):
                    y_score_te = best_model.decision_function(X_te_proc)
                else:
                    y_score_te = best_model.predict(X_te_proc).astype(float)

                # ---------- Apply Thresholds ----------
                met_tau = metrics_at_threshold(y_te, y_score_te, youden_threshold)
                met_05 = metrics_at_threshold(y_te, y_score_te, 0.5)
                met_sens_spec = metrics_at_threshold(y_te, y_score_te, thr_sens_spec_max)

                # ---------- Predictions in long format ----------
                pred_dfs = []
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": "youden",
                    "threshold_value": float(youden_threshold),
                    "y_pred": (y_score_te >= youden_threshold).astype(int),
                }))
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": "0p5",
                    "threshold_value": 0.5,
                    "y_pred": (y_score_te >= 0.5).astype(int),
                }))
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": f"{maximize}_max",
                    "threshold_value": float(thr_sens_spec_max),
                    "y_pred": (y_score_te >= thr_sens_spec_max).astype(int),
                }))
                records_df = pd.concat(pred_dfs, ignore_index=True)
                df_outer_predictions_records = pd.concat(
                    [df_outer_predictions_records, records_df], ignore_index=True
                )

                # ---------- Metrics ----------
                rows_long = [
                    _mk_row("youden", youden_threshold, met_tau),
                    _mk_row("0p5", 0.5, met_05),
                    _mk_row(f"{maximize}_max", thr_sens_spec_max, met_sens_spec),
                ]
                df_outer_metrics_records = pd.concat(
                    [df_outer_metrics_records, pd.DataFrame(rows_long)],
                    ignore_index=True
                )

                # ---------- Pretty Printing ----------
                print(
                    f"\n[Outer {outer_idx:02d}] {model_type} | {opt_type}"
                    f"\n  Thresholds:"
                    f"\n    Youden (tau*)       = {youden_threshold:.3f}"
                    f"\n    Fixed 0.5           = 0.500"
                    f"\n    {maximize}_max      = {thr_sens_spec_max:.3f}"
                    f"\n"
                    f"\n  Metrics:"
                    f"\n    AUC (tau*)          = {met_tau.get('auc_score'):.3f}"
                    f"\n    PRC (tau*)          = {met_tau.get('prc_score'):.3f}"
                    f"\n    Sens/Spec (tau*)    = {met_tau.get('sensitivity'):.3f} / {met_tau.get('specificity'):.3f}"
                    f"\n"
                    f"\n    AUC (0.5)           = {met_05.get('auc_score'):.3f}"
                    f"\n    PRC (0.5)           = {met_05.get('prc_score'):.3f}"
                    f"\n    Sens/Spec (0.5)     = {met_05.get('sensitivity'):.3f} / {met_05.get('specificity'):.3f}"
                    f"\n"
                    f"\n    AUC ({maximize}_max) = {met_sens_spec.get('auc_score'):.3f}"
                    f"\n    PRC ({maximize}_max) = {met_sens_spec.get('prc_score'):.3f}"
                    f"\n    Sens/Spec ({maximize}_max) = {met_sens_spec.get('sensitivity'):.3f} / {met_sens_spec.get('specificity'):.3f}"
                )

    # ---------- Confidence Intervals, sort, save ----------
    metric_cols = ['auc_score', 'prc_score', 'sensitivity', 'specificity', 'threshold_value']
    group_cols = ["model_type", "optimization", "threshold"]

    df_outer_metrics_records_ci = compute_ci_from_folds_average(
        df_metrics_per_fold=df_outer_metrics_records,
        group_cols=group_cols,
        col_metrics=metric_cols,
        suffix='_ci'
    )

    params_per_run = _extract_params_per_run(df_outer_metrics_records)
    per_fold_param = (
        df_outer_metrics_records
        .sort_values(by=["optimization", "threshold", "model_type"])
        .drop_duplicates(subset=["optimization", "threshold", "model_type"], keep="first")
        [group_cols + ["best_params_json"]]
    )

    df_outer_metrics_records_ci.sort_values(
        by=["model_type", "optimization", "threshold"],
        ascending=[True, True, True],
        inplace=True
    )

    df_outer_metrics_records_ci.to_csv(results_dir / "df_outer_metrics_records_ci.csv", index=False)
    df_outer_metrics_records.to_csv(results_dir / "metrics_outer_folds.csv", index=False)
    df_outer_predictions_records.to_csv(results_dir / "predictions_outer_folds.csv", index=False)
    df_inner_val_records.to_csv(results_dir / "inner_val_records.csv", index=False)
    params_per_run.to_csv(results_dir / "params_per_run.csv", index=False)
    per_fold_param.to_csv(results_dir / "per_fold_param.csv", index=False)

    print(f"\nSaved outer metrics and predictions in {results_dir.resolve()}")
    return df_outer_metrics_records_ci, df_outer_predictions_records, df_inner_val_records
