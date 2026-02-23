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
import inspect
import xgboost as xgb


def _get_device():
    try:
        dtrain = xgb.DMatrix([[0],[1]], label=[0,1])
        xgb.train({"tree_method": "hist", "device": "cuda"}, dtrain, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = _get_device(); print(DEVICE)


# ---------------------- Helper Functions ----------------------



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


def _pos_weight_ratio(y: np.ndarray) -> float:
    # n_neg / n_pos (guard against division by zero)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


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




# %%
# ---------------------------------------------------
# Threshold finder
# ---------------------------------------------------

# def _get_model_and_space(model_type:str,
#                          trial,
#                          pos_weight_ratio=None,
#                          n_jobs: int | None = 1,
#                          random_seed: int | None = 42,
#                          early_stopping_rounds: int = 200,
#                          use_gpu: bool = True):
#     if model_type == "xgboost":
#         low  = (pos_weight_ratio or 1.0) * 0.8
#         high = (pos_weight_ratio or 1.0) * 1.2
#         params = {
#             "max_depth": trial.suggest_int("max_depth", 3, 18),
#             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
#             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#             "objective": "binary:logistic",
#             # tree_method 'hist' + device='cuda' uses the GPU in XGB 2.x
#             "tree_method": "hist",
#             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.9),
#             "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 4),
#             "scale_pos_weight": trial.suggest_float("scale_pos_weight", low, high),
#         }
#         model = xgb.XGBClassifier(
#             **params,
#             n_estimators=5000,
#             eval_metric="auc",                 # pick a metric to **maximize** for ES
#             random_state=random_seed,
#             n_jobs=n_jobs,
#             verbosity=0,
#             device="cuda" if use_gpu else "cpu",
#             early_stopping_rounds=early_stopping_rounds,
#         )
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")
#     return model, params

# %% update loing format
def _get_model_and_space(model_type,
                         trial,
                         pos_weight_ratio: float | None = None,
                         n_jobs: int | None = 1,
                         random_seed: int | None = 42,
                         early_stopping_rounds: int = 200,
                         use_gpu: bool = True):
    """
    Return (estimator, params_dict_used) for a given model_type using Optuna trial.
    Early stopping is enabled at construction; you MUST pass eval_set in .fit(...) to activate it.
    """
    if model_type == "xgboost":
        low  = (pos_weight_ratio or 1.0) * 0.8
        high = (pos_weight_ratio or 1.0) * 1.2

        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 18),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective":        "binary:logistic",
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 0.9),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 4),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", low, high),
            # "n_estimators": trial.suggest_int("n_estimators", 200, 5000),
        }

        # Backward-compatible device handling (XGBoost 1.x vs 2.x)
        has_device_param = "device" in inspect.signature(xgb.XGBClassifier).parameters

        common = dict(
            n_estimators=5000,
            eval_metric="auc",               # good for ES and aligns with your objectives
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
        )
        if has_device_param:
            # XGBoost >= 2.0
            common.update(tree_method="hist", device=("cuda" if use_gpu else "cpu"))
        else:
            # XGBoost < 2.0
            common.update(
                tree_method=("gpu_hist" if use_gpu else "hist"),
                predictor=("gpu_predictor" if use_gpu else "auto")
            )

        model = xgb.XGBClassifier(**params, **common)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, params


def run_nested_cv_with_optuna_parallel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    col_id: str,                          # group id (e.g., subject_id)
    results_dir: Path,
    continuous_cols: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,  # set to ["xgboost"] unless you implement others
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
    outer_use_es: bool = True,
    outer_es_val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested CV with Optuna hyperparameter tuning, using StratifiedGroupKFold.
    - Inner loop uses ES via eval_set on the validation fold.
    - Outer final fit can also use ES by carving a small validation slice from the outer-train.

    Outputs:
      df_outer_metrics_records_ci, df_outer_predictions_records, df_inner_val_records
    """
    # -------------------------------
    # imports (kept local for drop-in use)
    # -------------------------------
    import json, pickle, re, inspect
    from copy import deepcopy
    from pathlib import Path as _Path
    import numpy as np
    import pandas as pd
    from tabulate import tabulate
    import optuna
    import xgboost as xgb
    from scipy import stats
    from typing import Any, Callable, Iterable, Dict, Tuple, List, Optional
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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
        path = _Path(path)
        if path.exists():
            with open(path, "rb") as f:
                folds = pickle.load(f)
            print(f"Loaded folds from {path}")
        else:
            try:
                folds = list(cv.split(X, y, groups=groups))
            except TypeError:
                folds = list(cv.split(X, y))
            with open(path, "wb") as f:
                pickle.dump(folds, f)
            print(f"Saved folds to {path}")
        return folds

    def _assert_no_group_leak(train_idx, valid_idx, ids_array, where: str):
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
            lo, hi = (0.0, 100.0) if (scale == 100.0) else (0.0, 1.0)
            return min(max(v, lo), hi)

        def _fmt_val(v: float, scale: float) -> str:
            if pd.isna(v):
                return "NA"
            if scale == 100.0:
                return f"{v:.1f}%"
            return f"{v:.3f}"

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

    def _pick_sgkf_val_split(X_tr, y_tr, groups_tr, outer_es_val_frac, random_seed):
        """
        Choose a single (sub-train, val) split from StratifiedGroupKFold so that
        val ≈ outer_es_val_frac of outer-train, subject to feasibility.
        """
        # target folds ~ 1/val_frac (clip to [2, 10])
        if 0.0 < outer_es_val_frac < 0.5:
            target_splits = int(round(1.0 / outer_es_val_frac))
            n_splits = min(max(target_splits, 2), 10)
        else:
            n_splits = 5

        # cannot exceed number of unique groups
        import numpy as np
        n_groups = len(np.unique(groups_tr))
        n_splits = min(n_splits, max(2, n_groups))  # at least 2, at most n_groups

        # try decreasing n_splits until a split works
        last_err = None
        for k in range(n_splits, 1, -1):
            try:
                sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_seed)
                # take the first fold as validation
                tr_sub_idx_rel, tr_val_idx_rel = next(sgkf.split(X_tr, y_tr, groups=groups_tr))
                return tr_sub_idx_rel, tr_val_idx_rel
            except Exception as e:
                last_err = e
                continue

        # if we get here, SGKF could not split
        raise RuntimeError(
            f"StratifiedGroupKFold could not produce a split for outer ES; "
            f"consider lowering outer_es_val_frac or check class/group balance. "
            f"Last error: {last_err}"
        )

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
        Uses early stopping by passing eval_set=(X_val, y_val) at fit-time.
        Returns:
          - study: the optuna study after optimization
          - inner_oof_for_trial(params): concatenated inner OOF predictions for a fixed trial
        """
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

        def _predict_proba_with_best(mdl, X):
            # Prefer iteration_range (XGB 2.x), fallback to ntree_limit (1.x)
            if hasattr(mdl, "best_iteration") and mdl.best_iteration is not None:
                try:
                    return mdl.predict_proba(X, iteration_range=(0, mdl.best_iteration + 1))[:, 1]
                except TypeError:
                    return mdl.predict_proba(X, ntree_limit=mdl.best_iteration + 1)[:, 1]
            return mdl.predict_proba(X)[:, 1]

        def objective(trial):
            model, _ = _get_model_and_space(
                model_type, trial, pos_weight_ratio=pos_weight_ratio,
                n_jobs=1, random_seed=random_seed
            )
            scores_auc, scores_youden, sens_spec = [], [], []

            for (Xt_proc, Xv_proc, yt_np, yv_np, _) in fold_cache:
                mdl = clone(model)
                mdl.fit(
                    Xt_proc, yt_np,
                    eval_set=[(Xv_proc, yv_np)],
                    verbose=False
                )
                y_score = _predict_proba_with_best(mdl, Xv_proc)

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
                pos_weight_ratio=pos_weight_ratio,
                n_jobs=1,
                random_seed=random_seed
            )
            y_true_all, y_score_all, subj_ids_all = [], [], []
            for (Xt_proc, Xv_proc, yt_np, yv_np, inner_va) in fold_cache:
                mdl = clone(model)
                mdl.fit(
                    Xt_proc, yt_np,
                    eval_set=[(Xv_proc, yv_np)],
                    verbose=False
                )
                y_score = _predict_proba_with_best(mdl, Xv_proc)

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

    def _mk_row(threshold_name: str, threshold_value, met_dict: Dict[str, float]):
        """
        Convert metrics and predictions into long format for easier computation of metrics and model evaluation
        :param threshold_name:
        :param threshold_value:
        :param met_dict:
        :return:
        """
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
    # setup
    # -------------------------------
    if model_types is None:
        model_types = ["xgboost"]  # keep to xgboost unless others are implemented in _get_model_and_space

    feature_cols = [c for c in feature_cols if c != col_id]  # remove id from features if present
    results_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    y = df[target_col]
    ids_np = df[col_id].to_numpy()        # GROUPS

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

    model_dir = results_dir.joinpath('folds')
    model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # outputs
    # -------------------------------
    df_outer_metrics_records = pd.DataFrame()
    df_outer_predictions_records = pd.DataFrame()
    df_inner_val_records = pd.DataFrame()  # inner-CV predictions
    df_pretty_records = pd.DataFrame()

    # ------ define directory
    metrics_dir = results_dir / "metrics"
    preds_dir = results_dir / "predictions"
    params_dir = results_dir / "params"
    logs_dir = results_dir / "logs"
    models_dir = results_dir / "models"
    for d in [metrics_dir, preds_dir, params_dir, logs_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)
    (params_dir / "best_params").mkdir(parents=True, exist_ok=True)
    # -------------------------------
    # Outer CV
    # -------------------------------
    for model_type in model_types:
        for outer_idx, (tr_idx, te_idx) in enumerate(outer_folds, start=1):
            _assert_no_group_leak(tr_idx, te_idx, ids_np, where=f"Outer {outer_idx}")
            print(f"\n=== Outer {outer_idx}/{n_outer_splits} | {model_type}  | Feature Set k={len(feature_cols)} ===")
            X_tr = X.iloc[tr_idx]; X_te = X.iloc[te_idx]
            y_tr = y.iloc[tr_idx]; y_te = y.iloc[te_idx]
            ids_te = ids_np[te_idx]
            groups_tr = ids_np[tr_idx]   # groups restricted to outer-train

            # ---- Optuna on inner CV (with cached preprocessed folds & ES)
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
                model_dir_model = model_dir.joinpath(model_type, opt_type)
                model_dir_model.mkdir(parents=True, exist_ok=True)

                params = trial.params
                print(f"[Outer {outer_idx}] {model_type} | Best {opt_type}: {trial.values[n_scoring]:.6f}")
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

                # train best model on full outer-train (with optional ES via a small val slice)
                best_model, _ = _get_model_and_space(
                    model_type=model_type,
                    trial=optuna.trial.FixedTrial(params),
                    pos_weight_ratio=_pos_weight_ratio(y_tr) if pos_weight else None,
                    random_seed=random_seed
                )
                X_tr_proc, X_te_proc, _ = _fit_transform_split(X_tr, y_tr, X_te)

                # ---------- Early Stop training ----------
                # Only run early stopping (ES) on the outer-train data if True and validation fraction (0 and 50%).
                if outer_use_es and 0.0 < outer_es_val_frac < 0.5:
                    # indices are relative to the OUTER-TRAIN set
                    tr_sub_idx, tr_val_idx = _pick_sgkf_val_split(
                        X_tr, y_tr, groups_tr, outer_es_val_frac, random_seed
                    )

                    y_tr_np = y_tr.to_numpy()
                    best_model.fit(
                        X_tr_proc[tr_sub_idx], y_tr_np[tr_sub_idx],
                        eval_set=[(X_tr_proc[tr_val_idx], y_tr_np[tr_val_idx])],
                        verbose=False
                    )

                    # predict with best iteration if available (XGBoost 2.x API first)
                    try:
                        y_score_te = best_model.predict_proba(
                            X_te_proc, iteration_range=(0, best_model.best_iteration + 1)
                        )[:, 1]
                    except Exception:
                        if hasattr(best_model, "best_iteration") and best_model.best_iteration is not None:
                            y_score_te = best_model.predict_proba(
                                X_te_proc, ntree_limit=best_model.best_iteration + 1
                            )[:, 1]
                        else:
                            y_score_te = best_model.predict_proba(X_te_proc)[:, 1]
                else:
                    # No outer ES: fit on full outer-train
                    best_model.fit(X_tr_proc, y_tr.to_numpy())
                    y_score_te = (best_model.predict_proba(X_te_proc)[:, 1]
                                  if hasattr(best_model, "predict_proba")
                                  else best_model.decision_function(X_te_proc))

                # ---------- Save model ----------
                model_out = models_dir / model_type
                model_out.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_model, model_out / f"outer{outer_idx}_{opt_type}.pkl")


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

                pretty_rows = [
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": "Youden",
                        "Thr_Val": youden_threshold,
                        "AUC": met_tau.get("auc_score"),
                        "PRC": met_tau.get("prc_score"),
                        "Sens": met_tau.get("sensitivity"),
                        "Spec": met_tau.get("specificity"),
                    },
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": "0.5",
                        "Thr_Val": 0.5,
                        "AUC": met_05.get("auc_score"),
                        "PRC": met_05.get("prc_score"),
                        "Sens": met_05.get("sensitivity"),
                        "Spec": met_05.get("specificity"),
                    },
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": f"{maximize}_max",
                        "Thr_Val": thr_sens_spec_max,
                        "AUC": met_sens_spec.get("auc_score"),
                        "PRC": met_sens_spec.get("prc_score"),
                        "Sens": met_sens_spec.get("sensitivity"),
                        "Spec": met_sens_spec.get("specificity"),
                    }
                ]
                # Convert to dataframe
                df_pretty = pd.DataFrame(pretty_rows)
                df_pretty.to_csv(
                    logs_dir / f"log_outer{outer_idx}_{model_type}_opt_{opt_type}.csv",
                    index=False
                )
                # Print with tabulate
                print("\n" + tabulate(df_pretty, headers="keys", tablefmt="psql", showindex=False))
                df_pretty_records = pd.concat([df_pretty_records, df_pretty], ignore_index=True)

                with open(params_dir / "best_params" / f"{model_type}_outer{outer_idx}_{opt_type}.json", "w") as f:
                    json.dump(params, f, indent=2)
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

    # ---------- Confidence Intervals, sort, save ----------
    df_outer_metrics_records.to_csv(metrics_dir   / "metrics_outer_folds.csv", index=False)

    df_outer_metrics_records_ci.to_csv(metrics_dir   / "metrics_outer_folds_ci.csv", index=False)

    df_outer_predictions_records.to_csv(preds_dir   / "predictions_outer_folds.csv", index=False)

    df_inner_val_records.to_csv(preds_dir   / "inner_val_records.csv", index=False)

    params_per_run.to_csv(params_dir   / "params_per_run.csv", index=False)

    per_fold_param.to_csv(params_dir   / "per_fold_param.csv", index=False)

    tbs_print = ['model_type',
                'optimization',
                 'threshold',
                 'auc_score_ci',
                 'sensitivity_ci',
                 'specificity_ci']

    df_outer_metrics_records_ci_print = df_outer_metrics_records_ci.copy()
    df_outer_metrics_records_ci_print.sort_values(by=['auc_score'], ascending=False, inplace=True)
    print(tabulate(tabular_data=df_outer_metrics_records_ci[tbs_print],
                   headers=tbs_print,
                   tablefmt='psql') )
    print(f"\nSaved outer metrics and predictions in {results_dir.resolve()}")
    return df_outer_metrics_records_ci, df_outer_predictions_records, df_inner_val_records
