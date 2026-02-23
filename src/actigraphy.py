# ─── CELL 1: NESTED CROSS-VALIDATION EXPERIMENT ──────────────────────────────────
import os, warnings, logging, json, joblib, optuna, xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna
from optuna.integration import XGBoostPruningCallback
from typing import Optional

# --- Global Config & Setup ---
RANDOM_SEED = 42
N_JOBS = -1  # Use all available cores
DATA_PATH = Path("../data/processed/nightly_features_qc3.csv").resolve()
RESULTS_DIR = Path("../results/final_nested_cv_run").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"💾 All artifacts will be saved to: {RESULTS_DIR}")

# ─── Data Loading & Feature Selection ---
print("🔄 Loading data and selecting approved features...")
df = pd.read_csv(DATA_PATH)
approved_features = [
    'TST', 'WASO', 'SE', 'On', 'Off', 'AI10', 'AI10_w', 'AI10_REM', 'AI10_REM_w',
    'AI10_NREM', 'AI10_NREM_w', 'AI30', 'AI30_w', 'AI30_REM', 'AI30_REM_w',
    'AI30_NREM', 'AI30_NREM_w', 'AI60', 'AI60_w', 'AI60_REM', 'AI60_REM_w',
    'AI60_NREM', 'AI60_NREM_w', 'TA0.5', 'TA0.5_w', 'TA0.5_REM',
    'TA0.5_REM_w', 'TA0.5_NREM', 'TA0.5_NREM_w', 'TA1', 'TA1_w', 'TA1_REM',
    'TA1_REM_w', 'TA1_NREM', 'TA1_NREM_w', 'TA1.5', 'TA1.5_w', 'TA1.5_REM',
    'TA1.5_REM_w', 'TA1.5_NREM', 'TA1.5_NREM_w', 'SIB0', 'SIB0_w',
    'SIB0_REM', 'SIB0_REM_w', 'SIB0_NREM', 'SIB0_NREM_w', 'SIB1', 'SIB1_w',
    'SIB1_REM', 'SIB1_REM_w', 'SIB1_NREM', 'SIB1_NREM_w', 'SIB5', 'SIB5_w',
    'SIB5_REM', 'SIB5_REM_w', 'SIB5_NREM', 'SIB5_NREM_w', 'LIB60', 'LIB60_w',
    'LIB60_REM', 'LIB60_REM_w', 'LIB60_NREM', 'LIB60_NREM_w', 'LIB120',
    'LIB120_w', 'LIB120_REM', 'LIB120_REM_w', 'LIB120_NREM',
    'LIB120_NREM_w', 'LIB300', 'LIB300_w', 'LIB300_REM', 'LIB300_REM_w',
    'LIB300_NREM', 'LIB300_NREM_w', 'MMAS', 'MMAS_w', 'MMAS_REM',
    'MMAS_REM_w', 'MMAS_NREM', 'MMAS_NREM_w', 'T_avg', 'T_avg_w', 'T_avg_REM',
    'T_avg_REM_w', 'T_avg_NREM', 'T_avg_NREM_w', 'T_std', 'T_std_w',
    'T_std_REM', 'T_std_REM_w', 'T_std_NREM', 'T_std_NREM_w', 'HP_A_ac',
    'HP_A_ac_w', 'HP_A_ac_REM', 'HP_A_ac_REM_w', 'HP_A_ac_NREM',
    'HP_A_ac_NREM_w', 'HP_M_ac', 'HP_M_ac_w', 'HP_M_ac_REM',
    'HP_M_ac_REM_w', 'HP_M_ac_NREM', 'HP_M_ac_NREM_w', 'HP_C_ac',
    'HP_C_ac_w', 'HP_C_ac_REM', 'HP_C_ac_REM_w', 'HP_C_ac_NREM', 'HP_C_ac_NREM_w'
]
X = df[approved_features]
y = df["label"]
groups = df["subject_id"]
print(f"✅ Shape: {X.shape} (rows × features)")

# ─── Preprocessing Pipeline ---





def run_actigraphy_model(X:pd.DataFrame,
                         y:pd.Series,
                         groups:pd.Series,
                         folds_outer:int=10,
                         optuna_direction:str='maximize',
                         optuna_n_trials:int=30,
                         random_seed:int=42,
                         n_jobs:Optional[int]=-1):
    preprocessor = ColumnTransformer(
        [("num",
          SimpleImputer(strategy="median"),
          approved_features)],
        remainder="drop"
    )

    # ─── Nested Cross-Validation ---
    outer_cv = GroupKFold(n_splits=folds_outer)

    all_fold_predictions = []
    for fold_num, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y, groups)):
        print(f"\n--- Outer Fold {fold_num + 1}/{outer_cv.get_n_splits()} ---")
        X_outer_train, y_outer_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        X_outer_test, y_outer_test = X.iloc[outer_test_idx], y.iloc[outer_test_idx]
        groups_outer_train = groups.iloc[outer_train_idx]

        def objective(trial):
            params = {  # Hyperparameters to tune
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                'max_depth': trial.suggest_int("max_depth", 3, 12),
                'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
                'subsample': trial.suggest_float("subsample", 0.5, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            fixed_params = {  # Fixed parameters for the classifier
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
                'n_estimators': 5000, 'n_jobs': N_JOBS, 'random_state': RANDOM_SEED,
                'early_stopping_rounds': 100
            }

            inner_cv = GroupKFold(n_splits=5)
            aucs = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train, groups_outer_train):
                pipeline = Pipeline(
                    [('prep', clone(preprocessor)), ('clf', xgb.XGBClassifier(**params, **fixed_params))])
                X_inner_train, y_inner_train = X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx]
                X_inner_val, y_inner_val = X_outer_train.iloc[inner_val_idx], y_outer_train.iloc[inner_val_idx]

                # Fit a temporary preprocessor and transform the validation set
                temp_preprocessor = clone(preprocessor).fit(X_inner_train)
                X_inner_val_transformed = temp_preprocessor.transform(X_inner_val)

                pipeline.fit(X_inner_train, y_inner_train,
                             clf__eval_set=[(X_inner_val_transformed, y_inner_val)],
                             clf__verbose=False)

                preds = pipeline.predict_proba(X_inner_val)[:, 1]
                aucs.append(roc_auc_score(y_inner_val, preds))
            return np.mean(aucs)

        print("  Running inner Optuna search...")
        study = optuna.create_study(direction=optuna_direction)
        study.optimize(objective, n_trials=optuna_n_trials)  # 50 trials is a solid search per fold

        print("  Training final model for this fold...")
        best_params = study.best_params | {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'n_estimators': 5000,
            'n_jobs': N_JOBS,
            'random_state': random_seed,
            'early_stopping_rounds': 100
        }

        # ... inside outer loop, after study.optimize()
        final_model_fold = Pipeline([('prep', clone(preprocessor)),
                                     ('clf', xgb.XGBClassifier(**best_params))])

        # --- Create a holdout validation set from the outer TRAIN data for early stopping ---
        X_train_fit, X_val_stop, y_train_fit, y_val_stop = train_test_split(
            X_outer_train, y_outer_train,
            test_size=0.15,  # Use 15% of the outer train data for stopping
            stratify=y_outer_train,  # Ensure label balance
            random_state=RANDOM_SEED
        )

        # Transform the validation set using a temporary, fitted preprocessor
        temp_preprocessor_stop = clone(preprocessor).fit(X_train_fit)
        X_val_stop_transformed = temp_preprocessor_stop.transform(X_val_stop)

        # Fit the final model on the larger training portion, using the holdout for stopping
        final_model_fold.fit(X_train_fit, y_train_fit,
                             clf__eval_set=[(X_val_stop_transformed, y_val_stop)],
                             clf__verbose=False)

        # Now, predict on the truly unseen outer test set
        preds = final_model_fold.predict_proba(X_outer_test)[:, 1]

        # ... rest of your code ...

        # Using .values ensures alignment between subject IDs, labels, and predictions
        fold_df = pd.DataFrame({
            'subject_id': groups.iloc[outer_test_idx].values,
            'label': y_outer_test.values,
            'score': preds
        })
        all_fold_predictions.append(fold_df)

        # Save study results for this fold for reproducibility
        study.trials_dataframe().to_csv(RESULTS_DIR / f'fold_{fold_num + 1}_study_results.csv', index=False)

    # Save the aggregated predictions from all folds
    final_results_df = pd.concat(all_fold_predictions, ignore_index=True)
    final_results_df.to_csv(RESULTS_DIR / 'nested_cv_predictions7.csv', index=False)

print("\n🎉 Nested CV complete. All artifacts saved.")

# %%
# ─── CELL 2: ANALYSIS & VISUALIZATION ───────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# --- Load the results from the experiment ---
RESULTS_DIR = Path("../results/final_nested_cv_run").resolve()
results_df = pd.read_csv(RESULTS_DIR / 'nested_cv_predictions.csv')
print("✅ Loaded prediction results.")

# --- Find Optimal Threshold & Calculate Metrics ---
print("🔄 Calculating metrics...")
subj_df = results_df.groupby('subject_id').agg(score=('score', 'mean'), label=('label', 'first')).reset_index()
fpr_subj, tpr_subj, thresholds_subj = roc_curve(subj_df['label'], subj_df['score'])
optimal_idx = np.argmax(tpr_subj - fpr_subj)
optimal_threshold = thresholds_subj[optimal_idx]
print(f"  Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

results_df['prediction'] = (results_df['score'] > optimal_threshold).astype(int)
subj_df['prediction'] = (subj_df['score'] > optimal_threshold).astype(int)

# --- Bootstrap Confidence Intervals for AUC ---
print("🔄 Bootstrapping AUC confidence intervals...")
n_bootstraps = 1000
bootstrapped_aucs = []
rng = np.random.RandomState(RANDOM_SEED)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(subj_df), len(subj_df))
    if len(np.unique(subj_df['label'].iloc[indices])) < 2: continue
    boot_auc = roc_auc_score(subj_df['label'].iloc[indices], subj_df['score'].iloc[indices])
    bootstrapped_aucs.append(boot_auc)
alpha = 0.95
lower_bound = np.percentile(bootstrapped_aucs, (1.0 - alpha) / 2.0 * 100)
upper_bound = np.percentile(bootstrapped_aucs, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"  95% CI for Subject-Level AUC: [{lower_bound:.4f} - {upper_bound:.4f}]")

# --- Final Report ---
auc_subj = roc_auc_score(subj_df['label'], subj_df['score'])
print("\n" + "="*60)
print("📊 FINAL MANUSCRIPT REPORT 📊")
print("="*60)
print(f"\nSubject-Level AUROC: {auc_subj:.4f} (95% CI: [{lower_bound:.4f} - {upper_bound:.4f}])")

# --- Generate Plots ---
print("\n🎨 Generating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. ROC Curve
fpr_night, tpr_night, _ = roc_curve(results_df['label'], results_df['score'])
auc_night = roc_auc_score(results_df['label'], results_df['score'])
plt.figure(figsize=(8, 7))
plt.plot(fpr_night, tpr_night, label=f'Night-Level (AUC = {auc_night:.3f})', lw=2, alpha=0.7)
plt.plot(fpr_subj, tpr_subj, label=f'Subject-Level (AUC = {auc_subj:.3f})', lw=3, linestyle='--')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', label='Chance')
plt.scatter(fpr_subj[optimal_idx], tpr_subj[optimal_idx], marker='o', color='red', s=100, zorder=5, label=f'Optimal Point ({optimal_threshold:.2f})')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
plt.legend(loc="lower right")
plt.show()

# 2. Calibration Curve
prob_true, prob_pred = calibration_curve(subj_df['label'], subj_df['score'], n_bins=10, strategy='uniform')
plt.figure(figsize=(7, 7))
plt.plot(prob_pred, prob_true, "s-", label="XGBoost")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
plt.title('Calibration Curve (Subject-Level)', fontsize=16)
plt.ylabel("Fraction of Positives")
plt.xlabel("Mean Predicted Probability")
plt.legend()
plt.show()


# %%

# ─── CELL ➜ LOAD PREDICTIONS & PREP ────────────────────────────────────────────
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_fscore_support,
                             accuracy_score, roc_auc_score)
import shap, joblib, json, itertools, warnings

# --- Load night-level predictions ---
RESULTS_DIR = Path("../results/final_nested_cv_run").resolve()
pred_df     = pd.read_csv(RESULTS_DIR / "nested_cv_predictions.csv")

# --- Aggregate to Subject-Level (This is the key DataFrame to use) ---
subj_df = (pred_df.groupby("subject_id")
                   .agg({"score":"mean", "label":"first"})
                   .reset_index())

# Helper: bootstrap AUC 95% CI
def bootstrap_auc(y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        # Ensure both classes are present in the bootstrap sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5])



# %%

# ─── CELL ➜ CORE PERFORMANCE METRICS (SUBJECT) ──────────────────────────────────
y_true  = subj_df["label"].values   # <-- CHANGE: Use subject-level data
y_score = subj_df["score"].values   # <-- CHANGE: Use subject-level data

# Choose optimal threshold from subject-level data
fpr, tpr, thr = roc_curve(y_true, y_score)
j_idx         = np.argmax(tpr - fpr)
opt_thresh    = thr[j_idx]

# Generate predictions based on the threshold
y_pred = (y_score >= opt_thresh).astype(int)

# Calculate metrics
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
spec             = confusion_matrix(y_true, y_pred)[0,0] / (y_true == 0).sum()
acc              = accuracy_score(y_true, y_pred)
auc_val          = roc_auc_score(y_true, y_score)
ci_low, ci_hi    = bootstrap_auc(y_true, y_score)

metrics = pd.Series({
    "Sensitivity (Recall)":   rec,
    "Specificity":            spec,
    "Precision (PPV)":        prec,
    "Accuracy":               acc,
    "F1-Score":               f1,
    "AUC":                    auc_val,
    "AUC 95% CI Low":         ci_low,
    "AUC 95% CI High":        ci_hi,
    "Optimal Threshold":      opt_thresh,
}).round(3)

print("--- Subject-Level Performance Metrics ---")
print(metrics)


#%% ─── CELL ➜ CONFUSION MATRIX (SEABORN) ───────────────────────────────────────
cm = confusion_matrix(y_true, y_pred) # This now uses subject-level variables
labels = np.array([["TN","FP"],["FN","TP"]])

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=labels+"\n"+cm.astype(str), fmt="", cmap="Blues",
            cbar=False, square=True, annot_kws={"size":12})
plt.title("Subject-Level Confusion Matrix") # <-- CHANGE: Updated title
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()


#%% ─── CELL ➜ ROC CURVE (1-Specificity vs Sensitivity) ─────────────────────────
plt.figure(figsize=(4,4))
# This now uses subject-level fpr, tpr, and auc_val from the metrics cell
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}") # <-- CHANGE: Plot fpr, not 1-fpr
plt.plot([0,1],[0,1],"--", lw=1, color="grey")
plt.xlabel("1 – Specificity (False Positive Rate)"); plt.ylabel("Sensitivity (True Positive Rate)"); plt.legend()
plt.title("Subject-Level ROC Curve") # <-- CHANGE: Updated title
plt.tight_layout(); plt.show()


# ─── CELL ➜ DISTRIBUTION OF PREDICTION SCORES ────────────────────────────────
# <-- CHANGE: Use subj_df for plotting
sns.kdeplot(data=subj_df, x="score", hue="label", common_norm=False, fill=True, alpha=.3)
plt.axvline(opt_thresh, ls="--", c="k"); plt.text(opt_thresh+.01, 1.0, "Threshold") # Adjusted text y-position
plt.title("Subject-Level Prediction-Score Distributions"); plt.xlabel("Mean Score"); plt.show() # <-- CHANGE: Updated labels



#%% ─── CELL ➜ CORE PERFORMANCE METRICS (SUBJECT) ──────────────────────────────────
y_true  = subj_df["label"].values
y_score = subj_df["score"].values

# Choose optimal threshold from subject-level data
fpr, tpr, thr = roc_curve(y_true, y_score)
j_idx         = np.argmax(tpr - fpr)
opt_thresh    = thr[j_idx]

# Generate predictions based on the threshold
y_pred = (y_score >= opt_thresh).astype(int)

# ❗ FIX: Add 'prediction' and 'correct' columns to the DataFrame for later use
subj_df['prediction'] = y_pred
subj_df['correct'] = (subj_df['prediction'] == subj_df['label']).astype(int)

# Calculate metrics
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
spec             = confusion_matrix(y_true, y_pred)[0,0] / (y_true == 0).sum()
acc              = accuracy_score(y_true, y_pred)
auc_val          = roc_auc_score(y_true, y_score)
ci_low, ci_hi    = bootstrap_auc(y_true, y_score)

metrics = pd.Series({
    "Sensitivity (Recall)":   rec,
    "Specificity":            spec,
    "Precision (PPV)":        prec,
    "Accuracy":               acc,
    "F1-Score":               f1,
    "AUC":                    auc_val,
    "AUC 95% CI Low":         ci_low,
    "AUC 95% CI High":        ci_hi,
    "Optimal Threshold":      opt_thresh,
}).round(3)

print("--- Subject-Level Performance Metrics ---")
print(metrics)


# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve

# 1. Load the night-level data
RESULTS_DIR = Path("../results/final_nested_cv_run").resolve()
pred_df = pd.read_csv(RESULTS_DIR / "nested_cv_predictions.csv")

# 2. Aggregate to get mean score and label per subject
subj_df = (pred_df.groupby("subject_id")
                   .agg(score=('score', 'mean'), label=('label', 'first'))
                   .reset_index())

# 3. Calculate and merge the number of nights for each subject
night_counts = pred_df.groupby("subject_id").size().rename("nights")
subj_df = pd.merge(subj_df, night_counts, on='subject_id')

# 4. Determine the optimal threshold based on subject-level scores
fpr, tpr, thresholds = roc_curve(subj_df['label'], subj_df['score'])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold determined from subject-level data: {optimal_threshold:.4f}")

# 5. Add the 'prediction' column based on the threshold
subj_df['prediction'] = (subj_df['score'] >= optimal_threshold).astype(int)

# 6. Add the 'correct' column by comparing prediction to the true label
subj_df['correct'] = (subj_df['prediction'] == subj_df['label']).astype(int)

# --- Display the final, complete DataFrame ---
print("\nFinal Subject-Level DataFrame with all requested columns:")
print(subj_df.head())

# --- Save the final DataFrame to a new, clearly named CSV ---
final_csv_path = RESULTS_DIR / 'subject_level_complete_results.csv'
subj_df.to_csv(final_csv_path, index=False)
print(f"\nSuccessfully saved the complete subject-level data to:\n{final_csv_path}")

# %%  Assumes 'subj_df' from your previous cell is available
plt.figure(figsize=(6, 5))
sns.boxplot(data=subj_df, x='label', y='nights')
plt.title('Distribution of Nights per Subject by Group')
plt.xlabel('Group (0=Control, 1=RBD)')
plt.ylabel('Number of Nights')
plt.show()