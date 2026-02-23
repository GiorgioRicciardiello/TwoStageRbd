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
from config.config import config
from sklearn.linear_model import LogisticRegression
from library.epidemiology.effect_measures_plot import EffectMeasurePlotGrouping, EffectMeasurePlot
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


def analyze_questionnaire(df: pd.DataFrame,
                          question_cols: list[str],
                          label_col: str = "label",) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze questionnaire data with:
    1. Logistic regression (dummy-coded Yes/DK vs No).
    2. Logistic regression (DK-only).
    3. Logistic regression with interactions (LASSO).
    """
    def fit_logit_model(df: pd.DataFrame, formula: str, source: str) -> pd.DataFrame:
        """Fit a logistic regression (statsmodels) and return coefficient + model stats."""
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        approx = 3
        ci = model.conf_int()
        ci_lower = np.exp(ci[0].values)
        ci_upper = np.exp(ci[1].values)

        res = pd.DataFrame({
            "variable": model.params.index,
            "coef": np.round(model.params.values, approx),
            "std_err": np.round(model.bse.values, approx),
            "odds_ratio": np.round(np.exp(model.params.values), approx),
            "p_value": np.round(model.pvalues.values, approx),
            "ci_lower": np.round(ci_lower, approx),
            "ci_upper": np.round(ci_upper, approx),
            "ci": [f"[{l:.3f}–{u:.3f}]" for l, u in zip(ci_lower, ci_upper)],
            "log_likelihood": round(model.llf, approx),
            "aic": round(model.aic, approx),
            "bic": round(model.bic, approx),
            "nobs": int(model.nobs),
            "df_model": int(model.df_model),
            "df_resid": int(model.df_resid),
            "pseudo_r2_mcfadden": round(model.prsquared, approx),
            "source": source
        }).reset_index(drop=True)

        return res

    def get_glmnet_path(X: pd.DataFrame, y: pd.Series, dummy_cols: list[str],
                        Cs=np.logspace(-3, 2, 40)) -> pd.DataFrame:
        """
        Compute GLMNET-style coefficient paths for the exact same pipeline
        (degree=7 PolynomialFeatures, interaction_only=True, with StandardScaler).

        Args:
            X: DataFrame with dummy-coded predictors.
            y: outcome (binary).
            dummy_cols: list of predictors to expand with PolynomialFeatures.
            Cs: grid of regularization strengths.

        Returns:
            DataFrame with columns: [C, logC, variable, coef]
        """

        # Same feature expansion as model
        poly = PolynomialFeatures(degree=7, interaction_only=True, include_bias=False)
        scaler = StandardScaler(with_mean=False)

        X_poly = poly.fit_transform(X[dummy_cols])
        X_poly = scaler.fit_transform(X_poly)
        feature_names = poly.get_feature_names_out(input_features=dummy_cols)

        # Store coefficient paths
        coef_path = []
        for C in Cs:
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=C, max_iter=5000
            ).fit(X_poly, y)

            for f, c in zip(feature_names, model.coef_[0]):
                coef_path.append({
                    "C": C,
                    "logC": -np.log10(C),
                    "variable": f,
                    "coef": c
                })

        df_path = pd.DataFrame(coef_path)
        return df_path


    df_copy = df.copy()

    # Dummy coding
    for q in question_cols:
        df_copy[f"{q}_Yes"] = (df_copy[q] == 1).astype(int)
        df_copy[f"{q}_DK"] = (df_copy[q] == 0.5).astype(int)

    dummy_cols = [c for c in df_copy.columns if any(q in c for q in question_cols) and ("_Yes" in c or "_DK" in c)]
    dk_cols = [c for c in dummy_cols if c.endswith("_DK")]

    # --- (1) Full dummy-coded logistic ---
    formula_full = f"{label_col} ~ " + " + ".join(dummy_cols)
    df_logit_full = fit_logit_model(df_copy, formula_full, "Dummy logistic")
    df_logit_full['model'] = 'Full Model'

    # --- (2) DK-only logistic ---
    if not dk_cols:
        raise ValueError("No DK dummy columns found!")
    formula_dk = f"{label_col} ~ " + " + ".join(dk_cols)
    df_logit_dk = fit_logit_model(df_copy, formula_dk, "DK-only logistic")
    df_logit_dk['model'] = 'DK Only'


    # --- (3) Full + DK logistic (explicit DK terms alongside full dummy set) ---
    formula_full_dk = f"{label_col} ~ " + " + ".join(dummy_cols + dk_cols)
    df_logit_full_dk = fit_logit_model(df_copy, formula_full_dk, "Full + DK logistic")
    df_logit_full_dk['model'] = 'Full Model and DK Only'

    # merge the logistic model
    df_logit = pd.concat([df_logit_full, df_logit_dk, df_logit_full_dk])

    # (4) LASSO with interactions
    X = df_copy[dummy_cols]
    y = df_copy[label_col]

    poly = PolynomialFeatures(degree=7, interaction_only=True, include_bias=False)
    pipeline = Pipeline([
        ("poly", poly),
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",  # LASSO
            solver="liblinear",
            max_iter=1000,
            scoring="roc_auc"
        ))
    ])
    lasso_model = pipeline.fit(X, y)
    # --- Predictions for evaluation ---
    y_pred_proba = lasso_model.predict_proba(X)[:, 1]
    y_pred = lasso_model.predict(X)

    # Metrics
    auc = roc_auc_score(y, y_pred_proba)  # ROC AUC
    pr_auc = average_precision_score(y, y_pred_proba)  # Precision-Recall AUC
    ll_model = -log_loss(y, y_pred_proba, normalize=False)  # total log-likelihood
    brier = brier_score_loss(y, y_pred_proba)  # calibration

    # Null model for pseudo-R²
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    y_null_proba = dummy.predict_proba(X)[:, 1]
    ll_null = -log_loss(y, y_null_proba, normalize=False)
    pseudo_r2 = 1 - (ll_model / ll_null)

    # --- Coefficient extraction ---
    feature_names = lasso_model.named_steps["poly"].get_feature_names_out(input_features=dummy_cols)
    coefs = lasso_model.named_steps["logreg"].coef_[0]

    nonzero = [(f, c, np.exp(c)) for f, c in zip(feature_names, coefs) if abs(c) > 1e-6]
    df_lasso = pd.DataFrame(nonzero, columns=["variable", "coef", "odds_ratio"])

    # Add model-level metrics to every row
    df_lasso["roc_auc"] = auc
    df_lasso["pr_auc"] = pr_auc
    df_lasso["log_loss"] = -ll_model / len(y)  # average log loss
    df_lasso["brier_score"] = brier
    df_lasso["pseudo_r2_mcfadden"] = pseudo_r2
    df_lasso["model"] = "LASSO with interactions"

    # --- (5) GLMNET-style coefficient path ---
    df_path = get_glmnet_path(
        X=df_copy,
        y=df_copy[label_col],
        dummy_cols=dummy_cols,
        Cs=np.logspace(-3, 2, 50)
    )


    return df_logit, df_lasso


def plot_glmnet_paths(df_path: pd.DataFrame, only_dk=True, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    df_plot = df_path.copy()
    if only_dk:
        df_plot = df_plot[df_plot["variable"].str.contains("_DK")]

    for var, group in df_plot.groupby("variable"):
        plt.plot(group["logC"], group["coef"], label=var, alpha=0.8)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("-log10(C)")
    plt.ylabel("Coefficient")
    plt.title("GLMNET-style Coefficient Paths (LASSO, degree=7 interactions)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_lasso_coefficients_vertical(df_lasso: pd.DataFrame,
                                     auc: float,
                                     pr_auc: float,
                                     pseudo_r2: float,
                                     brier: float,
                                     log_loss_val: float,
                                     top_n: int = None,
                                     figsize: tuple = (14, 8),
                                     save_path: str = None):
    """
    Vertical barplot of LASSO logistic regression coefficients (log-odds).
    Negative coefs shown as darker bars.
    """

    # --- Helper to format variable labels ---
    def _format_label(var: str) -> str:
        mapping = {
            "q1_rbd": "RBD",
            "q2_smell": "Smell",
            "q4_constipation": "Constipation",
            "q5_orthostasis": "Orthostasis"
        }
        # mapping = {
        #     "q1_rbd": "R",
        #     "q2_smell": "S",
        #     "q4_constipation": "C",
        #     "q5_orthostasis": "O"
        # }

        parts = var.split(" ")
        labels = []
        for p in parts:
            for key, label in mapping.items():
                if key in p:
                    if p.endswith("_Yes"):
                        labels.append(f"{label} (Y)")
                    elif p.endswith("_DK"):
                        labels.append(f"{label} (DK)")
        return "\n".join(labels)

    # --- Sort and select top N ---
    df_plot = df_lasso.copy()
    df_plot = df_plot.loc[(df_plot['coef'] > 0.02) | (df_plot['coef'] < -0.02)]
    df_plot = df_plot.sort_values(by="coef", key=lambda x: abs(x), ascending=False)
    if top_n:
        df_plot = df_plot.head(top_n)
    df_plot["label"] = df_plot["variable"].apply(_format_label)

    # --- Define new base palette ---
    blue_ = sns.color_palette("Set2")[0]  # Yes
    green_ = sns.color_palette("Set2")[2]  # DK
    gray_ = "lightgray"  # Interaction no DK
    pink_ = "#f4a6b8"  # Interaction with DK (soft pink)

    def _darken(color, factor=0.6):
        """Darken any RGB or hex color for negative coefficients."""
        import matplotlib.colors as mc, colorsys
        c = mc.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, max(0, l * factor), s)

    bar_colors = []
    for var, coef in zip(df_plot["variable"], df_plot["coef"]):
        if " " not in var:  # single variable
            if var.endswith("_Yes"):
                base = blue_
            elif var.endswith("_DK"):
                base = green_
            else:
                base = gray_  # fallback
        else:  # interaction
            if "_DK" in var:
                base = pink_  # highlight DK interactions
            else:
                base = gray_  # neutral gray for Yes/Yes interactions
        bar_colors.append(_darken(base) if coef < 0 else base)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot["coef"], color=bar_colors, edgecolor="black")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Coefficient")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_plot["label"], rotation=90, ha="center")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    # Annotate raw coef values
    for bar, coef in zip(bars, df_plot["coef"]):
        ypos = bar.get_height()
        if coef >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.02,
                    f"{coef:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, ypos - 0.02,
                    f"{coef:.2f}", ha="center", va="top", fontsize=9)

    # Legend with metrics
    metrics_text = (f"AUC={auc:.3f}\nPR-AUC={pr_auc:.3f}\n"
                    f"R²={pseudo_r2:.3f}\nBrier={brier:.3f}\nLogLoss={log_loss_val:.3f}")

    legend_patches = [
        mpatches.Patch(color=blue_, label="Single Yes (Y)"),
        mpatches.Patch(color=green_, label="Single Do Not Know (DK)"),
        mpatches.Patch(color=gray_, label="Interaction (no DK)"),
        mpatches.Patch(color=pink_, label="Interaction (with DK)"),
        mpatches.Patch(color="goldenrod", label=metrics_text)
    ]

    ax.legend(handles=legend_patches, loc="upper right", frameon=False)
    ax.grid(alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return df_plot

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

    # %%  =================== Logistic Regression Modeling
    # --- valuate whether the “Do not know” (0.5) option is informative beyond “No” (0) and “Yes” (1) ---
    df_logit, df_lasso = analyze_questionnaire(df=df_data,
                          question_cols=question_cols,
                          label_col=label_col,)

    df_plot = plot_lasso_coefficients_vertical(
        df_lasso=df_lasso,
        auc=df_lasso["roc_auc"].iloc[0],
        pr_auc=df_lasso["pr_auc"].iloc[0],
        pseudo_r2=df_lasso["pseudo_r2_mcfadden"].iloc[0],
        brier=df_lasso["brier_score"].iloc[0],
        log_loss_val=df_lasso["log_loss"].iloc[0],
        top_n=60,
        figsize=(16, 8),
        save_path=None
    )


    plotter = EffectMeasurePlotGrouping(df_logit,
                                        label_col="variable",
                                        effect_col="odds_ratio",
                                        lcl_col="ci_lower",
                                        ucl_col="ci_upper",
                                        pval_col="p_value",
                                        group_col="model")

    plotter.plot(figsize=(12, 8))
    plt.show()







