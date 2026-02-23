Here’s a complete **README.md** draft for your project. It combines the scientific background you provided with the structure and scripts from the uploaded files.

---

# A Two-Stage Questionnaire and Actigraphy Screening for Isolated REM Sleep Behavior Disorder in a Multicenter Cohort 

## Overview

This repository contains the code and analysis pipeline for the study:

**"A Two-Stage Questionnaire and Actigraphy Protocol for Remote Detection of Isolated REM Sleep Behavior Disorder in a Multicenter Cohort."**

The project integrates:

1. **A 4-item screening questionnaire** (RBD symptoms, hyposmia, constipation, orthostatic symptoms).
2. **Actigraphy-derived sleep features** extracted from wrist-worn AX6/AX3 accelerometers.
3. **A machine learning pipeline** using nested cross-validation and hyperparameter optimization (Optuna) to train classifiers.
4. **A two-stage screening protocol**: Stage 1 questionnaire → Stage 2 actigraphy confirmation.

<div style="text-align: center;">
  <img src="static/two_stage_asthetic-Two-Stage iRBD Detection.png" width="60%">
</div>
---

## Study Cohorts

* **Mount Sinai Sleep and Healthy Aging Study (SHAS)**: Questionnaire + wearable data (n=62).
* **VascBrain cohort**: Wearable data (n=25).
* **Stanford Sleep Center**: Questionnaire + wearable data (n=84; 63 clinic, 21 community).
* **Stanford ADRC**: Wearable controls (n=78).
* **Mount Sinai sleep clinics**: Questionnaire only (n=147).

Inclusion: Age 40–80, no overt neurodegenerative disease. All iRBD cases were PSG-confirmed; controls with RBD-like symptoms but negative PSG were considered “mimics.”

---

## Methods

### Questionnaire

* **4 items**: RBD, hyposmia, constipation, orthostasis.
* Responses: *No = 0, Don’t know = 0.5, Yes = 1*.
* Models tested: Random Forest, LightGBM, XGBoost, Elastic Net.

### Actigraphy

* Devices: **Axivity AX6 (50 Hz, ±8g)** and **AX3 (100 Hz)**.
* Nights recorded: **6,620 nights** across 78 iRBD and 158 controls.
* Features extracted (n=113):

### Machine Learning Pipeline

* **Nested cross-validation**:
  * Outer loop: 10 folds (performance estimation).
  * Inner loop: 5 folds (Optuna hyperparameter tuning).
* **Models**: XGBoost for actigraphy, multiple classifiers for questionnaire.
* **Thresholds**:
  * τ = 0.5 (default).
  * τ* = Youden’s J (balanced Se/Sp).
  * Custom τ for maximizing Se (questionnaire) or Sp (actigraphy).
* Implemented in **Python 3.10** with scikit-learn, XGBoost, Optuna, statsmodels.

### Two-Stage Screening

* Stage 1: Questionnaire (maximize sensitivity).
* Stage 2: Actigraphy (maximize specificity).
* Final rule: classified as iRBD if **both** stages positive.

---

## Running the Pipelines

### Questionnaire Model

```bash
python ml_questionnaire.py
```

### Actigraphy Model


```bash
python ml_actigraphy.py
```

### Two-Stage Protocol

```bash
python two_stage_predictions.py
```

---

* **Results**:

The performance of the questionnaire alone was assessed in 95 iRBD and 194 controls. Dream enactment had an AUC of 0.85, sensitivity 77.9%, and specificity 92.3%. Hyposmia had an AUC of 0.69, sensitivity 56.8%, and specificity 80.9%. Constipation had an AUC of 0.62, sensitivity 55.8%, and specificity 67.0%. Orthostatic hypotension had an AUC of 0.52, sensitivity 31.6%, and specificity 75.3% 
<div style="text-align: center;">
  <img src="static/Figure 2.png" width="60%">
</div>

Based on within-subject probabilities averaged across nights, XGBoost model achieved an AUC of 0.88 (95% CI: 0.84–0.92) and an area under the precision–recall curve of 0.85 (95% CI: 0.78–0.91). At τ= 0.5, the model achieved sensitivity 80.8% and specificity 84.8%; at  τ_Sp = 0.62 sensitivity 61.5% and specificity 96.2%; and at τ^* = 0.49, sensitivity 82.1% and specificity 83.5%. 
<div style="text-align: center;">
  <img src="static/Figure 3.png" width="60%">
</div>

Applied consecutively, the two-stage screening starting with the dream enactment question followed by actigraphy achieved a final sensitivity of 67.6% and specificity of 100%, while using instead the 4-item questionnaire (Figure 4A), final sensitivity and specificity were 68.9% and 100% with default thresholds, and 73.3% and 100%, with optimized thresholds, respectively 
<div style="text-align: center;">
  <img src="static/Figure 4a.png" width="60%">
  <img src="static/Figure 4b.png" width="60%">
</div>

In the lasso regression model, the largest positive coefficients were observed for “dream enactment yes” (β = 0.84) and “smell yes” (β = 0.77). Among interaction terms, the highest coefficients were for interaction between “smell do not know” and “orthostasis yes” (β = 0.65), followed by the interaction between “orthostasis do not know” and “smell yes” (β = 0.58), and the interaction between “orthostasis do not know” and “constipation yes” (β = 0.52). A negative coefficient was identified for the interaction between “dream enactment yes” and “orthostasis do not know” (β = –0.41) 
<div style="text-align: center;">
  <img src="static/Figure S1.png" width="60%">
</div>

---

## Key Features

* Reproducible **nested CV** with group-stratified folds (subject-level).
* Configurable thresholds for clinical screening needs.
* Two-stage diagnostic mimicry (questionnaire + wearable).
* Confidence intervals for robust statistical reporting.
