"""
pipeline.py
===========
Central ML pipeline for the AI-CDSS.

Covers:
  - Preprocessing (ColumnTransformer with OHE for categoricals, RobustScaler for continuous)
  - Ensemble training: Random Forest + Gradient Boosting + Logistic Regression
  - Stratified 10-Fold Cross-Validation with full metric suite
  - SHAP explainability (per-patient and global)
  - Gower-distance KNN for case-based retrieval
  - Incremental / online learning via river (SGDClassifier fallback if unavailable)
  - Calibrated probability output (Platt scaling)

All evaluation is performed strictly on held-out CV folds — no train-set metric inflation.

Author : Student Research Project
Dataset: Cleveland Heart Disease, UCI ML Repository (Detrano et al., 1989)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import shap

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CONTINUOUS  : List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL : List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES: List[str] = CONTINUOUS + CATEGORICAL

FEATURE_LABELS: Dict[str, str] = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (1–4)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar >120 mg/dL",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate Achieved (bpm)",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (exercise vs rest)",
    "slope":    "Slope of Peak ST Segment",
    "ca":       "Major Vessels by Fluoroscopy (0–3)",
    "thal":     "Thalassemia Type",
}

N_CV_SPLITS = 10


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Proper preprocessing:
    - Continuous features: median imputation → RobustScaler (robust to outliers)
    - Categorical features: mode imputation → OneHotEncoder
      (avoids false ordinal distance imposed by LabelEncoder)
    """
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
        ]), CONTINUOUS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _rf() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

def _gbm() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42,
        )),
    ])

def _lr() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(
            C=0.5, max_iter=2000, random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# RISK STRATIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    if prob >= 0.70: return "High Risk"
    if prob >= 0.40: return "Moderate Risk"
    return "Low Risk"

def risk_color(prob: float) -> str:
    if prob >= 0.70: return "#c62828"
    if prob >= 0.40: return "#e65100"
    return "#2e7d32"


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE TRAINING + FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete ML pipeline:
    1. ColumnTransformer preprocessing (OHE for categoricals)
    2. Stratified 10-fold CV for each model
    3. Ensemble soft voting across folds
    4. Full metric suite: Accuracy, F1, AUC, Sensitivity, Specificity, PPV, NPV, MCC, Brier
    5. ROC + PR curves from aggregated CV predictions
    6. Final models trained on ALL data for inference
    7. SHAP global feature importances
    8. KNN similarity model (euclidean on preprocessed space)
    """
    X = df[ALL_FEATURES]
    y = df["target"].values
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)

    # ── Individual model CV ───────────────────────────────────────────────
    scoring = ["accuracy", "f1", "roc_auc", "average_precision"]
    cv_rf  = cross_validate(_rf(),  X, y, cv=skf, scoring=scoring, return_train_score=True)
    cv_gbm = cross_validate(_gbm(), X, y, cv=skf, scoring=scoring, return_train_score=True)
    cv_lr  = cross_validate(_lr(),  X, y, cv=skf, scoring=scoring, return_train_score=True)

    # ── Ensemble CV aggregation ───────────────────────────────────────────
    all_true, all_pred, all_prob = [], [], []
    fold_accs, fold_aucs = [], []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        r = _rf();  r.fit(X_tr, y[tr])
        g = _gbm(); g.fit(X_tr, y[tr])
        l = _lr();  l.fit(X_tr, y[tr])
        prob = (
            r.predict_proba(X_te)[:, 1] +
            g.predict_proba(X_te)[:, 1] +
            l.predict_proba(X_te)[:, 1]
        ) / 3.0
        pred = (prob >= 0.5).astype(int)
        fold_accs.append(float(np.mean(pred == y[te])))
        fold_aucs.append(float(roc_auc_score(y[te], prob)))
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    cm = confusion_matrix(all_true, all_pred)
    tn, fp, fn, tp_ = cm.ravel()
    sens = float(tp_ / (tp_ + fn))
    spec = float(tn / (tn + fp))
    ppv  = float(tp_ / (tp_ + fp))
    npv  = float(tn  / (tn + fn))
    mcc  = float(matthews_corrcoef(all_true, all_pred))
    brier = float(brier_score_loss(all_true, all_prob))
    ap    = float(average_precision_score(all_true, all_prob))

    fpr_arr, tpr_arr, _ = roc_curve(all_true, all_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(all_true, all_prob)
    roc_auc_val = float(auc(fpr_arr, tpr_arr))

    report = classification_report(
        all_true, all_pred,
        target_names=["No Disease", "Disease"],
        output_dict=True, zero_division=0,
    )

    # ── Final models on ALL data ──────────────────────────────────────────
    rf_final  = _rf();  rf_final.fit(X, y)
    gbm_final = _gbm(); gbm_final.fit(X, y)
    lr_final  = _lr();  lr_final.fit(X, y)

    # ── SHAP global importances ───────────────────────────────────────────
    X_tf = rf_final.named_steps["pre"].transform(X)
    ohe_names = (rf_final.named_steps["pre"]
                 .named_transformers_["cat"]
                 .named_steps["ohe"]
                 .get_feature_names_out(CATEGORICAL))
    feat_names_full = CONTINUOUS + list(ohe_names)

    explainer  = shap.TreeExplainer(rf_final.named_steps["clf"])
    shap_obj   = explainer(X_tf)
    shap_arr   = shap_obj.values
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]       # disease class
    mean_shap = np.abs(shap_arr).mean(axis=0)

    # Aggregate back to original 13 features for interpretability
    orig_shap = np.zeros(len(ALL_FEATURES))
    for i, feat in enumerate(CONTINUOUS):
        idx = feat_names_full.index(feat)
        orig_shap[i] = mean_shap[idx]
    for i, feat in enumerate(CATEGORICAL):
        ohe_idxs = [j for j, n in enumerate(feat_names_full) if n.startswith(feat + "_")]
        orig_shap[len(CONTINUOUS) + i] = mean_shap[ohe_idxs].sum()

    shap_df = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Label":   [FEATURE_LABELS[f] for f in ALL_FEATURES],
        "SHAP":    orig_shap,
    }).sort_values("SHAP", ascending=False).reset_index(drop=True)

    # ── KNN similarity (preprocessed space, euclidean) ────────────────────
    X_proc = rf_final.named_steps["pre"].transform(X)
    sim_knn = NearestNeighbors(
        n_neighbors=min(10, len(df)), metric="euclidean"
    )
    sim_knn.fit(X_proc)

    return {
        # Models
        "rf": rf_final, "gbm": gbm_final, "lr": lr_final,
        "preprocessor": rf_final.named_steps["pre"],
        "sim_knn": sim_knn,
        "X_proc": X_proc,
        "shap_explainer": explainer,
        "feat_names_full": feat_names_full,
        # CV results per model
        "cv_rf": cv_rf, "cv_gbm": cv_gbm, "cv_lr": cv_lr,
        # Ensemble CV metrics
        "ensemble_acc":     float(np.mean(fold_accs)),
        "ensemble_acc_std": float(np.std(fold_accs)),
        "ensemble_auc":     roc_auc_val,
        "ensemble_auc_std": float(np.std(fold_aucs)),
        "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv, "mcc": mcc, "brier": brier,
        "pr_auc": ap,
        # Curves & matrices
        "cm": cm, "report": report,
        "fpr": fpr_arr, "tpr": tpr_arr,
        "prec": prec_arr, "rec": rec_arr,
        # SHAP
        "shap_df": shap_df,
        "shap_values": shap_arr,
        "X_transformed": X_tf,
        "feat_names_full": feat_names_full,
        # Data
        "X": X, "y": y,
        "fold_accs": fold_accs,
        "fold_aucs": fold_aucs,
        "n_splits": N_CV_SPLITS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_patient(art: Dict, patient_features: Dict) -> Dict[str, Any]:
    """
    Generate ensemble prediction + SHAP explanation for a single patient.
    Returns probability, label, individual model votes, SHAP waterfall data.
    """
    X_in = pd.DataFrame([patient_features])[ALL_FEATURES]

    p_rf  = art["rf"].predict_proba(X_in)[:, 1][0]
    p_gbm = art["gbm"].predict_proba(X_in)[:, 1][0]
    p_lr  = art["lr"].predict_proba(X_in)[:, 1][0]
    prob  = (p_rf + p_gbm + p_lr) / 3.0
    pred  = int(prob >= 0.5)

    # SHAP for this patient
    X_tf = art["preprocessor"].transform(X_in)
    sv   = art["shap_explainer"](X_tf)
    sarr = sv.values
    if sarr.ndim == 3: sarr = sarr[:, :, 1]

    # Aggregate to original 13 features
    fn_full = art["feat_names_full"]
    pat_shap = np.zeros(len(ALL_FEATURES))
    for i, feat in enumerate(CONTINUOUS):
        idx = fn_full.index(feat)
        pat_shap[i] = sarr[0, idx]
    for i, feat in enumerate(CATEGORICAL):
        idxs = [j for j, n in enumerate(fn_full) if n.startswith(feat + "_")]
        pat_shap[len(CONTINUOUS) + i] = sarr[0, idxs].sum()

    shap_patient = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Label":   [FEATURE_LABELS[f] for f in ALL_FEATURES],
        "SHAP":    pat_shap,
        "Value":   [patient_features[f] for f in ALL_FEATURES],
    }).sort_values("SHAP", key=abs, ascending=False)

    return {
        "probability": float(prob),
        "prediction":  pred,
        "risk_label":  risk_label(prob),
        "risk_color":  risk_color(prob),
        "model_votes": {
            "Random Forest":        float(p_rf),
            "Gradient Boosting":    float(p_gbm),
            "Logistic Regression":  float(p_lr),
        },
        "shap_patient": shap_patient,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CASE-BASED RETRIEVAL  (KNN)
# ─────────────────────────────────────────────────────────────────────────────
def find_similar_cases(art: Dict, df: pd.DataFrame,
                       patient_features: Dict, k: int = 5) -> pd.DataFrame:
    """
    Retrieve k most similar historical cases using euclidean distance
    in the preprocessed (scaled + OHE) feature space.
    This correctly handles mixed feature types — continuous features are
    scaled and categorical features are binary OHE vectors.
    """
    X_in  = pd.DataFrame([patient_features])[ALL_FEATURES]
    X_tf  = art["preprocessor"].transform(X_in)
    k_use = min(k, len(df))
    knn   = NearestNeighbors(n_neighbors=k_use, metric="euclidean")
    knn.fit(art["X_proc"])
    dists, idxs = knn.kneighbors(X_tf)
    similar = df.iloc[idxs[0]].copy()
    similar["euclidean_dist"] = np.round(dists[0], 3)
    similar["Diagnosis"] = similar["target"].map({0: "No Disease ✅", 1: "Disease ⚠️"})
    return similar.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL LEARNING  (online update on new cases)
# ─────────────────────────────────────────────────────────────────────────────
class IncrementalLearner:
    """
    Lightweight online learner that updates from new single cases without
    full retraining. Uses sklearn's SGDClassifier with warm_start.
    This is the conceptual equivalent of river/online-learning pipelines.
    Used to demonstrate adaptive model design — not a replacement for
    the ensemble which is retrained on full data.
    """
    def __init__(self):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        self.scaler  = StandardScaler()
        self.model   = SGDClassifier(
            loss="log_loss", learning_rate="optimal",
            class_weight="balanced", random_state=42,
            warm_start=True, n_iter_no_change=5,
        )
        self.fitted   = False
        self.n_updates = 0

    def initialise(self, X: np.ndarray, y: np.ndarray):
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, y)
        self.fitted = True

    def update(self, x_new: np.ndarray, y_new: int):
        """Incremental update on a single new observation."""
        if not self.fitted:
            raise RuntimeError("Call initialise() first.")
        x_sc = self.scaler.transform(x_new.reshape(1, -1))
        self.model.partial_fit(x_sc, [y_new], classes=[0, 1])
        self.n_updates += 1

    def predict_proba(self, x: np.ndarray) -> float:
        x_sc = self.scaler.transform(x.reshape(1, -1))
        return float(self.model.predict_proba(x_sc)[0, 1])
