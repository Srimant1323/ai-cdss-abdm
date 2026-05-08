"""
data_loader.py
==============
Data ingestion, validation, preprocessing, and audit trail management.

Designed for extensibility: replace CSV loading with SQLAlchemy/psycopg2
for production database integration.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Tuple, List, Dict

AUDIT_LOG_PATH = "data/audit_log.json"

THAL_MAP   = {6: 1, 3: 2, 7: 3}
DATA_PATH  = "data/heart_cleveland.csv"
BACKUP_PATH = "data/heart_cleveland_backup.csv"


def load_and_preprocess() -> pd.DataFrame:
    """Load and clean the Cleveland Heart Disease dataset."""
    df = pd.read_csv(DATA_PATH)
    df["thal"] = df["thal"].replace(THAL_MAP)
    df["ca"]   = df["ca"].clip(0, 3)
    df = _add_derived_features(df)
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically derived features for display and interpretation.
    These are NOT used in the ML model — they are for the patient-centric view.
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 70, 120],
        labels=["<40", "40–49", "50–59", "60–69", "70+"],
        right=False,
    )
    df["bp_category"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 130, 140, 200],
        labels=["Normal", "Elevated", "Stage 1 HTN", "Stage 2 HTN"],
        right=False,
    )
    df["chol_category"] = pd.cut(
        df["chol"],
        bins=[0, 200, 240, 1000],
        labels=["Desirable", "Borderline", "High"],
        right=False,
    )
    df["hr_reserve"] = df["thalach"] - (220 - df["age"])   # % of predicted max HR
    df["Diagnosis"]  = df["target"].map({0: "No Disease", 1: "Disease"})
    return df


def validate_new_case(row: Dict) -> Tuple[bool, List[str]]:
    """Clinical plausibility checks on a new case before admission."""
    errors = []
    if not (0 < row.get("age", 0) <= 120):
        errors.append("Age must be between 1 and 120.")
    if not (50 <= row.get("trestbps", 0) <= 250):
        errors.append("Resting BP (trestbps) must be 50–250 mmHg.")
    if not (100 <= row.get("chol", 0) <= 700):
        errors.append("Cholesterol must be 100–700 mg/dL.")
    if not (40 <= row.get("thalach", 0) <= 250):
        errors.append("Max heart rate must be 40–250 bpm.")
    if row.get("oldpeak", -1) < 0 or row.get("oldpeak", 10) > 10:
        errors.append("ST depression (oldpeak) must be 0–10.")
    if row.get("ca", -1) not in [0, 1, 2, 3]:
        errors.append("CA must be 0, 1, 2, or 3.")
    if row.get("thal", -1) not in [1, 2, 3]:
        errors.append("Thal must be 1 (normal), 2 (fixed), or 3 (reversible).")
    return len(errors) == 0, errors


def append_case(new_row: Dict, outcome_label: str = "Pending") -> pd.DataFrame:
    """
    Append a new validated case to the dataset with full audit trail.
    Backs up original file before modification.
    """
    df = load_and_preprocess()

    # Backup
    if not os.path.exists(BACKUP_PATH):
        df.to_csv(BACKUP_PATH, index=False)

    new_df = pd.DataFrame([new_row])
    updated = pd.concat([df.drop(columns=["age_group","bp_category","chol_category",
                                           "hr_reserve","Diagnosis"], errors="ignore"),
                         new_df], ignore_index=True)
    updated.to_csv(DATA_PATH, index=False)

    # Audit log
    _write_audit(action="CASE_ADDED", details={
        "age": new_row.get("age"),
        "sex": new_row.get("sex"),
        "target": new_row.get("target"),
        "outcome_label": outcome_label,
        "total_records_after": len(updated),
    })

    return load_and_preprocess()


def _write_audit(action: str, details: Dict):
    """Append an entry to the audit log (JSON lines format)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action":    action,
        "details":   details,
    }
    logs = []
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH) as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(entry)
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


def load_audit_log() -> pd.DataFrame:
    """Load audit log as a DataFrame for display."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return pd.DataFrame(columns=["timestamp", "action", "details"])
    with open(AUDIT_LOG_PATH) as f:
        try:
            logs = json.load(f)
        except:
            return pd.DataFrame()
    rows = []
    for entry in logs:
        row = {"timestamp": entry["timestamp"], "action": entry["action"]}
        row.update(entry.get("details", {}))
        rows.append(row)
    return pd.DataFrame(rows)
