"""
evaluate.py
-----------
Comprehensive Evaluation Suite for Weather Anomaly Detection & Diagnostics.
Assesses Model Performance against Ground-Truth Injected Anomalies.

Key Evaluation Criteria:
1. Primary Metric: RECALL (Ensuring zero-missed real sensor faults / extreme weather hazards)
2. Precision, F1-Score, Accuracy, and ROC-AUC
3. Granular Breakdown by Anomaly Category (Spike, Frozen, Drift, Communication Gap)
4. Diagnostic Root-Cause Classification Confusion Matrix
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from data_generator import generate_synthetic_weather_data
from anomaly_injector import inject_anomalies
from detector import WeatherAnomalyDetector


def evaluate_system(
    days: int = 30,
    contamination: float = 0.05,
    seed: int = 42
) -> dict:
    """
    Run end-to-end evaluation pipeline on 30 days of weather telemetry.
    """
    print("=" * 70)
    print("      AUTOMATIC WEATHER STATION ANOMALY DETECTION EVALUATION       ")
    print("=" * 70)

    # 1. Generate Base Telemetry
    print(f"[*] Generating {days} days of realistic weather telemetry...")
    df_clean = generate_synthetic_weather_data(days=days, interval_minutes=5, seed=seed)

    # 2. Inject Anomalies
    print("[*] Injecting synthetic multi-category sensor anomalies...")
    df_data = inject_anomalies(df_clean, n_spikes=8, n_frozen=8, n_drift=8, n_gaps=8, seed=seed)

    # 3. Fit and Run Multivariate PyOD IForest Detector
    print("[*] Training and executing Multivariate PyOD Isolation Forest Detector...")
    detector = WeatherAnomalyDetector(contamination=contamination, random_state=seed)
    df_results = detector.detect_and_diagnose(df_data)

    y_true = df_results["is_anomaly"].to_numpy()
    y_pred = df_results["pred_is_anomaly"].to_numpy()
    y_scores = df_results["anomaly_score"].to_numpy()

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 5. Print High-Level Evaluation Summary
    print("\n" + "-" * 70)
    print("                     CORE PERFORMANCE METRICS                      ")
    print("-" * 70)
    print(f"  Total Data Points Evaluated : {len(df_results):,}")
    print(f"  Ground-Truth Anomalies      : {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"  Model-Flagged Anomalies     : {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    print("-" * 70)
    print(f"  >>> RECALL (CRITICAL METRIC): {rec*100:.2f}%  [Target: >90% - Minimizes Missed Hazards]")
    print(f"  >>> PRECISION               : {prec*100:.2f}%")
    print(f"  >>> F1-SCORE                : {f1*100:.2f}%")
    print(f"  >>> ROC-AUC SCORE           : {auc*100:.2f}%")
    print(f"  >>> ACCURACY                : {acc*100:.2f}%")
    print("-" * 70)
    print(f"  Confusion Matrix: TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print("-" * 70)

    # 6. Granular Breakdown by Injected Anomaly Category
    print("\n" + "-" * 70)
    print("            PER-ANOMALY-CATEGORY DETECTION PERFORMANCE             ")
    print("-" * 70)
    print(f"{'Anomaly Category':<22} | {'Injected':<10} | {'Detected':<10} | {'Recall Rate':<12}")
    print("-" * 70)

    categories = [c for c in df_results["anomaly_type"].unique() if c != "Normal"]
    category_metrics = {}

    for cat in categories:
        cat_mask = df_results["anomaly_type"] == cat
        cat_total = cat_mask.sum()
        cat_detected = (cat_mask & (df_results["pred_is_anomaly"] == 1)).sum()
        cat_recall = (cat_detected / cat_total * 100.0) if cat_total > 0 else 0.0
        category_metrics[cat] = {"injected": int(cat_total), "detected": int(cat_detected), "recall": cat_recall}
        print(f"{cat:<22} | {cat_total:<10} | {cat_detected:<10} | {cat_recall:.2f}%")

    print("-" * 70)

    # 7. Root-Cause Diagnostic Performance
    print("\n" + "-" * 70)
    print("            ROOT-CAUSE DIAGNOSTIC CLASSIFICATION MATRIX            ")
    print("-" * 70)
    anom_subset = df_results[df_results["is_anomaly"] == 1]
    diag_crosstab = pd.crosstab(
        anom_subset["anomaly_type"],
        anom_subset["pred_anomaly_type"],
        rownames=["True Anomaly Type"],
        colnames=["Predicted Diagnosis"]
    )
    print(diag_crosstab)
    print("-" * 70)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "category_metrics": category_metrics
    }


if __name__ == "__main__":
    evaluate_system()
