"""Explicit, reproducible production training workflow.

Loads the 24-station HISTORICAL dataset (v2 = station-distinct synthetic),
splits it CHRONOLOGICALLY into TRAIN / VALIDATION / TEST (70 / 15 / 15 by
timestamp, never shuffled), trains the production anomaly detector on the TRAIN
split only, and persists a versioned model artifact (models/skyguard_v3/) plus
metadata that carries the actual training provenance (dataset version, sha256,
station/variable/feature counts).

The controlled EVALUATION dataset is intentionally NEVER used here; it belongs
exclusively to the Evaluation page.

No random, fabricated, or hardcoded predictions are produced by this module.
"""

import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MODEL_DIR, BASE_DIR, DATASET_VERSION, CALIBRATION_METHOD
from ..ingestion import load_csv, normalize
from ..preprocessing import validate, impute
from ..data_generator_meta import STATION_META
from .pipeline import ModelManager, ANOMALY_THRESHOLD, CALIBRATION_STATUS, TEMPORAL_COLS, BASELINE_COLS
from .multivariate import MULTIVARIATE_FEATURES

MODEL_SUBDIR = "skyguard_v3"
MODEL_VERSION = "skyguard-multivariate-v3"
MODEL_NAME = "SkyGuard Multivariate Fusion (Isolation Forest)"
PRODUCTION_THRESHOLD = 0.30

DATA_SOURCE = "HISTORICAL"
VARIABLES_COVERED = ["temperature", "pressure", "humidity"]

# Chronological split fractions (earliest -> latest). Never shuffled.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.85  # cumulative end of validation (0.15 validation band)


def _load_historical():
    """Load and validate the canonical 24-station historical source (v2)."""
    raw = load_csv()
    df = normalize(raw)
    df = validate(df)
    df = df.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df


def _frame_sha256(df) -> str:
    """Deterministic SHA-256 of the actual dataset content (source of truth)."""
    digest = hashlib.sha256()
    digest.update(df.sort_values(["station_id", "timestamp"]).to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def split_chronologically(df) -> tuple:
    """Split a time-series frame chronologically (no shuffle) into
    TRAIN / VALIDATION / TEST by timestamp quantiles. Returns
    (train_df, val_df, test_df)."""
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")
    ts = pd.to_datetime(df["timestamp"])
    t_train = ts.quantile(TRAIN_FRACTION)
    t_val = ts.quantile(VAL_FRACTION)
    train_df = df[ts <= t_train].copy()
    val_df = df[(ts > t_train) & (ts <= t_val)].copy()
    test_df = df[ts > t_val].copy()
    return train_df, val_df, test_df


def _artifact_dir() -> Path:
    d = MODEL_DIR / MODEL_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_production(force: bool = False) -> Path:
    """Train (or force-retrain) and persist the production model.

    Returns the artifact directory path. If an artifact already exists and
    force=False, it is loaded and returned without retraining.
    """
    art = _artifact_dir()
    model_path = art / "model.pkl"
    meta_path = art / "metadata.json"
    schema_path = art / "feature_schema.json"

    if not force and model_path.exists() and meta_path.exists():
        return art

    df = _load_historical()
    train_df, val_df, test_df = split_chronologically(df)

    # Train ONLY on the TRAIN split, across all 24 canonical stations.
    model = ModelManager(station_id="MULTI_STATION_TRAIN")
    model.train(train_df)

    full_cols = model.full_cols
    threshold = PRODUCTION_THRESHOLD  # production decision threshold
    station_ids = sorted(df["station_id"].unique())
    dataset_sha256 = _frame_sha256(df)

    # Persist the model artifact.
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "version": MODEL_VERSION,
                "model_name": MODEL_NAME,
                "dataset_version": DATASET_VERSION,
                "dataset_sha256": dataset_sha256,
                "baseline": model.baseline,
                "temporal": model.temporal,
                "multivariate": model.multivariate,
                "seasonal": model.seasonal,
                "full_cols": full_cols,
                "train_decisions": getattr(model.baseline, "train_decisions", None),
                "station_ids": station_ids,
                "threshold": threshold,
            },
            f,
        )

    # Persist feature schema.
    schema = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "features": full_cols,
        "feature_engineering": [
            "temporal: change/rolling_mean/rolling_std/roc/volatility/hour",
            "multivariate: dew_point/humidity_vs_vapor/mixing_ratio/delta_t/delta_p",
        ],
        "grouping": "per-station feature engineering to avoid cross-station bleed",
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Persist training metadata (actual training artifacts, never hard-coded).
    meta = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "algorithm": "Isolation Forest (baseline + temporal + multivariate + seasonal fusion)",
        "threshold": threshold,
        "data_source": DATA_SOURCE,
        "dataset_file": str(BASE_DIR.parent / "data" / "sample_weather.csv"),
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": dataset_sha256,
        "synthetic": True,
        "training_stations": station_ids,
        "n_training_stations": len(station_ids),
        "stations_covered": station_ids,
        "n_stations": len(station_ids),
        "variables_covered": VARIABLES_COVERED,
        "n_variables": len(VARIABLES_COVERED),
        "n_observations": int(len(df)),
        "n_training_obs": int(len(train_df)),
        "n_validation_obs": int(len(val_df)),
        "n_test_obs": int(len(test_df)),
        "split": {
            "method": "chronological by timestamp quantile, never shuffled",
            "train_fraction": TRAIN_FRACTION,
            "train": "earliest 70%",
            "validation": "next 15%",
            "test": "latest 15%",
            "validation_cumulative_end": VAL_FRACTION,
        },
        "n_features": len(full_cols),
        "feature_count": len(full_cols),
        "features": full_cols,
        "calibration": {
            "method": CALIBRATION_METHOD,
            "status": CALIBRATION_STATUS,
            "note": "Uncalibrated until a validated calibration split proves improvement.",
        },
        "notes": "Synthetic, station-distinct v2 telemetry (lat/elevation/diurnal/spatial correlation); not live AWS.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return art


def load_production_model():
    """Load the persisted production model artifact.

    Returns (ModelManager, metadata dict) or (None, None) if absent.
    """
    return load_versioned(MODEL_SUBDIR)


def load_versioned(subdir):
    """Load a model artifact from a specific version subdir (v1/v2/...).

    Returns (ModelManager, metadata dict) or (None, None) if absent.
    Used for the Evaluation page's measured v1-vs-v2 comparison.
    """
    art = MODEL_DIR / subdir
    model_path = art / "model.pkl"
    meta_path = art / "metadata.json"
    if not model_path.exists():
        return None, None

    # Ensure backend directory is in sys.path for pickle unpickling
    backend_dir = Path(__file__).resolve().parent.parent.parent
    backend_dir_str = str(backend_dir)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)

    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    model = ModelManager(station_id="MULTI_STATION_TRAIN")
    model.baseline = blob.get("baseline")
    model.temporal = blob.get("temporal")
    model.multivariate = blob.get("multivariate")
    model.seasonal = blob.get("seasonal")
    model.full_cols = blob.get("full_cols") or []
    model.threshold = blob.get("threshold", ANOMALY_THRESHOLD)
    model.fusion_weights = blob.get("fusion_weights", {"ml": 1.0, "drift": 0.0, "flat": 0.0, "miss": 0.0})
    model.extra_channels = blob.get("extra_channels", {})
    if getattr(model.baseline, "train_decisions", None) is None and "train_decisions" in blob:
        model.baseline.train_decisions = blob["train_decisions"]
    model.trained = True
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta


def try_ensure_trained():
    """Ensure a persisted model exists, training if needed. Returns metadata."""
    model, meta = load_production_model()
    if model is not None:
        return meta
    train_production(force=True)
    _, meta = load_production_model()
    return meta


# ==========================================================================
#  v3 — mechanism-specific campaign (additive; never touches v1/v2)
# ==========================================================================

V3_SUBDIR = "skyguard_v3"
V3_VERSION = "skyguard-multivariate-v3"
V3_MODEL_NAME = "SkyGuard v3 (ML + dedicated drift/flatline/missingness channels)"
V3_DATA_FILE = BASE_DIR.parent / "data" / "sample_weather_v3.csv"
V3_META_FILE = BASE_DIR.parent / "data" / "sample_weather_v3.meta.json"
V3_EXPECTED_SHA256 = "7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511"
V3_TRAIN_FRACTION = 0.70
V3_VAL_FRACTION = 0.85
V3_VAL_SEED = 11        # validation tuning frame (never the benchmark seed 7/19)
V3_UNSEEN_SEED = 19     # robustness frame, used only for final reporting (Phase 3)

# Detector-parameter configurations trialed on VALIDATION only. {} = defaults.
# Every combination is recorded in ml/experiments/; none are trained on TEST.
V3_DETECTOR_CONFIGS = [
    {},
    {"drift": {"ewma_alpha": 0.40}},
    {"flatline": {"window": 8}},
    {"flatline": {"window": 8, "ratio_floor": 0.30}},
    {"drift": {"ewma_alpha": 0.40}, "flatline": {"window": 8, "ratio_floor": 0.30}},
]


def _run_slug():
    """Short deterministic-ish run identifier for experiment logs."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _verify_v3_dataset() -> dict:
    """Verify the on-disk v3 dataset matches the published provenance.

    Checks: file exists, the generator meta.json exists, station/observation
    counts and the content SHA-256 match the expected constant. Raises on any
    mismatch so training STOPS rather than silently using a different dataset.
    """
    if not V3_DATA_FILE.exists():
        raise FileNotFoundError(f"v3 dataset missing: {V3_DATA_FILE}")
    if not V3_META_FILE.exists():
        raise FileNotFoundError(f"v3 meta missing: {V3_META_FILE}")

    with open(V3_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = _load_v3_historical()
    content_sha = _frame_sha256(df)

    if content_sha != V3_EXPECTED_SHA256:
        raise ValueError(
            f"v3 dataset SHA-256 MISMATCH: expected {V3_EXPECTED_SHA256}, got {content_sha}. "
            "STOPPING training; do not regenerate or substitute the dataset."
        )
    if (meta.get("sha256") or "").lower() != content_sha:
        raise ValueError("v3 generator meta sha256 disagrees with the on-disk content hash.")
    if meta.get("n_observations") != len(df):
        raise ValueError(f"v3 observation count mismatch: meta={meta.get('n_observations')}, file={len(df)}")
    if meta.get("n_stations") != df["station_id"].nunique():
        raise ValueError("v3 station count mismatch.")
    return meta


def _load_v3_historical():
    raw = load_csv(V3_DATA_FILE)
    df = normalize(raw)
    df = validate(df)
    df = df.drop_duplicates(subset=["station_id", "timestamp"], keep="last").reset_index(drop=True)
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def split_chronological_v3(df):
    """Chronological TRAIN/VALIDATION/TEST split by timestamp (never shuffled).

    Returns (train_df, val_df, test_df, boundaries) where boundaries carries
    train/val/test start+end timestamps for provenance.
    """
    ts = pd.to_datetime(df["timestamp"])
    t_train = ts.quantile(V3_TRAIN_FRACTION)
    t_val = ts.quantile(V3_VAL_FRACTION)
    train_df = df[ts <= t_train].copy()
    val_df = df[(ts > t_train) & (ts <= t_val)].copy()
    test_df = df[ts > t_val].copy()

    def bounds(sub):
        s = pd.to_datetime(sub["timestamp"])
        return {"start": str(s.min()), "end": str(s.max())}

    boundaries = {
        "method": "chronological by timestamp quantile, never shuffled",
        "train_fraction": V3_TRAIN_FRACTION,
        "train": bounds(train_df),
        "validation": bounds(val_df),
        "test": bounds(test_df),
    }
    return train_df, val_df, test_df, boundaries


def _effective_params(det, names):
    return {k: getattr(det, k) for k in names if hasattr(det, k)}


def _ml_hyperparameters(model):
    """Record the ACTUAL fitted ML-channel hyperparameters (never fabricated)."""
    channel_features = {
        "baseline": list(BASELINE_COLS) if BASELINE_COLS else None,
        "temporal": list(TEMPORAL_COLS),
        "multivariate": list(MULTIVARIATE_FEATURES),
    }

    def iso(label, det):
        if det is None or det.model is None:
            return None
        return {
            "n_estimators": det.model.n_estimators,
            "max_samples": det.model.max_samples,
            "max_features": det.model.max_features,
            "contamination": det.model.contamination,
            "random_state": det.model.random_state,
            "feature_names": list(getattr(det, "features", None) or channel_features.get(label) or []),
            "features_standardized_by": "StandardScaler (fit on TRAIN only)",
            "decision_to_score": "anchored sigmoid of standardized decision function (uncalibrated)",
        }

    return {
        "baseline": iso("baseline", model.baseline),
        "temporal": iso("temporal", model.temporal),
        "multivariate": iso("multivariate", model.multivariate),
        "seasonal": getattr(model.seasonal, "hyperparameters", None)
        or "SeasonalDetector (diurnal profile residuals; no sklearn RandomState used)",
    }


def _tune_v3_fusion(model, val_scores, train_df, context):
    """Validation-only grid search: detector params x fusion weights x threshold.

    Selection objective (primary): maximize F1 subject to precision >= 0.85;
    (secondary, in order): PR-AUC, min(DRIFT/FROZEN/MISSING recall), ROC-AUC.
    Every candidate experiment is persisted under ml/experiments/ (failed
    experiments included, never deleted).

    Returns (weights, threshold, selected_detector_config, selected_experiment,
             records) and sets nothing on `model` (caller persists).
    """
    from .drift import DriftDetector
    from .flatline import FlatlineDetector
    from .missingness import MissingnessDetector
    from .evaluation import evaluate

    df, gt, true_type, summary = val_scores
    injected_type = df["injected_type"].to_numpy(dtype=object) if "injected_type" in df.columns else None
    ml = np.asarray(model.ml_fused_scores(df), dtype=float)
    n = len(df)

    weights_grid = [0.0, 0.10, 0.20, 0.30, 0.45]
    thresholds = np.round(np.arange(0.28, 0.781, 0.02), 3)

    experiments = []
    counter = 0

    for dc in V3_DETECTOR_CONFIGS:
        drift_params = dc.get("drift", {})
        flat_params = dc.get("flatline", {})
        miss_params = dc.get("missingness", {})
        drift_det = DriftDetector(**drift_params).fit(train_df)
        flat_det = FlatlineDetector(**flat_params).fit(train_df)
        miss_det = MissingnessDetector(**miss_params).fit(train_df)
        drift_scores = np.asarray(drift_det.score(df), dtype=float)
        flat_scores = np.asarray(flat_det.score(df), dtype=float)
        miss_scores = np.asarray(miss_det.score_raw(df), dtype=float)
        detector_params = {
            "drift": _effective_params(drift_det, ("ewma_alpha", "cusum_k", "cusum_h", "slope_win", "epsilon")),
            "flatline": _effective_params(flat_det, ("window", "ratio_floor", "eps")),
            "missingness": _effective_params(miss_det, ("interval_tol_fraction", "window", "eps")),
        }

        for wd in weights_grid:
            for wf in weights_grid:
                for wm in weights_grid:
                    extra = wd + wf + wm
                    if 1.0 - extra < 0.20:
                        continue
                    weights = {"ml": round(1.0 - extra, 3), "drift": wd, "flat": wf, "miss": wm}
                    fused = weights["ml"] * ml + wd * drift_scores + wf * flat_scores + wm * miss_scores
                    for thr in thresholds:
                        counter += 1
                        y_pred = (fused > thr).astype(int)
                        m = evaluate(gt, y_pred, scores=fused)
                        per_cat = {}
                        if injected_type is not None:
                            for typ in ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA"):
                                mask = (injected_type == typ) & (gt == 1)
                                total = int(mask.sum())
                                per_cat[typ] = round(
                                    int((mask & (y_pred == 1)).sum()) / total, 4) if total else None
                        target_recalls = [r for k, r in per_cat.items()
                                          if k in ("DRIFT", "FROZEN_SENSOR", "MISSING_DATA") and r is not None]
                        experiments.append({
                            "experiment_id": f"{context['run_id']}:{counter:05d}",
                            "timestamp_utc": context["timestamp_utc"],
                            "dataset_hash": context["dataset_hash"],
                            "split": context["split_pub"],
                            "features": context["feature_list"],
                            "detector_parameters": detector_params,
                            "fusion_weights": weights,
                            "threshold": float(thr),
                            "n": n,
                            "anomaly_rate": round(float(gt.mean()), 4),
                            "precision": m["precision"],
                            "recall": m["recall"],
                            "f1": m["f1"],
                            "roc_auc": m["roc_auc"],
                            "pr_auc": m["pr_auc"],
                            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                            "per_category_recall": per_cat,
                            "min_target_recall": round(min(target_recalls), 4) if target_recalls else None,
                            "confusion_matrix": m["confusion_matrix"],
                            "selection_reason": None,
                            "params_key": dc,
                        })

    # ---- selection rule (validation only, no test/benchmark inspection) ----
    feasible = [e for e in experiments if e["precision"] >= 0.85]

    def key(e):
        return (
            e["f1"],
            e["pr_auc"] if e["pr_auc"] is not None else 0.0,
            e["min_target_recall"] if e["min_target_recall"] is not None else 0.0,
            e["roc_auc"] if e["roc_auc"] is not None else 0.0,
        )

    if feasible:
        selected = max(feasible, key=key)
        reason = ("max F1 (precision >= 0.85); tie-break PR-AUC, then min(DRIFT/"
                  "FROZEN/MISSING recall), then ROC-AUC; all on VALIDATION seed 11")
    else:
        # fallback: v2-equivalent ML-only at the system threshold (recorded,
        # not silently tuned) — honest control if nothing beats the floor.
        selected = min(experiments, key=lambda e: abs(e["precision"] - 0.85) + abs(e["threshold"] - PRODUCTION_THRESHOLD))
        selected = {"fusion_weights": {"ml": 1.0, "drift": 0.0, "flat": 0.0, "miss": 0.0},
                    "threshold": PRODUCTION_THRESHOLD}
        reason = ("NO candidate reached validation precision floor (0.85) -> "
                  "fallback to ML-only v2-equivalent at production threshold 0.30 (control)")

    n_experiments = len(experiments)
    records = {
        "campaign": "skyguard_v3 validation tuning (Phase 2)",
        "run_id": context["run_id"],
        "timestamp_utc": context["timestamp_utc"],
        "dataset_hash": context["dataset_hash"],
        "frame": f"validation_seed{V3_VAL_SEED}",
        "n": n,
        "anomaly_rate": round(float(gt.mean()), 4),
        "selection_rule": "max F1 subject to precision>=0.85; tie-break PR-AUC, min target recall, ROC-AUC",
        "selection_reason": reason,
        "selected_experiment_id": selected.get("experiment_id"),
        "selected": {k: selected[k] for k in ("fusion_weights", "threshold", "precision", "recall",
                                              "f1", "roc_auc", "pr_auc", "per_category_recall",
                                              "detector_parameters", "tp", "fp", "fn", "tn")} if isinstance(selected, dict) and "precision" in selected else selected,
        "n_experiments": n_experiments,
        "experiments": experiments,
    }

    exp_dir = Path(BASE_DIR).parent / "ml" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    slug = _run_slug()
    with open(exp_dir / f"v3_val_tuning_seed11_{slug}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    with open(exp_dir / "v3_val_tuning_seed11_latest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    if "experiment_id" in selected:
        return (selected["fusion_weights"], float(selected["threshold"]),
                selected["detector_parameters"], selected, records)
    return ({"ml": 1.0, "drift": 0.0, "flat": 0.0, "miss": 0.0}, PRODUCTION_THRESHOLD,
            V3_DETECTOR_CONFIGS[0], selected, records)


def _build_v3_val_frame(val_df):
    """Validation tuning frame: injected anomalies on the v3 validation period."""
    from ..services import app as skyguard  # lazy: services imports this module

    base = val_df[val_df["station_id"] == "AWS-023"].sort_values("timestamp").reset_index(drop=True)
    base = base[["timestamp", "temperature", "pressure", "humidity"]].head(600).copy()
    return skyguard._build_eval_frame(rows=600, seed=V3_VAL_SEED, base_frame=base)


def train_v3(force: bool = False) -> Path:
    """Train and persist skyguard_v3 (offline, frozen, validation-tuned).

    * Dataset: sample_weather_v3.csv, SHA-256 verified BEFORE training.
    * Split: chronological 70/15/15 (train/validation/test), never shuffled.
    * ML channel: identical feature sets/formula as v2, trained on TRAIN only.
    * Dedicated channels (drift/flatline/missingness): fitted on TRAIN only;
      missingness scored on the RAW pre-impute stream.
    * Fusion weights + threshold + detector params: selected on the VALIDATION
      tuning frame (seed 11). Benchmark (seed 7) and unseen (seed 19) are NEVER
      used for model/hyperparameter selection.
    * TEST is not used anywhere in this phase (final evaluation is Phase 3).
    """
    from .drift import DriftDetector
    from .flatline import FlatlineDetector
    from .missingness import MissingnessDetector

    art = MODEL_DIR / V3_SUBDIR
    art.mkdir(parents=True, exist_ok=True)
    model_path = art / "model.pkl"
    if not force and model_path.exists():
        return art

    gen_meta = _verify_v3_dataset()  # STOPS (raises) if the dataset hash is wrong
    df = _load_v3_historical()
    train_df, val_df, test_df, boundaries = split_chronological_v3(df)

    model = ModelManager(station_id="MULTI_STATION_TRAIN")
    model.train(train_df)  # ML channels: baseline/temporal/multivariate/seasonal on TRAIN only

    val_scores = _build_v3_val_frame(val_df)
    full_cols = model.full_cols
    dataset_sha256 = _frame_sha256(df)
    station_ids = sorted(df["station_id"].unique())
    now = datetime.utcnow().isoformat() + "Z"
    run_id = _run_slug()

    context = {
        "run_id": run_id,
        "timestamp_utc": now,
        "dataset_hash": dataset_sha256,
        "feature_list": full_cols,
        "split_pub": boundaries,
    }
    weights, threshold, det_params, val_metrics, records = _tune_v3_fusion(
        model, val_scores, train_df, context
    )
    model.fusion_weights = weights
    model.threshold = float(threshold)
    # rebuild the channels with the SELECTED detector configuration (fitted on TRAIN)
    model.extra_channels = {
        "drift": DriftDetector(**det_params.get("drift", {})).fit(train_df),
        "flatline": FlatlineDetector(**det_params.get("flatline", {})).fit(train_df),
        "missingness": MissingnessDetector(**det_params.get("missingness", {})).fit(train_df),
    }

    # ---- persist artifact (model.pkl / metadata.json / feature_schema.json) ----
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "version": V3_VERSION,
                "model_name": V3_MODEL_NAME,
                "dataset_version": "sample_weather_v3",
                "dataset_sha256": dataset_sha256,
                "baseline": model.baseline,
                "temporal": model.temporal,
                "multivariate": model.multivariate,
                "seasonal": model.seasonal,
                "full_cols": full_cols,
                "train_decisions": getattr(model.baseline, "train_decisions", None),
                "station_ids": station_ids,
                "threshold": model.threshold,
                "fusion_weights": model.fusion_weights,
                "extra_channels": model.extra_channels,
            },
            f,
        )

    ml_hp = _ml_hyperparameters(model)
    with open(art / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_version": V3_VERSION,
            "dataset_version": "sample_weather_v3",
            "features": full_cols,
            "ml_feature_sets": {
                "baseline": ["temperature", "pressure", "humidity"],
                "temporal": list(TEMPORAL_COLS),
                "multivariate": list(MULTIVARIATE_FEATURES),
                "seasonal": "cyclic hour-of-day + raw vars",
            },
            "ml_hyperparameters": ml_hp,
            "channel_features": {
                "drift": "residual = observed - train-hour-of-day expectation; robust slope + EWMA + CUSUM + persistence",
                "flatline": "roll-std vs train baseline + zero-diff ratio + unique-value count + plateau persistence",
                "missingness": "RAW pre-impute: NaN/invalid runs, per-var missingness, trailing fraction, timestamp gaps",
            },
            "detector_parameters": det_params,
            "grouping": "per-station feature engineering to avoid cross-station bleed",
        }, f, indent=2)

    val_out = val_metrics if "precision" in val_metrics else None
    meta = {
        "model_name": V3_MODEL_NAME,
        "model_version": V3_VERSION,
        "dataset_name": "sample_weather_v3",
        "trained_at_utc": now,
        "run_id": run_id,
        "algorithm": "Isolation Forest ML fusion (baseline+temporal+multivariate+seasonal) "
                     "+ statistical drift/flatline/missingness channels + weighted evidence fusion",
        "threshold": model.threshold,
        "fusion_weights": model.fusion_weights,
        "data_source": "HISTORICAL (synthetic v3)",
        "dataset_file": str(V3_DATA_FILE),
        "dataset_version": "sample_weather_v3",
        "dataset_sha256": dataset_sha256,
        "synthetic": True,
        "feature_version": "v3.0",
        "training_stations": station_ids,
        "n_stations": len(station_ids),
        "station_count": len(station_ids),
        "variables_covered": VARIABLES_COVERED,
        "n_observations": int(len(df)),
        "observation_count": int(len(df)),
        "n_training_obs": int(len(train_df)),
        "n_validation_obs": int(len(val_df)),
        "n_test_obs": int(len(test_df)),
        "split": boundaries,
        "n_features": len(full_cols),
        "feature_count": len(full_cols),
        "features": full_cols,
        "ml_hyperparameters": ml_hp,
        "detector_hyperparameters": det_params,
        "random_seeds": {
            "dataset_generation": {
                "seed": gen_meta.get("seed"),
                "regime_seed": gen_meta.get("regime_seed"),
                "episode_seed": gen_meta.get("episode_seed"),
            },
            "validation_frame": V3_VAL_SEED,
            "benchmark_frame": 7,
            "unseen_frame": V3_UNSEEN_SEED,
            "isolation_forest": 42,
        },
        "tuning": {
            "method": "chronological validation frame only (seed 11)",
            "selection_rule": "max F1 subject to precision>=0.85; tie-break PR-AUC, then min(DRIFT/FROZEN/MISSING recall)",
            "benchmark_seed": 7,
            "unseen_seed": V3_UNSEEN_SEED,
            "never_uses": "benchmark (seed 7) and unseen (seed 19) frames; TEST split",
        },
        "validation_metrics": val_out,
        "test_metrics": None,
        "test_metrics_note": "Not computed in Phase 2. Final benchmark (seed 7) and unseen "
                             "(seed 19) evaluation land in Phase 3 and are appended here.",
        "calibration": {"method": "none", "status": "uncalibrated",
                        "note": "Scores remain uncalibrated; no calibration split claimed; "
                                "detector outputs are evidence, not probabilities."},
        "training_mode": {
            "offline": True,
            "frozen_during_inference": True,
            "online_retraining": False,
            "note": "Trained once offline and frozen; no online retraining/adaptation during inference.",
        },
        "provenance": {
            "generator": gen_meta.get("generator"),
            "interval_minutes": gen_meta.get("interval_minutes"),
            "start": gen_meta.get("start"),
            "days": gen_meta.get("days"),
            "regimes": gen_meta.get("regimes"),
            "experiment_log": f"ml/experiments/v3_val_tuning_seed11_{run_id}.json",
        },
        "notes": "v3 campaign: expanded 60-day synthetic dataset, mechanism-specific detectors; "
                 "training is offline and frozen; benchmark metrics reported separately in Phase 3.",
    }
    with open(art / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[v3] TRAIN OK run={run_id}")
    print(f"[v3] dataset sha256={dataset_sha256}")
    print(f"[v3] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({len(station_ids)} stations)")
    print(f"[v3] threshold={model.threshold} weights={model.fusion_weights}")
    if val_out:
        print(f"[v3] validation F1={val_out['f1']} precision={val_out['precision']} "
              f"recall={val_out['recall']} roc_auc={val_out['roc_auc']} pr_auc={val_out['pr_auc']}")
        print(f"[v3] validation per-category={val_out['per_category_recall']}")
    return art


def verify_v3_train_load() -> bool:
    """Load the persisted v3 artifact from disk and verify it reproduces the
    validation metrics recorded in metadata (deterministic check).

    Returns True when the reloaded model reproduces recorded validation metrics
    (AND the recorded threshold/weights), i.e. a clean train->save->load cycle.
    """
    from .evaluation import evaluate

    model, meta = load_versioned(V3_SUBDIR)
    if model is None or meta is None:
        print("[v3] VERIFY FAIL: artifact not found")
        return False

    df = _load_v3_historical()
    _, val_df, _test_df, _bounds = split_chronological_v3(df)
    ev_df, gt, _true_type, _summary = _build_v3_val_frame(val_df)
    scores = np.asarray(model.score_series_scores(ev_df), dtype=float)
    recorded = meta.get("validation_metrics")
    reported = evaluate(gt, (scores > model.threshold).astype(int), scores=scores)

    ok_threshold = abs(model.threshold - recorded["threshold"]) < 1e-9
    ok_f1 = abs(reported["f1"] - recorded["f1"]) < 1e-4
    ok_pr = abs(reported["precision"] - recorded["precision"]) < 1e-4
    ok_rec = abs(reported["recall"] - recorded["recall"]) < 1e-4
    ok_roc = (reported["roc_auc"] == (recorded["roc_auc"] if recorded.get("roc_auc") is not None else None))
    ok_pr_auc = (reported["pr_auc"] == (recorded["pr_auc"] if recorded.get("pr_auc") is not None else None))
    ok = all([ok_threshold, ok_f1, ok_pr, ok_rec, ok_roc, ok_pr_auc])
    print(f"[v3] VERIFY reloaded: threshold={model.threshold} weights={model.fusion_weights}")
    print(f"[v3] VERIFY reproduced: F1={reported['f1']} precision={reported['precision']} "
          f"recall={reported['recall']} roc={reported['roc_auc']} pr_auc={reported['pr_auc']}")
    print("[v3] VERIFY OK" if ok else "[v3] VERIFY FAIL: reloaded metrics differ from recorded metadata")
    return ok

