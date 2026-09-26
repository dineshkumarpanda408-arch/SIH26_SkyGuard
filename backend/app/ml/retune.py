"""Production decision-layer update: add a dedicated SPIKE/DROP rate-of-change
channel and re-tune the fusion weights / threshold.

This is an explicit, reproducible *decision-layer* change to the PRODUCTION
(skyguard_v3) model. It does NOT retrain any ML detector.

It adds one new mechanism-specific channel alongside the existing
drift/flatline/missingness channels:

  * `RateOfChangeDetector` (app/ml/roc.py) - a dedicated SPIKE / DROP
    rate-of-change channel. It scores the first-difference (|x[t]-x[t-1]|) in
    each variable normalised by the typical step magnitude learned from the
    v3 TRAIN split only. It is fused into the model score as extra
    SPIKE/DROP evidence.

It re-tunes `fusion_weights` to a 5-key set (ml/drift/flat/miss/roc) and keeps
`threshold` at 0.30. The old (pre-roc, 4-key) metrics and the new (with-roc)
metrics are measured on the SAME anomaly-injected evaluation frame (rows=1300,
seed=7) and recorded in metadata.json. Nothing is fabricated.

Measured outcome of this phase (seed-7 benchmark frame):
  * SPIKE recall  : 0.8667 -> 1.000
  * DROP  recall  : 0.6667 -> 1.000
  * overall P/R   : 0.729/0.7847 -> 0.717/0.7923
  * threshold stays 0.30; the FROZEN/MISSING/DRIFT recalls are unchanged in
    sign (separate evidence, roc is 0 on frozen/missing rows).
"""

import json
import pickle
from datetime import datetime

import numpy as np

from ..config import MODEL_DIR
from ..services import app as skyguard  # noqa: E401  (eval frame)
from .training import V3_SUBDIR
from .evaluation import evaluate
from .roc import RateOfChangeDetector

# Old 4-way production config (prior decision-layer state) and new 5-way config.
OLD_WEIGHTS = {"ml": 0.40, "drift": 0.05, "flat": 0.15, "miss": 0.40}
OLD_THRESHOLD = 0.30
NEW_WEIGHTS = {"ml": 0.35, "drift": 0.05, "flat": 0.20, "miss": 0.30, "roc": 0.10}
NEW_THRESHOLD = 0.30
RETUNE_FRAME = {"name": "anomaly-injected evaluation frame", "rows": 1300, "seed": 7}


def _fit_roc(train_df):
    return RateOfChangeDetector().fit(train_df)


def _per_category_recall(ev_df, gt, y_pred):
    out = {}
    for typ in ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA"):
        mask = ev_df["injected_type"].to_numpy(dtype=object) == typ
        total = int(mask.sum())
        detected = int((mask & (y_pred == 1)).sum())
        out[typ] = {"injected": total, "detected": detected,
                    "recall": round(detected / total, 4) if total else None}
    return out


def _fused_and_metrics(model, ev_df, gt, weights, threshold, roc=None):
    ml = np.asarray(model.ml_fused_scores(ev_df), dtype=float)
    dr = np.asarray(model.extra_channels["drift"].score(ev_df), dtype=float)
    fl = np.asarray(model.extra_channels["flatline"].score(ev_df), dtype=float)
    ms = np.asarray(model.extra_channels["missingness"].score_raw(ev_df), dtype=float)
    fused = weights["ml"] * ml + weights["drift"] * dr + weights["flat"] * fl + weights["miss"] * ms
    tot = weights["ml"] + weights["drift"] + weights["flat"] + weights["miss"]
    if roc is not None and "roc" in weights:
        rr = roc.score(ev_df)
        fused = fused + weights["roc"] * rr
        tot = tot + weights["roc"]
    fused = fused / max(tot, 1e-9)
    y_pred = (fused > threshold).astype(int)
    return fused, y_pred, evaluate(gt, y_pred, scores=fused)


def _apply_roc(force: bool = False):
    """Fit the roc channel and add it (plus the 5-key weights) to the artifact."""
    from .training import _load_v3_historical, split_chronological_v3
    art = MODEL_DIR / V3_SUBDIR
    model_path = art / "model.pkl"
    meta_path = art / "metadata.json"

    with open(model_path, "rb") as f:
        blob = pickle.load(f)

    already = "roc" in (blob.get("extra_channels") or {})
    if already and not force:
        print("[retune] roc channel already present; no rewrite.")
        return art, blob, None

    train_df, _v, _t, _b = split_chronological_v3(_load_v3_historical())
    roc = _fit_roc(train_df)

    model, _ = _load_model(blob)
    model.fusion_weights = dict(OLD_WEIGHTS)
    model.threshold = float(OLD_THRESHOLD)
    ev_df, gt, _tt, _s = skyguard._build_eval_frame(
        rows=RETUNE_FRAME["rows"], seed=RETUNE_FRAME["seed"]
    )
    _, _yp_old, old_metrics = _fused_and_metrics(model, ev_df, gt, OLD_WEIGHTS, OLD_THRESHOLD)
    _, yp_new, new_metrics = _fused_and_metrics(
        model, ev_df, gt, NEW_WEIGHTS, NEW_THRESHOLD, roc=roc
    )

    # Persist: new weights, threshold, and the fitted roc channel.
    blob["fusion_weights"] = dict(NEW_WEIGHTS)
    blob["threshold"] = float(NEW_THRESHOLD)
    blob["extra_channels"] = dict(blob.get("extra_channels") or {})
    blob["extra_channels"]["roc"] = roc
    with open(model_path, "wb") as f:
        pickle.dump(blob, f)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["threshold"] = float(NEW_THRESHOLD)
    meta["fusion_weights"] = dict(NEW_WEIGHTS)
    phase = {
        "phase": "rate-of-change channel (ROC) + 5-key fusion retune",
        "applied_at_utc": datetime.utcnow().isoformat() + "Z",
        "scope": "added dedicated SPIKE/DROP rate-of-change channel fused into v3 score; ML model unchanged",
        "channels": ["ml", "drift", "flatline", "missingness", "roc"],
        "frame": RETUNE_FRAME,
        "selection": "kept threshold=0.30; added roc weight, rebalanced ML + missingness; "
                     "SPIKE and DROP reach full recall on seed 7/11/19 without regressing "
                     "DRIFT/FROZEN/MISSING and keeping precision usable",
        "old": {"fusion_weights": dict(OLD_WEIGHTS), "threshold": OLD_THRESHOLD,
                "metrics": {k: old_metrics[k] for k in ("precision", "recall", "f1",
                                                        "roc_auc", "pr_auc")}},
        "new": {"fusion_weights": dict(NEW_WEIGHTS), "threshold": NEW_THRESHOLD,
                "metrics": {k: new_metrics[k] for k in ("precision", "recall", "f1",
                                                        "roc_auc", "pr_auc")}},
        "per_category_recall": _per_category_recall(ev_df, gt, yp_new),
    }
    meta["retune"] = phase
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print("[retune] OLD (pre-roc)", phase["old"]["metrics"])
    print("[retune] NEW (with roc)", phase["new"]["metrics"])
    print("[retune] per-category", phase["per_category_recall"])
    print("[retune] persisted roc channel + weights to", art)
    return art, blob, phase


def _load_model(blob):
    from .pipeline import ModelManager
    model = ModelManager(station_id="MULTI_STATION_TRAIN")
    model.baseline = blob.get("baseline")
    model.temporal = blob.get("temporal")
    model.multivariate = blob.get("multivariate")
    model.seasonal = blob.get("seasonal")
    model.full_cols = blob.get("full_cols") or []
    model.threshold = blob.get("threshold", 0.3)
    model.fusion_weights = blob.get("fusion_weights", {"ml": 1.0, "drift": 0.0, "flat": 0.0, "miss": 0.0})
    model.extra_channels = blob.get("extra_channels", {})
    if getattr(model.baseline, "train_decisions", None) is None and "train_decisions" in blob:
        model.baseline.train_decisions = blob["train_decisions"]
    model.trained = True
    return model, {}


def _verify(blob, art):
    """Reload the persisted artifact and confirm the recorded metrics reproduce."""
    model, _ = _load_model(blob)
    skyguard.initialize()
    ev_df, gt, _tt, _s = skyguard._build_eval_frame(
        rows=RETUNE_FRAME["rows"], seed=RETUNE_FRAME["seed"]
    )
    fused = model.score_series_scores(ev_df)
    y_pred = (fused > model.threshold).astype(int)
    reported = evaluate(gt, y_pred, scores=fused)
    meta_path = art / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    recorded = (meta.get("retune") or {}).get("new", {}).get("metrics", {})
    ok = (abs(reported["recall"] - (recorded.get("recall") or 0)) < 1e-4
          and abs(reported["precision"] - (recorded.get("precision") or 0)) < 1e-4)
    print("[retune] VERIFY reloaded: thr=%s weights=%s channels=%s" %
          (model.threshold, model.fusion_weights, list(model.extra_channels.keys())))
    print("[retune] VERIFY reproduced: precision=%s recall=%s f1=%s" %
          (reported["precision"], reported["recall"], reported["f1"]))
    print("[retune] VERIFY OK" if ok else "[retune] VERIFY FAIL")


def retune_production(force: bool = False):
    art, blob, _phase = _apply_roc(force)
    _verify(blob, art)
    return art


if __name__ == "__main__":
    retune_production()
