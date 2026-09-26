"""Phase 3 — strict, read-only evaluation. No tuning, no writes to models/data.

Runs:
  * seed 7 (unchanged benchmark, n=1300) for v1 / v2 / v3, each scored at its
    own persisted artifact threshold.
  * seed 19 (unseen frame, same generator/framework) for v2 vs v3.

Reports precision, recall, F1, ROC-AUC, PR-AUC, per-category recall (SPIKE,
DROP, DRIFT, FROZEN_SENSOR, MISSING_DATA) and the confusion matrix.

The v3 configuration is FROZEN (threshold 0.30, weights ml=0.30 drift=0.10
flat=0.30 miss=0.30). Nothing here changes models, thresholds, weights, the
benchmark generator, or MODEL_SUBDIR.
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np

from app.services import app as skyguard  # only for the (unchanged) eval-frame builder
from app.ml.training import load_versioned, MODEL_SUBDIR, V3_SUBDIR
from app.ml.evaluation import evaluate

EVAL_TYPES = ("SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA")


def _version_keys(*keys):
    """Deduplicated version list (production may now be v3)."""
    out = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def per_category_recall(ev_df, gt, y_pred):
    it = ev_df["injected_type"].to_numpy(dtype=object)
    out = {}
    for typ in EVAL_TYPES:
        mask = (it == typ) & (gt == 1)
        total = int(mask.sum())
        out[typ] = round(int((mask & (y_pred == 1)).sum()) / total, 4) if total else None
    return out


def score_version(subdir, ev_df, gt):
    model, meta = load_versioned(subdir)
    if model is None:
        raise RuntimeError(f"artifact missing: {subdir}")
    thr = float(getattr(model, "threshold", 0.72))
    scores = np.asarray(model.score_series_scores(ev_df), dtype=float)
    y_pred = (scores > thr).astype(int)
    m = evaluate(gt, y_pred, scores=scores)
    return {
        "artifact": subdir,
        "model_name": (meta or {}).get("model_name", subdir),
        "threshold_used": thr,
        "scores_shape": list(scores.shape),
        **{k: m[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc",
                             "tp", "fp", "fn", "tn", "confusion_matrix")},
        "per_category_recall": per_category_recall(ev_df, gt, y_pred),
    }


def run_frame(seed, versions, label):
    ev_df, gt, _tt, _summary = skyguard._build_eval_frame(rows=1300, seed=seed)
    n = int(len(gt))
    results = [score_version(subdir, ev_df, gt) for subdir in versions]
    payload = {
        "frame": label,
        "seed": seed,
        "n": n,
        "anomalies": int(gt.sum()),
        "anomaly_rate": round(float(gt.mean()), 4),
        "versions": results,
    }
    return payload


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity: v3 frozen config (read-only check of the persisted artifact)
    v3_model, v3_meta = load_versioned(V3_SUBDIR)
    print("[phase3] v3 artifact checkout:")
    print(f"  threshold={v3_model.threshold}  weights={v3_model.fusion_weights}")

    seed7 = run_frame(7, _version_keys("skyguard_v1", "skyguard_v2", MODEL_SUBDIR, V3_SUBDIR), "benchmark_seed7")
    with open(out_dir / "phase3_seed7_versions.json", "w", encoding="utf-8") as f:
        json.dump(seed7, f, indent=2, default=str)

    seed19 = run_frame(19, _version_keys("skyguard_v2", MODEL_SUBDIR, V3_SUBDIR), "unseen_seed19")
    with open(out_dir / "phase3_seed19_v2v3.json", "w", encoding="utf-8") as f:
        json.dump(seed19, f, indent=2, default=str)

    print()
    for payload in (seed7, seed19):
        print(f"=== {payload['frame']}  seed={payload['seed']}  n={payload['n']}  "
              f"anomalies={payload['anomalies']}  rate={payload['anomaly_rate']} ===")
        hdr = f"{'model':12s} {'thr':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'ROC':>7s} {'PR':>7s} | "
        hdr += " ".join(f"{t:>8s}" for t in EVAL_TYPES)
        print(hdr)
        print("-" * len(hdr))
        for v in payload["versions"]:
            line = f"{v['model_name'][:12]:12s} {v['threshold_used']:6.2f} {v['precision']:7.4f} " \
                   f"{v['recall']:7.4f} {v['f1']:7.4f} {v['roc_auc']:7.4f} {v['pr_auc']:7.4f} | "
            pc = v["per_category_recall"]
            line += " ".join(f"{(pc[t] if pc[t] is not None else float('nan')):8.4f}" for t in EVAL_TYPES)
            print(line)
        for v in payload["versions"]:
            print(f"  {v['model_name']} confusion grid [tn,fp;fn,tp]={v['confusion_matrix']['grid']}")


if __name__ == "__main__":
    main()