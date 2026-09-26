"""Model evaluation on ground-truth anomaly-injected data.

Computes precision, recall, F1, false-positive/negative rates, accuracy, ROC-AUC
(when a valid continuous score vector with both classes is available), a binary
confusion matrix, and a multi-class anomaly-diagnosis confusion matrix.
Only measured values are produced (never fabricated).
"""

import numpy as np

CATEGORIES = ["SPIKE", "DROP", "DRIFT", "FROZEN_SENSOR", "MISSING_DATA"]


def _roc_auc(y_true, scores):
    """ROC-AUC from continuous anomaly scores vs labels.

    Requires at least one positive and one negative observed class, otherwise it
    is undefined and we return None (callers must not fabricate a number).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        scores = np.where(np.isfinite(scores), scores, 0.0)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # rank-based AUC (Mann-Whitney U) robust to any scoring granularity;
    # avoids importing scipy and matches sklearn.rocr_auc_score semantics.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for tied scores
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_ranks = np.zeros(len(scores), dtype=float)
    for r in range(len(counts)):
        mask = inv == r
        avg_ranks[mask] = np.mean(ranks[mask])
    sum_pos = float(np.sum(avg_ranks[y_true == 1]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 4)


def _pr_auc(y_true, scores):
    """Precision-recall AUC (average precision) from continuous scores.

    Returns None when either class is absent or scores are not usable, matching
    ROC-AUC honesty rules. Kept metric-namespace-local (same style as _roc_auc).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        scores = np.where(np.isfinite(scores), scores, 0.0)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y_sort = y_true[order]
    tp = 0
    precisions = []
    n_total = len(y_sort)
    for i in range(n_total):
        if y_sort[i] == 1:
            tp += 1
        precisions.append(tp / (i + 1))
    ap = float(np.sum(np.asarray(precisions)[y_sort == 1])) / n_pos
    return round(ap, 4)


def evaluate(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = len(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    auc = _roc_auc(y_true, scores) if scores is not None else None
    pr_auc = _pr_auc(y_true, scores) if scores is not None else None

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "grid": [[tn, fp], [fn, tp]],
        },
    }


def multiclass_confusion_matrix(true_types, pred_types, categories=None):
    """Multi-class diagnosis confusion matrix (true type x predicted type).

    Only anomalous readings that carry a non-None predicted diagnosis are used.
    Returns a dict with row/column labels and the raw integer grid.
    """
    cats = categories if categories is not None else CATEGORIES
    labels = cats + ["NORMAL"]
    grid = {t: {p: 0 for p in labels} for t in labels}

    for t, p in zip(true_types, pred_types):
        t_key = t if t in grid else "NORMAL"
        p_key = p if p in grid else "NORMAL"
        grid[t_key][p_key] += 1

    return {
        "labels": labels,
        "grid": grid,
        "matrix": [[grid[t][p] for p in labels] for t in labels],
    }


def evaluate_by_type(results):
    """results: list of {'true_type', 'y_true', 'y_pred'} per injected group."""
    by_type = {}
    for r in results:
        t = r["true_type"]
        m = evaluate([r["y_true"]], [r["y_pred"]])
        by_type.setdefault(t, []).append(m)
    out = {}
    for t, metrics in by_type.items():
        avg = {
            "precision": round(np.mean([m["precision"] for m in metrics]), 3),
            "recall": round(np.mean([m["recall"] for m in metrics]), 3),
            "f1": round(np.mean([m["f1"] for m in metrics]), 3),
            "count": len(metrics),
        }
        out[t] = avg
    return out
