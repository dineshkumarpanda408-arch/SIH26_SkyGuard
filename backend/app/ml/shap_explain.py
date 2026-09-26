"""Explainability for the detection model (real SHAP).

We compute per-feature contributions with the canonical ``shap.TreeExplainer``
over the underlying sklearn ``IsolationForest`` (the baseline detector is
Isolation-Forest based). SHAP gives an additive, game-theoretically grounded
decomposition of the tree-based model's output.

Let ``f`` be the IsolationForest ``decision_function`` (higher = more normal).
TreExplainer produces ``shap_values(f)`` whose sum reconstructs ``f``. Because
an *anomaly* corresponds to a *low* decision, we report the SHAP values of the
negated decision so that a positive contribution signals "this feature pushed
the sample toward being flagged anomalous" (the same convention previously used
by the SHAP layer, so callers and the narrative stay unchanged).

The detector standardizes features before training (a ``StandardScaler``), so
each raw sample row is transformed through the same scaler before being passed
to the explainer to keep the attribution in the model's true input space.
"""

import numpy as np

import shap

_TREE_EXPLAINER = None


def _explainer(model):
    """Build (and cache) a TreeExplainer for the detector's raw IsolationForest."""
    global _TREE_EXPLAINER
    detector = getattr(model, "model", None)
    if detector is None:
        return None
    key = id(detector)
    if _TREE_EXPLAINER is None or getattr(_TREE_EXPLAINER, "_detector_id", None) != key:
        data = None
        bg = getattr(model, "background", None)
        scaler = getattr(model, "scaler", None)
        if bg is not None and len(bg) > 0:
            bg_arr = np.asarray(bg, dtype=float)
            if bg_arr.ndim == 2:
                # background is stored on the detector's RAW feature columns
                feats = getattr(model, "features", None)
                if feats is not None and len(feats) > 0:
                    k = len(feats)
                    bg_feats = bg_arr[:, :k]
                else:
                    bg_feats = bg_arr
                if scaler is not None:
                    bg_scaled = scaler.transform(bg_feats)
                else:
                    bg_scaled = bg_feats
                data = bg_scaled
        explainer = shap.TreeExplainer(detector, data=data)
        explainer._detector_id = key
        _TREE_EXPLAINER = explainer
    return _TREE_EXPLAINER


def explain(model, X_row, feature_cols, background=None, expected=None):
    """Compute per-feature SHAP contributions for one sample.

    Parameters
    ----------
    model      : detector with .model (raw IsolationForest) and .scaler (StandardScaler)
    X_row      : 1-D array of the sample's RAW (pre-standardization) feature values
    feature_cols : list of feature names matching X_row order
    background : (n, k) array of typical RAW values used as the EXPLAINER's
                 background/data for the path-dependent TreeExplainer
    expected   : optional precomputed per-feature expected values (ignored; SHAP
                 infers its own baseline from the trained trees)

    Returns list of SHAP contributions aligned with feature_cols.
    """
    X_row = np.asarray([float(x) for x in X_row], dtype=float)
    k = len(feature_cols)

    explainer = _explainer(model)
    detector = getattr(model, "model", None)
    if explainer is None or detector is None:
        return [0.0] * k

    # The detector standardizes raw features before scoring; reproduce that step
    # so the row is in the exact input space the IsolationForest was trained on.
    scaler = getattr(model, "scaler", None)
    X_model = np.asarray([X_row], dtype=float)
    if scaler is not None:
        X_model = scaler.transform(X_model)

    shap_values = explainer.shap_values(X_model)
    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim == 2:
        sv = sv[0]
    # negate: SHAP explains decision_function (higher = normal); an anomaly is a
    # low decision, so positive contribution -> drove the sample toward anomaly
    contribs = -sv[:k]

    return [float(c) for c in contribs]


def narrative(contributions, feature_cols, top_k=3):
    """Build a human-readable explanation from actual SHAP contributions."""
    pairs = sorted(zip(feature_cols, contributions), key=lambda p: -abs(p[1]))
    top = pairs[:top_k]
    if not top:
        return "No feature contributions available."

    parts = []
    for feat, c in top:
        if abs(c) < 0.005:
            continue
        if c > 0:
            parts.append(
                f"{feat} contributed strongly to the anomaly decision because its "
                f"value deviated substantially from learned normal behavior (SHAP contribution {c:+.2f})"
            )
        else:
            parts.append(
                f"{feat} pushed against the anomaly decision (SHAP contribution {c:+.2f})"
            )
    if not parts:
        return "The model flagged this reading as anomalous based on a combination of subtle feature deviations."
    return "The anomaly decision was driven primarily by: " + " ".join(parts) + "."
