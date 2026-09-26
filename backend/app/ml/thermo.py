"""Dew-point / thermodynamic consistency evidence.

A pure, deterministic physics check on the joint (temperature, humidity) pair
using the Magnus/Tetens formulation — the SAME constants used by the
engineered feature set in `app.features.add_multivariate_features`
(a=17.27, b=237.7, gamma formulation), so this signal never disagrees with the
features the model was trained on.

This module is EVIDENCE ONLY: it does NOT retrain anything, does NOT change the
fused model score, does NOT override the production detector, and never
independently classifies a reading. It returns an explanatory signal that the
anomaly/evaluation UI can surface alongside the ML verdict, exactly as
required for the dew-point/thermodynamic-consistency evidence task.

Rules (conservative on purpose):
  * missing/NaN temperature or humidity      -> UNAVAILABLE (no misleading claim)
  * humidity outside the computable domain   -> INCONSISTENT
    (RH < 0 or RH > 100; RH = 0% is treated as unusable for this check because
    the dew-point calculation is undefined at zero RH)
  * computed dew point above temperature      -> INCONSISTENT (numerically
    impossible supersaturation for a standard station hygrometer)
  * otherwise                                 -> CONSISTENT (dew point is below the
    air temperature, which is the physically required ordering)
"""

import math

import numpy as np

# Magnus/Tetens constants, identical to app.features.add_multivariate_features
_MAGNUS_A = 17.27
_MAGNUS_B = 237.7

# Tolerance (deg C) above which a dew point >T is treated as supersaturated.
_DWPT_EPS_C = 0.05


def dew_point(t_c, rh):
    """Dew point (C) from temperature (C) and relative humidity (%), Magnus form.

    Returns None when the pair cannot be evaluated (missing/NaN input, or RH
    outside the computable domain (0, 100]).
    """
    if t_c is None or rh is None:
        return None
    try:
        t = float(t_c)
        r = float(rh)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(t) and np.isfinite(r)):
        return None
    if r <= 0.0 or r > 100.0:
        return None
    # gamma is the Magnus argument; Td < T for RH < 100 and Td == T at RH == 100.
    gamma = (_MAGNUS_A * t) / (_MAGNUS_B + t) + math.log(r / 100.0)
    denom = _MAGNUS_A - gamma
    if abs(denom) < 1e-12:
        return None
    td = (_MAGNUS_B * gamma) / denom
    if not math.isfinite(td):
        return None
    return td


def thermodynamic_evidence(t_c, rh):
    """Structured thermodynamic-consistency evidence for one (T, RH) pair.

    Returns a dict with:
      * available  : bool  - the computation produced a usable verdict
      * status     : "CONSISTENT" | "INCONSISTENT" | "UNAVAILABLE"
      * dew_point  : float | None
      * detail     : human-readable explanation
      * computed_by: provenance tag ("physics:magnus_dew_point")
      * heuristic  : True (deterministic physics check, not a trained model)
    """
    missing_msg = (
        "Dew-point evidence unavailable: temperature and/or humidity is missing "
        "for this reading; no thermodynamic claim is made."
    )

    if t_c is None or rh is None:
        return _unavailable(missing_msg)

    try:
        t = float(t_c)
        r = float(rh)
    except (TypeError, ValueError):
        return _unavailable(missing_msg)

    if not (np.isfinite(t) and np.isfinite(r)):
        return _unavailable(missing_msg)

    if r < 0.0 or r > 100.0:
        return {
            "available": True,
            "status": "INCONSISTENT",
            "dew_point": None,
            "detail": (
                f"Relative humidity {r:.1f}% is outside the physical domain "
                "[0, 100]% — a physically impossible humidity measurement, "
                "pointing to a sensor/data fault rather than genuine weather."
            ),
            "computed_by": "physics:magnus_dew_point",
            "heuristic": True,
        }

    if r == 0.0:
        return {
            "available": True,
            "status": "INCONSISTENT",
            "dew_point": None,
            "detail": (
                "Relative humidity is 0.0% — treated as unusable for this "
                "dew-point consistency check because the dew point is undefined "
                "at zero RH."
            ),
            "computed_by": "physics:magnus_dew_point",
            "heuristic": True,
        }

    td = dew_point(t, r)
    if td is None:
        return _unavailable(missing_msg)

    if td > t + _DWPT_EPS_C:
        return {
            "available": True,
            "status": "INCONSISTENT",
            "dew_point": round(td, 1),
            "detail": (
                f"Computed dew point {td:.1f}C exceeds air temperature {t:.1f}C "
                "— numerically supersaturated, which a standard weather station "
                "cannot physically produce; the T/RH pair is thermodynamically "
                "inconsistent."
            ),
            "computed_by": "physics:magnus_dew_point",
            "heuristic": True,
        }

    return {
        "available": True,
        "status": "CONSISTENT",
        "dew_point": round(td, 1),
        "detail": (
            f"T={t:.1f}C, RH={r:.0f}% implies dew point {td:.1f}C (Magnus/Tetens); "
            "the joint T/RH pair is thermodynamically consistent."
        ),
        "computed_by": "physics:magnus_dew_point",
        "heuristic": True,
    }


def _unavailable(detail):
    return {
        "available": False,
        "status": "UNAVAILABLE",
        "dew_point": None,
        "detail": detail,
        "computed_by": "physics:magnus_dew_point",
        "heuristic": True,
    }