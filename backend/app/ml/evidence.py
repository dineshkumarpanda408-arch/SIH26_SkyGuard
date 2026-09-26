"""Structured anomaly evidence / provenance builder.

Every item in the returned list is grounded in a signal that the pipeline
ACTUALLY computed for the record: the fused model score, per-component channel
scores, deterministic rule flags, the dew-point/thermodynamic consistency of
the joint T/RH pair, and the spatial (weather-vs-sensor) assessment.

The builder NEVER invents reasons from the anomaly type label and NEVER
invents numeric scores: a reason is only reported when its underlying computed
signal is available. Items carry an explicit `computed_by` provenance tag and
a `kind` (model / rule / heuristic) so the UI can never mistake a physics/rule
signal for ML output.
"""

# Mapping from component key (as produced by ModelManager.analyze_row /
# score_series_scores) to a display label + provenance tag.
COMPONENT_LABELS = {
    "baseline": ("Baseline ML detector (raw temperature/pressure/humidity)", "ml:baseline_isolation_forest"),
    "temporal": ("Temporal ML detector (rolling-window context)", "ml:temporal_isolation_forest"),
    "multivariate": ("Multivariate ML detector (dew-point/vapor/pressure co-movement)", "ml:multivariate_isolation_forest"),
    "seasonal": ("Diurnal ML detector (hour-of-day context)", "ml:seasonal_isolation_forest"),
    "drift": ("Drift channel", "ml:drift_detector"),
    "flatline": ("Flatline (stuck-sensor) channel", "ml:flatline_detector"),
    "missingness": ("Missingness/communication channel", "ml:missingness_detector"),
    "roc": ("Rate-of-change channel", "ml:roc_detector"),
}

# Channel score above which a component is reported as elevated. Components
# below this are nominal behavior, not evidence of an anomaly.
COMPONENT_FLOOR = 0.5


def build_evidence(*, score, threshold, component_scores, rule_flags, spatial,
                   thermo, model_version, calibration_status="uncalibrated"):
    """Assemble the structured evidence list for one detection record.

    Parameters
    ----------
    score : float fused model score (uncalibrated).
    threshold : model detection threshold.
    component_scores : dict of component/channel scores from analyze_row.
    rule_flags : dict of deterministic rule flags (missing_imputed / frozen_plateau).
    spatial : dict with keys `assessment`, `is_weather_event`,
        `sensor_fault_likelihood`, `evidence_lines` (list of strings).
    thermo : dict from ml.thermo.thermodynamic_evidence.
    model_version : production model version string.

    Returns a list of evidence dicts; empty list only if nothing is available.
    """
    items = []

    # 1) ML fusion verdict (always present: the record only exists because of it)
    items.append({
        "signal": "ml",
        "label": "ML anomaly signal",
        "status": "flagged",
        "detail": (
            f"Fused model score {score:.3f} exceeds the {threshold:.2f} detection "
            "threshold. The fused score (uncalibrated, not a probability) is a "
            "weighted combination of the ML detectors and dedicated "
            "drift/flatline/missingness/rate-of-change channels."
        ),
        "computed_by": f"ml:{model_version}",
        "kind": "model",
    })

    # 2) Elevated component channels (only when actually elevated)
    for key, (label, cby) in COMPONENT_LABELS.items():
        val = component_scores.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if val < COMPONENT_FLOOR:
            continue
        items.append({
            "signal": key,
            "label": label,
            "status": "elevated",
            "detail": (
                f"Channel score {val:.3f} (elevated; uncalibrated 0-1 scale), "
                "contributing model-side evidence for this reading."
            ),
            "computed_by": cby,
            "kind": "model",
        })

    # 3) Deterministic rule flags (structural evidence computed by rules)
    if rule_flags.get("missing_imputed"):
        items.append({
            "signal": "missingness",
            "label": "Missing/imputed value",
            "status": "flagged",
            "detail": (
                "A raw temperature/pressure/humidity value was absent and imputed "
                "before scoring; the missingness evidence was NOT fed into the ML "
                "score."
            ),
            "computed_by": "rule:missing_value_imputed",
            "kind": "rule",
        })
    if rule_flags.get("frozen_plateau"):
        items.append({
            "signal": "flatline",
            "label": "Stuck-sensor plateau",
            "status": "flagged",
            "detail": (
                "The last readings of a variable were constant (flat plateau), "
                "the structural signature of a frozen sensor."
            ),
            "computed_by": "rule:flat_plateau",
            "kind": "rule",
        })

    # 4) Dew-point / thermodynamic consistency evidence (TASK 1 signal)
    if thermo:
        items.append({
            "signal": "thermodynamic",
            "label": "Dew-point / thermodynamic consistency",
            "status": str(thermo.get("status", "UNAVAILABLE")).lower(),
            "detail": thermo.get("detail", ""),
            "computed_by": thermo.get("computed_by", "physics:magnus_dew_point"),
            "kind": "heuristic",
        })

    # 5) Spatial weather-vs-sensor assessment
    spatial_item = _spatial_item(spatial)
    if spatial_item:
        items.append(spatial_item)

    # 6) Provenance: calibration honesty (uncalibrated fused score)
    items.append({
        "signal": "calibration",
        "label": "Score calibration status",
        "status": "uncalibrated",
        "detail": (
            "Fused scores are uncalibrated: they are anomaly-confidence scores, "
            "not probabilities, consistent with the model's calibration_status."
        ),
        "computed_by": f"ml:{model_version}",
        "kind": "system",
    })

    return items


def _spatial_item(spatial):
    if not spatial:
        return None
    assessment = spatial.get("assessment")
    if assessment == "UNAVAILABLE":
        return {
            "signal": "spatial",
            "label": "Spatial (weather vs sensor) assessment",
            "status": "unavailable",
            "detail": "Spatial evidence unavailable: no neighboring station data at this time.",
            "computed_by": "statistical-heuristic:spatial_consistency",
            "kind": "heuristic",
        }
    is_weather = spatial.get("is_weather_event", False)
    likelihood = spatial.get("sensor_fault_likelihood")
    lines = spatial.get("evidence_lines") or []
    detail = lines[0] if lines else (assessment or "No spatial detail available.")
    detail = f"{detail} (sensor-fault likelihood heuristic = {likelihood:.2f})." \
        if likelihood is not None else detail
    return {
        "signal": "spatial",
        "label": "Spatial (weather vs sensor) assessment",
        "status": "weather" if is_weather else "sensor_fault",
        "detail": detail,
        "computed_by": "statistical-heuristic:spatial_consistency",
        "kind": "heuristic",
    }


def fallback_evidence(*, score, threshold, feature, event_assessment,
                      is_weather_event, sensor_fault_likelihood, model_version):
    """Minimal evidence list for legacy persisted records that predate the
    structured evidence capture, built ONLY from fields actually stored on the
    row (score, spatial assessment, classification provenance). It never
    fabricates component scores that were not persisted."""
    items = [
        {
            "signal": "ml",
            "label": "ML anomaly signal",
            "status": "flagged",
            "detail": (
                f"Fused model score {score:.3f} exceeded the {threshold:.2f} "
                "detection threshold. Record persisted before structured "
                "evidence/component scores were stored, so only the fused "
                "verdict is available."
            ),
            "computed_by": f"ml:{model_version}",
            "kind": "model",
        },
        {
            "signal": "classification",
            "label": "Anomaly classification",
            "status": "flagged",
            "detail": (
                "Root-cause type was produced by the deterministic classifier on "
                "the recovered series (not by ML); component-score evidence from "
                "that run was not persisted for this legacy record."
            ),
            "computed_by": "rule/classifier (not ML)",
            "kind": "rule",
        },
    ]
    if event_assessment:
        items.append({
            "signal": "spatial",
            "label": "Spatial (weather vs sensor) assessment",
            "status": "weather" if is_weather_event else "sensor_fault",
            "detail": f"{event_assessment}. Sensor-fault likelihood heuristic = "
                      f"{sensor_fault_likelihood:.2f}." if sensor_fault_likelihood is not None
                       else str(event_assessment),
            "computed_by": "statistical-heuristic:spatial_consistency",
            "kind": "heuristic",
        })
    items.append({
        "signal": "feature",
        "label": "Affected variable",
        "status": "flagged",
        "detail": f"The flagged variable is {feature or 'unknown'}.",
        "computed_by": "rule:zscore(missing/frozen)",
        "kind": "rule",
    })
    return items