"""Weather-event vs sensor-fault assessment using spatial consistency.

Principle: when several nearby stations deviate in a comparable way at the same
time, the cause is far more likely to be a genuine (spatially coherent) weather
event; when a single station deviates while its neighbours stay in their normal
range, it is far more likely to be a local sensor/data fault.

This is a DETERMINISTIC RULE-BASED SPATIAL HEURISTIC — NOT a trained model and
NOT a calibrated probability. The returned `sensor_fault_likelihood` is a
continuous heuristic in [0,1] that weights partial agreement (neighbours that are
elevated but below the agreement band) at half strength, so an isolated station
against wholly-normal neighbours scores near 1, a regionally-coherent event near
0, and ambiguous cases land in between. Every message reports the full counts:
N abnormal · N partial · N normal out of the total neighbours.
"""

import numpy as np

ASSESSMENTS = {
    "LIKELY_WEATHER_EVENT": 0,
    "LIKELY_SENSOR_FAULT": 1,
    "UNAVAILABLE": 2,
}

# A neighbour "supports a weather event" when its (relative) deviation is at
# least this fraction of the target station's deviation. This threshold is
# deliberately below 1.0 so a slightly weaker but still-clearly-elevated reading
# across many stations still counts as spatial agreement.
AGREEMENT_FRACTION = 0.5

# A neighbour is considered NORMAL ("healthy") when its relative deviation from
# its own expectation is below this band — i.e. within normal sensor noise. This
# also acts as an absolute noise floor so that tiny absolute target deviations
# (large-base variables) never cause ordinary neighbours to look "agreed".
# Neighbours above this band but below the agreement threshold are PARTIAL.
NOISE_FLOOR = 0.04

# Fraction of stations (target + agreeing neighbours + half of partial ones)
# above which we call it a weather event and below which a sensor fault, with a
# smooth rise in between (smoothstep).
WEATHER_HIGH = 0.60
WEATHER_LOW = 0.25

# Neighbours in the partial band are weighted at half strength: they are
# suggestive but not decisive.
PARTIAL_WEIGHT = 0.5


def _rel_dev(v, e):
    if v is None or e is None:
        return None
    denom = abs(e)
    if denom < 1e-9:
        denom = 1e-9
    return abs(v - e) / denom


def assess(anomaly_row, neighbors=None):
    """Return (assessment, sensor_fault_likelihood, evidence_list).

    anomaly_row: {'station_id', 'value', 'variable', 'expected_value'}
    neighbors  : list of dicts {'station_id', 'value', 'expected_value'} from
                 other stations at the same timestamp.

    sensor_fault_likelihood is a rule-based heuristic in [0,1]; callers must
    NOT present it as a calibrated probability.
    """
    evidence = []

    if not neighbors:
        return (
            "UNAVAILABLE",
            None,
            ["Spatial evidence unavailable: no neighboring station data at this time."],
        )

    anomaly_val = anomaly_row.get("value")
    expected = anomaly_row.get("expected_value")
    if anomaly_val is None or (isinstance(anomaly_val, float) and np.isnan(anomaly_val)):
        # no value -> sensor-side
        return (
            "LIKELY_SENSOR_FAULT",
            0.9,
            [
                "Anomalous record contains no valid value (missing/comm failure); "
                "spatial assessment heuristic=rule.",
            ],
        )

    # target deviation (relative to its own expectation)
    anomaly_dev = _rel_dev(anomaly_val, expected)
    if anomaly_dev is None or not np.isfinite(anomaly_dev):
        anomaly_dev = np.inf

    # classify every neighbour as abnormal (spatially agreeing), normal, or
    # partial (elevated but below the agreement band)
    neighbor_devs = []
    abnormal_ids = []
    normal_ids = []
    partial_ids = []
    for n in neighbors:
        rd = _rel_dev(n.get("value"), n.get("expected_value"))
        if rd is None:
            continue
        neighbor_devs.append(rd)
        # elevated if it deviates comparably to the target AND beyond sensor noise
        agreement_threshold = max(AGREEMENT_FRACTION * anomaly_dev, NOISE_FLOOR)
        if rd >= agreement_threshold:
            abnormal_ids.append(n.get("station_id"))
        elif rd < NOISE_FLOOR:
            normal_ids.append(n.get("station_id"))
        else:
            partial_ids.append(n.get("station_id"))

    n_neighbors = len(neighbor_devs)
    if n_neighbors == 0:
        return (
            "UNAVAILABLE",
            None,
            ["No valid neighbouring values at this timestamp for spatial assessment."],
        )

    n_abnormal = len(abnormal_ids)
    n_normal = len(normal_ids)
    n_partial = len(partial_ids)
    n_total = n_neighbors + 1  # include the anomalous target station

    # continuous spatial evidence: target + abnormal + half of partial
    share = (1.0 + n_abnormal + PARTIAL_WEIGHT * n_partial) / n_total

    # smooth weather likelihood: 0 below WEATHER_LOW, rises to 1 by WEATHER_HIGH
    if share <= WEATHER_LOW:
        weather_likelihood = 0.0
    elif share >= WEATHER_HIGH:
        weather_likelihood = 1.0
    else:
        t = (share - WEATHER_LOW) / (WEATHER_HIGH - WEATHER_LOW)
        weather_likelihood = t * t * (3 - 2 * t)  # smoothstep for continuity

    sensor_fault_likelihood = float(np.clip(1.0 - weather_likelihood, 0.0, 1.0))

    counts_line = (
        f"{n_abnormal} of {n_neighbors} neighbouring stations elevated "
        f"({n_abnormal} abnormal, {n_partial} partial, {n_normal} normal). "
    )
    heuristic_note = "Spatial assessment is a deterministic heuristic (computed_by=rule, heuristic=true), not a trained probability."

    if share >= WEATHER_HIGH:
        assessment = "LIKELY_WEATHER_EVENT"
        evidence.append(
            counts_line
            + f"{share * 100:.0f}% spatial agreement -> regionally coherent "
            "weather event affecting multiple stations, not a local sensor fault. "
            + heuristic_note
        )
    elif share <= WEATHER_LOW:
        assessment = "LIKELY_SENSOR_FAULT"
        evidence.append(
            counts_line
            + "station deviates while its neighbours remain in range -> likely "
            "local sensor/data fault, weather ruled out. "
            + heuristic_note
        )
    else:
        assessment = "LIKELY_SENSOR_FAULT"
        evidence.append(
            counts_line
            + "partial spatial agreement -> ambiguous, weighted toward sensor fault. "
            + heuristic_note
        )

    # always include the counts for the confidence/severity display
    evidence.append(
        "Spatial summary: "
        f"{n_abnormal} abnormal \u00b7 {n_partial} partial \u00b7 {n_normal} normal "
        f"of {n_neighbors} neighbours."
    )

    return assessment, sensor_fault_likelihood, evidence