# ML Audit & Improvement Plan — SkyGuard AI

> **HISTORICAL / LEGACY record (pre-campaign audit).** This audit documents the
> **v2 baseline** as it was before the v3 campaign. The pipeline values it records
> (`ANOMALY_THRESHOLD = 0.72`, baseline metrics 0.9483 / 0.3819 / 0.5446 / 0.7506,
> and the §8 success targets including a precision ≥ 0.85 floor) describe the audited
> v2 baseline and the **campaign's** targets — they are not the current production
> configuration and the 0.85 floor is not the active acceptance criterion. The plan
> at the end has since been implemented: the v3 detection architecture exists and
> `skyguard-multivariate-v3` is now the current production model (see
> `docs/ML_DATA_PROVENANCE.md` §9).

Status: **Complete** (reproduction + root-cause, measured) — as of the pre-campaign
date. The plan ("Next: implementation of a v3 detection architecture on an expanded
v3 dataset") has since been implemented; v3 is now the current production artifact.

---

## 1. Pipeline inventory (all verified by reading code)

| Component | File | Notes |
|---|---|---|
| Data generator (synthetic v2) | `backend/data_generator.py` | 24 stations, 14 days, 15-min grid, 32,256 obs; station-distinct diurnal/regional; seeded & byte-reproducible |
| Station metadata | `backend/app/data_generator_meta.py` | STATION_META (id,name,lat,lon), STATION_ELEV |
| Ingestion / normalize | `backend/app/ingestion.py` | CSV → lowercase schema; `sample_weather.csv` (v2) |
| Preprocessing | `backend/app/preprocessing.py` | `validate` (bounds→NaN, dedupe), `impute` (**interpolate limit=3**), `missing_flags` |
| Feature engineering | `backend/app/features.py` | temporal (hour/day/month, change, rolling_mean/std/roc, volatility) + multivariate (vapor pressure sat, dew point, humidity_vs_vapor, mixing_ratio, delta_t/p) |
| Baseline detector | `backend/app/detector.py` | IsolationForest n=100, contamination 0.05, seed 42; z-scored; decision → sigmoid score; **median-fill of NaN** |
| Temporal detector | `backend/app/ml/temporal.py` | IsolationForest on temp cols |
| Multivariate detector | `backend/app/ml/multivariate.py` | IsolationForest on dew/mixing/vapor/deltas |
| Seasonal detector | `backend/app/ml/seasonal.py` | IsolationForest on cyclic hour-of-day + raw |
| Fusion | `backend/app/ml/pipeline.py` | `fused_last`/`score_series_scores`: 0.45·base + 0.20·temporal + 0.15·multivariate + 0.20·seasonal; **ANOMALY_THRESHOLD = 0.72**; `analyze_row` (missing/frozen flags, z-scored primary feature) |
| Type classifier (rule) | `backend/app/ml/classifier.py` | order: MISSING → DRIFT (monotonic) → SUDDEN_SHIFT → FROZEN (ptp<0.05·prev_std) → SPIKE/DROP (rel>4) → DRIFT fallback → NOISE |
| Injection (ground truth) | `backend/app/simulator.py` | SPIKE/DROP = 1 point; DRIFT monotonic; FROZEN constant; MISSING = NaN; labels + `injection_meta` |
| Training/reproduce | `backend/app/ml/training.py` | chronological 70/15/15 by timestamp quantile (never shuffled); persists `models/skyguard_v2/{model.pkl, metadata.json, feature_schema.json}`; sha256 of frame |
| Evaluation (detailed) | `backend/app/services.py` `_build_eval_frame` + `_score_eval` | 1,300-row hard frame, seed 7, varied magnitudes (SPIKE/DROP σ∈[4,9], DRIFT σ∈[5,11]); threshold 0.72 for binary + per-category recall |
| Measured v1 comparison | `services._measured_version_comparison` | same frame scored by persisted v1/v2 |
| Ablation (research) | `services._compute_model_comparison` | standalone analysers on same frame (NOT the fused production path) |
| Metrics helpers | `backend/app/ml/evaluation.py` | `evaluate`, `multiclass_confusion_matrix` |
| API | `routes/{stations,anomalies,analytics}.py` | sim persists rows; evaluation cached ttl=120 |

## 2. Baseline reproduction (measured, NOT re-stated)

Ran `reproduce_baseline.py` against the persisted `skyguard_v2` artifact on the
unchanged hard frame. Results:

```
precision = 0.9483 | recall = 0.3819 | f1 = 0.5446 | roc_auc = 0.7506 | n = 1300

PER-CATEGORY recall (injected -> detected) + fused-score stats on injected rows:
  SPIKE          15 ->  12  recall 0.800   mean_score=0.774 max=0.864
  DROP           15 ->  10  recall 0.667   mean_score=0.746 max=0.856
  DRIFT          48 ->  33  recall 0.688   mean_score=0.759 max=0.846
  FROZEN_SENSOR  36 ->   0  recall 0.000   mean_score=0.503 max=0.664
  MISSING_DATA   30 ->   0  recall 0.000   mean_score=0.434 max=0.514
NORMAL reference: mean 0.478, p95 0.618
```

Saved to `ml/evaluation/baseline_before_improvement.json`. Reproduces the exact
numbers shown on the Evaluation page — nothing was re-stated from memory.

## 3. Root-cause (measured, not guessed)

### 3.1 MISSING_DATA recall = 0.000
1. `simulator.inject(..., "MISSING_DATA")` writes NaN values but **keeps the
   timestamp grid** — there is no gap in timestamps to detect.
2. `preprocessing.impute()` interpolates NaN runs up to `limit=3` both directions.
   Measured: 10 injected humidity NaNs → **0 after impute** (all filled).
3. `detector.decision_scores` then replaces the few remaining NaNs with the column
   **median** (`X.fillna(df[self.features].median())`) before scoring.
4. Result: the ML never sees an anomaly; fused score ≈ 0.43 (below normal mean).

So missingness is erased before detection and no human/rule/timestamp signal
survives. This cannot be fixed by tuning the isolation forest.

### 3.2 FROZEN_SENSOR recall = 0.000
A constant plateau produces `change ≈ 0`, `roc ≈ 0`, `rolling_std ≈ 0`,
`volatility ≈ 0`. That pattern is not a clean outlier in the current isolation
forest feature space (max fused score 0.664 < 0.72). Detection of "a sensor that
stopped behaving like a sensor" is a **variance/persistence** question, not an
isolation-forest question. The rule classifier has a plateau test but per-category
recall counts the ML score crossing 0.72 first, so the rule never gets to fire.

### 3.3 DRIFT recall = 0.688
Drift rows average 0.759, barely above 0.72 — magnitudes vary (σ∈[5,11]) and the
trend is diluted across 8 points. Detection would improve with a residual-based
signal (expected-weather-adjusted) plus a persistence test (EWMA/CUSUM), rather
than relying on the forest seeing the raw level movement.

### 3.4 Precision is already strong (0.9483, FP=3)
Keep the fusion/feature base honest. Do not chase recall by lowering the global
threshold.

## 4. Design (the only real fix): mechanism-specific detectors + fusion

```
   RAW station stream (all 24)
      │
      ├── ML channel  → Isolation Forest fusion (base+temporal+multivariate+seasonal)
      ├── TS channel  → residual-based DRIFT (diurnal/seasonal-adjusted residual +
      │                EWMA / CUSUM / change-point persistence), SPIKE/DROP
      │                robust-z on the residual
      ├── Missingness → RAW (pre-impute) NaN runs, invalid values, timestamp
      │                gaps (> interval+tolerance), consecutive missing, per-variable
      └── Flatline    → rolling var/std, unique-count, zero-diff ratio, plateau
                        width (configurable persistence window)

   Evidence fusion (transparent weights; threshold on VALIDATION only)
        → score, type, severity, computed_by provenance, correction
```

- Missingness and flatline are **deterministic, statistically justified signals**,
  measured on the raw stream pre-imputation — exactly the "the signal stopped
  behaving like a sensor" reasoning judges can interrogate.
- They fuse like the existing channels; `computed_by` labels every contribution.
- v2's 4-channel scorer is untouched; v3 adds channels (+ output is additive).

## 5. Expanded dataset v3 (scientific, not duplicated)

Extend `backend/data_generator.py` to a longer, richer, seeded & reproducible v3:

- Span 14 → **60 days** (24 stations × 60 d × 96 = 138,240 obs), more synoptic
  regimes via additional low-frequency drivers.
- Realistic sensor behaviors generated as **separate persisted event tables** with
  explicit ground truth (`injection_meta`: type, start, duration, magnitude,
  variable, station) → drifts (slow +/−, piecewise, intermittent, with seasonal
  background), freezes (short/medium/long, tiny noise, flatline-after-jump),
  missing bursts / single-variable outages / communication dropouts, multi-variable
  weather events (station-correlated via regional field).
- Same schema so ingestion/training/eval wiring is unchanged. `synthetic=true`
  preserved; dataset v3 sha256 + meta file written.
- Chronological TRAIN/VAL/TEST (70/15/15) on v3; all preprocessing/feature stats
  fitted on TRAIN only.

## 6. Threshold & hyperparameters — validation only

- Grid search weights + threshold on the VALIDATION split; report
  threshold/precision/recall/F1/PR-AUC/confusion for the chosen point.
- Isolation-forest hyperparameters searched legitimately (n_estimators,
  max_samples, max_features, contamination, seeds) on validation; every run
  recorded under `ml/experiments/{experiment_id}/`.

## 7. Evaluation — unchanged benchmark + honest additions

- v3 scored on the **same hard frame**, same threshold policy, so v3-vs-v2-vs-v1
  comparison stays apples-to-apples.
- Also PR-AUC (missing today) + full multi-class confusion.
- One additional unseen frame later (different seed) as robustness evidence.

## 8. Success targets (honest, not guaranteed)

MISSING ≥ 0.85 · FROZEN ≥ 0.60 · DRIFT ≥ 0.78 · precision ≥ 0.85 · F1 ≥ 0.65 ·
ROC-AUC ≥ 0.80. All trade-offs reported raw. If a target is missed, it is reported.

## 9. File touch-plan (for approval)

| File | Change |
|---|---|
| `backend/data_generator.py` | v3 generator: 60 days, regimes, realistic sensor-events persistence (new module `backend/app/data_events_v3.py` optional) |
| `backend/app/ml/missingness.py` | NEW: raw missingness detector (NaN runs, gaps, per-variable) |
| `backend/app/ml/flatline.py` | NEW: frozen/plateau detector (variance, persistence) |
| `backend/app/ml/drift.py` | NEW: residual + EWMA/CUSUM drift detector |
| `backend/app/features.py` | residual-based + flatline + missingness causal features |
| `backend/app/ml/pipeline.py` | v3 fusion channels + threshold metadata (v2 path untouched) |
| `backend/app/ml/training.py` | `skyguard_v3` persistence/metadata/split provenance |
| `backend/app/services.py` | v3-aware scoring + evaluation reporting (unchanged frame), PR-AUC |
| `backend/app/routes/analytics.py` | additive v3 metadata/eval endpoints (backward compatible) |
| `ml/experiments/` | experiment records |
| `docs/ML_Performance_Improvement_Report.md`, `docs/ML_DATA_PROVENANCE.md` | final reports |
| Frontend `Evaluation.tsx` | add Production-v3 row & PR-AUC (labels only, same honest structure) |

## 10. Verification sequence (at the end)

train-from-scratch → save → load → new benchmark → unchanged benchmark → 3 demo
scenarios (SPIKE AWS-023, humidity DRIFT, cross-station SPIKE) still detect →
attribution/correction/provenance intact → backend tests (29, must stay green) →
frontend tests (3) + build → numbers written to the report.