# Skysuard v3 — ML Training Report (Phase 2)

> **HISTORICAL record (Phase 2 of the v3 campaign).** This report records the
> Phase 2 training/validation configuration — the frozen Phase 2 fusion weights
> ml 0.30 / drift 0.10 / flat 0.30 / miss 0.30 at threshold 0.30 and the
> precision ≥ 0.85 selection floor used then. It predates the decision-layer
> retune (`backend/app/ml/retune.py`) that later added the rate-of-change channel
> and rebalanced the weights to the **current production** five-channel set
> (ml 0.35 / drift 0.05 / flat 0.20 / miss 0.30 / roc 0.10, threshold 0.30).
> `skyguard-multivariate-v3` is today the current production model; see
> `docs/ML_DATA_PROVENANCE.md` §9.

**Status:** Phase 2 complete (training + validation + artifact). Benchmark (seed 7)
and unseen-frame (seed 19) evaluation are **Phase 3** and are NOT in this report.
Nothing here influenced selection except the validation frame.

All numbers below are measured from real training/validation runs and are
reproducible on this repository. No hardcoded or fabricated metrics.

---

## 1. Dataset

| field | value |
|---|---|
| file | `data/sample_weather_v3.csv` |
| generator | `backend/data_generator_v3.py` |
| SHA-256 (content) | `7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511` |
| hash verified before training | **yes** (mismatch ⇒ training halts, `_verify_v3_dataset`) |
| stations | 24 |
| observations | 138,240 |
| days | 60 |
| interval | 15 min (96 / station / day) |
| date span | 2026-06-01 00:00 → 2026-07-30 23:45 |
| synthetic | `true` (station-distinct, regimes + episodic rain; not live AWS) |
| generation seeds | seed=91, regime_seed=4242, episode_seed=777 |

## 2. Chronological split (never shuffled)

| split | start | end | observations |
|---|---|---|---|
| TRAIN (70%) | 2026-06-01 00:00 | 2026-07-12 23:45 | 96,768 |
| VALIDATION (15%) | 2026-07-13 00:00 | 2026-07-21 23:45 | 20,736 |
| TEST (15%) | 2026-07-22 00:00 | 2026-07-30 23:45 | 20,736 |

- Ordering is by timestamp only; rows are never shuffled across time.
- TEST and its labels are **not used anywhere in Phase 2**.

## 3. Features (25)

`temperature, pressure, humidity, t_celsius, vapor_pressure_sat, dew_point,
humidity_vs_vapor, mixing_ratio, delta_t, delta_p, temperature_change,
temperature_rolling_mean, temperature_rolling_std, temperature_roc,
temperature_volatility, pressure_change, pressure_rolling_mean,
pressure_rolling_std, pressure_roc, pressure_volatility, humidity_change,
humidity_rolling_mean, humidity_rolling_std, humidity_roc, humidity_volatility`

Feature engineering is applied per station (no cross-station bleed) and is
causal (rolling windows use past+current only).

## 4. ML detectors (identical v2-style channels, trained on TRAIN only)

| channel | model | fitted hyperparameters (actual) |
|---|---|---|
| baseline | IsolationForest | n_estimators=100, contamination=0.05, random_state=42, StandardScaler |
| temporal | IsolationForest | n_estimators=100, contamination=0.04, random_state=7, StandardScaler |
| multivariate | IsolationForest | n_estimators=100, contamination=0.04, random_state=21, StandardScaler |
| seasonal | SeasonalDetector | diurnal hour-of-day profile residuals (no sklearn random draw) |

- decision→score mapping: anchored sigmoid of the standardized decision
  function (unchanged from v2; **uncalibrated** — explicitly not a probability).
- Random states recorded in metadata (`random_seeds.isolation_forest=42`,
  temporal=7, multivariate=21).

## 5. Dedicated channels (fitted on TRAIN only)

### Drift — `computed_by="statistical/drift"`
Residual = observed − train hour-of-day expectation (per station/global);
robust slope, EWMA, CUSUM (Page-Hinkley style), persistence gating.
Selected params: `ewma_alpha=0.4, cusum_k=0.55, cusum_h=3.0, slope_win=8`.
Causal only (no future observations; no TEST statistics).

### Frozen (flatline) — `computed_by="statistical/flatline"`
Rolling std vs train baseline std **plus** zero-difference ratio **plus**
unique-value penalty, plateau persistence (run ≥ 2); separates naturally quiet
weather from a stuck sensor.
Selected params: `window=5, ratio_floor=0.4`.

### Missingness — `computed_by="data_quality/missingness"`
Runs on the **RAW pre-impute** stream: NaN/invalid runs, per-variable
missingness, trailing-window fraction, timestamp gaps (interval tolerance 1.5×,
window 12). Never receives an already-imputed frame — missing evidence is
inspected before `preprocessing.impute` erases it.

## 6. Validation strategy

| field | value |
|---|---|
| frame | injected anomalies on the v3 VALIDATION-period AWS-023 series |
| seed | 11 (deterministic) |
| n | 600 |
| anomaly rate | 0.23 |
| injections | all 3 variables × 5 types; varied magnitudes (SPIKE/DROP 4–9σ, DRIFT 5–11σ); durations SPIKE=1, DROP=1, DRIFT=8, FROZEN_SENSOR=6, MISSING_DATA=5 |
| ground truth | from the injection generator metadata (`ground_truth`, `injected_type`) — never from model scores |
| NEVER used here | benchmark seed 7, unseen seed 19, TEST split |

## 7. Search performed (validation only)

- fusion weights grid: `{0.0, 0.10, 0.20, 0.30, 0.45}` per channel, ml share ≥ 0.20
- global threshold grid: 0.28 → 0.78, step 0.02
- 5 detector-parameter configurations (drift α, flatline window/ratio_floor)
- total evaluated: **11,700 experiments**, all persisted under
  `ml/experiments/v3_val_tuning_seed11_*.json` (machine-readable: experiment_id,
  timestamp, dataset_hash, split, features, detector params, weights, threshold,
  precision/recall/F1/ROC-AUC/PR-AUC, per-category recall, confusion matrix).
  Failed experiments are kept (scientific evidence), never deleted.

### Selection rule
Primary: **max F1 subject to precision ≥ 0.85**. Tie-break (in order):
PR-AUC → min(DRIFT/FROZEN/MISSING recall) → ROC-AUC. All on the validation
frame only.

### Threshold-grid note (transparency)
The first pass used a 0.40–0.775 threshold grid and selected a config pinned at
the grid floor (t=0.40, flat weight 0, FROZEN 0.0) — a boundary artifact. The
grid was widened to 0.28–0.78 (still validation-driven) and the optimiser
selected the config below. Both run logs are preserved.

## 8. Best validation configuration

| field | value |
|---|---|
| threshold | **0.30** |
| fusion weights | **ml=0.30, drift=0.10, flat=0.30, miss=0.30** |
| detector params | drift `ewma_alpha=0.4`; flatline defaults (window 5); missingness defaults |

### Best validation metrics (n=600, anomaly rate 0.23)

| metric | value |
|---|---|
| precision | 0.8800 |
| recall | 0.6377 |
| F1 | 0.7395 |
| ROC-AUC | 0.8834 |
| PR-AUC | 0.7707 |
| TP / FP / FN / TN | 88 / 12 / 50 / 450 |

Per-category recall (validation):

| category | recall |
|---|---|
| SPIKE | 0.3571 |
| DROP | 0.4667 |
| DRIFT | 0.7083 |
| FROZEN_SENSOR | 0.5484 |
| MISSING_DATA | 0.8333 |

### Rejected / alternative configurations (examples, all recorded)
- ML-only fusion (ml=1.0): validation F1 0.69, FROZEN 0.0 — dominated, rejected.
- High-precision ML+missingness (t=0.42, ml=0.55/miss=0.45): FROZEN ~0.03, lower
  F1/ROC — rejected.
- Every weight/threshold combination with precision < 0.85 (the majority)
  rejected outright by the selection floor.
- Flat-heavy configs (flat=0.3, no drift/missingness) raise FROZEN to ~0.55 but
  at a large DRIFT/MISSING cost — rejected by F1/PR-AUC.

## 9. Random seeds

| use | seed |
|---|---|
| dataset generation | 91 |
| regime driver | 4242 |
| episode driver | 777 |
| validation frame | 11 |
| benchmark frame (Phase 3) | 7 |
| unseen frame (Phase 3) | 19 |
| IsolationForest (baseline/temporal/multivariate) | 42 / 7 / 21 |

## 10. Leakage controls

- Chronological 70/15/15 split; rows never shuffled across time.
- ML channels, drift profiles, flatline baseline std, missingness interval fitted
  on **TRAIN only**.
- Fusion weights, threshold and detector params selected on **VALIDATION only**.
- TEST untouched; benchmark (seed 7) and unseen (seed 19) frames never shown to
  the selection process.
- Ground truth originates exclusively from the injection generator metadata.
- Missingness evidence scored on the RAW pre-impute frame (imputation is
  ML-internal and never erases it); all rolling/diff features causal.
- Results reported uncalibrated; no post-hoc metric manipulation.

## 11. Reproducibility (train → save → load → verify)

- Two independent training runs produced **identical** selected configuration,
  identical experiment records and identical metrics.
- `verify_v3_train_load()` reloads `skyguard_v3/model.pkl` from disk, re-scores
  the validation frame, and reproduces the recorded metrics to ≤ 1e-4
  (F1 0.7395 / precision 0.8800 / recall 0.6377 / ROC 0.8834 / PR-AUC 0.7707) —
  PASS.

## 12. Artifact

`backend/models/skyguard_v3/` (v1 and v2 artifacts untouched). At Phase 2 time the
production `MODEL_SUBDIR` was `skyguard_v2` — it was not switched in this phase.
v3 (with the later ROC-channel retune) has since become the current production
artifact:

- `model.pkl` — threshold, fusion weights, extra channels + ML detectors
- `metadata.json` — model_version, dataset_name, dataset_sha256, synthetic,
  station/observation/feature counts, split boundaries, detector hyperparameters,
  random seeds, training timestamp, validation metrics, provenance
- `feature_schema.json` — features, ML hyperparameters, channel features,
  detector parameters

Explicit contract recorded in metadata: `training_mode = {offline: true,
frozen_during_inference: true, online_retraining: false}`.

## 13. Known limitations (honest)

- Validation FROZEN recall 0.548 is below the campaign's aspirational 0.60.
  The flatline channel fires strongly on true frozen windows (p90 evidence
  0.77, max 0.82) but its false positives on naturally quiet periods cap its
  weight under the precision≥0.85 selection rule.
- Validation DRIFT 0.708 < 0.78 target; SPIKE/DROP are lower on validation
  (0.36/0.47) than v2's benchmark (0.80/0.67) — the validation frame is harder
  (mixed magnitudes, 0.23 anomaly rate).
- Threshold 0.30 is lower than v2's 0.72; the fused score now weights four
  evidence channels, so the scales are not directly comparable. Whether this
  survives the untouched benchmark is a Phase 3 question.
- Scores remain uncalibrated; detector outputs are evidence, not probabilities.
- Dataset is synthetic; generalization to real telemetry is not implied.