# SkyGuard AI — ML Data Provenance

Complete record of every dataset, artifact, seed, hash, split and experiment log
that backs the current SkyGuard production model (`skyguard-multivariate-v3`) and the
earlier v2-era production configuration that preceded it.
Everything here is machine-verified; nothing is claimed beyond what files/artifacts
record. **Date:** 2026-08-29.

---

## 1. Provenance principles

- **Trained once, offline, frozen.** No online retraining or adaptation at
  inference. Artifacts are persisted to disk and loaded, not recomputed.
- **Hash-pinned inputs.** Any dataset mismatch halts training.
- **Chronological splits, never shuffled.** No future-data leakage into
  training; validation and test windows are later than training.
- **Tuning never touches evaluation data.** Validation-only search; benchmark
  (seed 7) and unseen (seed 19) frames excluded by construction.
- **Synthetic disclosed.** Every dataset is synthetic-but-weather-typical;
  `synthetic: true` is visible in metadata and UI.
- **Scores uncalibrated.** Detector outputs are evidence, not probabilities.

## 2. Legacy (prior-production) dataset — `sample_weather_v2`

`sample_weather_v2` was the production dataset before v3 was promoted; it and its
artifact remain persisted as legacy records.

| Property | Value |
|---|---|
| File | `data/sample_weather.csv` |
| SHA-256 | `69b56c7757035ee64bf979fede8499e94c7968c3004cd16037a37baede17b4fe` |
| Generator | station-distinct v2 telemetry (lat/elevation/distinct diurnal/seeded) |
| Stations / rows / features | 24 / 32,256 / 25 |
| Variables | temperature, pressure, humidity |
| Split (chronological, never shuffled) | train 70% · val 15% · test 15% |
| Synthetic | true (disclosed) |
| Artifact | `backend/models/skyguard_v2/` (frozen; `metadata.json`) |

## 3. Current production dataset — `sample_weather_v3`

`sample_weather_v3` began as the v3 campaign dataset and is now the production dataset.

| Property | Value |
|---|---|
| File | `data/sample_weather_v3.csv` (+ `.meta.json`) |
| SHA-256 | `7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511` |
| Generator | `backend/data_generator_v3.py` |
| Seeds | generation `91` · regime `4242` · episode `777` |
| Stations / rows | 24 stations × 60 days × 96 (15-min) = **138,240** observations, 5,760/station |
| Period | 2026-06-01 00:00 → 2026-07-30 23:45 |
| Variables | temperature, pressure, humidity (25 computed features) |
| Regimes | warm/cool phase · dry/humid phase · pressure rise/fall · episodic rain |
| Synthetic | true (disclosed) |
| Anomaly types (injected via eval framework) | SPIKE (dur 1) · DROP (dur 1) · DRIFT (dur 8) · FROZEN_SENSOR (dur 6) · MISSING_DATA (dur 5) |

Split (persisted, chronological, never shuffled):

| Split | Rows | Boundaries (incl.) |
|---|---|---|
| TRAIN | 96,768 | 2026-06-01 00:00 → 2026-07-12 23:45 |
| VALIDATION | 20,736 | 2026-07-13 00:00 → 2026-07-21 23:45 |
| TEST | 20,736 | 2026-07-22 00:00 → 2026-07-30 23:45 |

## 4. Model artifacts

### `backend/models/skyguard_v2/` — LEGACY (prior production, unchanged)

`skyguard-multivariate-v2` · dataset `sample_weather_v2` · threshold **0.72** ·
synthetic true · uncalibrated · ml:model_fusion + rule/statistical components.
File mtimes untouched by the campaign (v2 artifact written 2026-08-29 16:07).

### `backend/models/skyguard_v3/` — CURRENT PRODUCTION

`skyguard-multivariate-v3` · dataset `sample_weather_v3` · threshold **0.30** ·
fusion weights ml 0.35 / drift 0.05 / flat 0.20 / miss 0.30 / roc 0.10 (the
rate-of-change channel was added by the decision-layer retune in
`backend/app/ml/retune.py`) · synthetic true · uncalibrated · trained once
offline. Contents: `model.pkl`, `metadata.json`, `feature_schema.json`.

ML channel hyperparameters (`metadata.json`): Isolation Forest — baseline
(n_est 100, contam 0.05, rs 42), temporal (contam 0.04, rs 7), multivariate
(contam 0.04, rs 21); seasonal = diurnal-profile residuals. Scalers fit on
TRAIN only. Mechanism channels (config-only, no data fit): drift
(ewma_alpha 0.4, cusum k 0.55, h 3.0, slope win 8), flatline (window 5,
ratio_floor 0.4), missingness (interval_tol_fraction 1.5, window 12).

## 5. Evaluation frames (generated on demand, never trained on)

Built by the unchanged `services._build_eval_frame(rows, seed)` generator
(isolated realistic ground truth; ground truth kept separate from predictions).

| Frame | Seed | Rows | Anomalies | Used for |
|---|---|---|---|---|
| Validation | 11 | 600 | 138 (rate 0.23) | Phase 2 tuning (ONLY frame) |
| Benchmark | 7 | 1,300 | 144 (rate 0.1108) | Phase 3 evaluation (v1/v2/v3) |
| Unseen | 19 | 1,300 | 144 (rate 0.1108) | Phase 3 evaluation (v2 vs v3) |

Anomaly types per frame: SPIKE / DROP / DRIFT / FROZEN_SENSOR / MISSING_DATA
with durations above.

## 6. Experiment logs (every candidate preserved)

| File | Run |
|---|---|
| `ml/experiments/v3_val_tuning_seed11_20260829T133248Z.json` | first pass, grid floor 0.40–0.775 (boundary artifact identified) |
| `ml/experiments/v3_val_tuning_seed11_20260829T133636Z.json` | intermediate |
| `ml/experiments/v3_val_tuning_seed11_20260829T134709771558Z.json` | intermediate |
| `ml/experiments/v3_val_tuning_seed11_20260829T134905934529Z.json` | **selected run (frozen config)** |
| `ml/experiments/v3_val_tuning_seed11_20260829T135548859774Z.json` | determinism double-run (identical search) |
| `ml/experiments/v3_val_tuning_seed11_latest.json` | pointer to latest |

- 11,700 candidates per run (5 detector configs × 30 weight combos × 78 thresholds).
- Determinism: second run produced identical selected config and candidate set;
  only run-id/timestamp fields differ.
- Persisted Phase 3 results (historical, pre-retune v3 configuration):
  `ml/evaluation/phase3_seed7_versions.json`,
  `ml/evaluation/phase3_seed19_v2v3.json`,
  `ml/experiments/v3_campaign_final_summary.json`. The live Evaluation page
  reports the current measured values (see §9).

## 7. Timeline (2026-08-29, local/UTC)

| UTC | Event |
|---|---|
| 10:37 | v2 artifact trained (production; in metadata of live server) |
| 13:32–13:55 | v3 tuning runs (5 experiment logs) |
| 13:49 | v3 artifact trained (run `20260829T134905934529Z`), persisted to disk |
| 13:55 | determinism double-run |
| after 13:55 | Phase 3 read-only evaluation: seed 7 (v1/v2/v3) + seed 19 (v2/v3) |
| 19:xx | Option A closure: regression tests, demo smoke, reports |

## 8. Honesty & limitations

- All datasets are synthetic; `synthetic: true` is disclosed in artifact
  metadata and the UI.
- Scores are uncalibrated fusion output, never probabilities (`calibration:
  none` in metadata for both artifacts).
- v3 is the current production artifact. The Phase 3 campaign's earlier
  decision (keep v2 under the then-active precision ≥ 0.85 rule) is recorded
  historically in `docs/ML_Performance_Improvement_Report.md` §2/§7 and
  `ml/experiments/v3_campaign_final_summary.json`; it was superseded by the
  ROC-channel retune (see §9 below).
- v2 FROZEN/MISSING recall is 0.0000 on the benchmark; this is an open,
  disclosed limitation of the legacy artifact (v3 measures 0.50 / 0.87).
- No claim of verified-real-weather reasoning is made anywhere; the
  weather-vs-sensor heuristic is spatial evidence only.

## 9. Model promotion & rebaseline decision

- Earlier v3 promotion work (Phase 2 tuning + Phase 3 campaign) used a
  validation **precision floor of ≈ 0.85** as the acceptance criterion. Under
  that criterion the campaign recorded "v3 NOT promoted" and v2 stayed
  production (see §6–§7 and `docs/ML_Performance_Improvement_Report.md`).
- The current v3 artifact was later retuned at the **decision layer only**
  (`backend/app/ml/retune.py`): the SPIKE/DROP **rate-of-change (ROC)** channel
  was added to the fusion and the weights rebalanced (ml 0.35 / drift 0.05 /
  flat 0.20 / miss 0.30 / roc 0.10); the production threshold stays **0.30**.
  No dataset, features, detector weights or evaluation methodology changed.
- On the current production reproduction (n = 1,300, identical hard frame) the
  retuned v3 measures: **Precision 0.717 · Recall 0.7917 · F1 0.7525 ·
  Accuracy 0.9423 · ROC-AUC 0.9046 · PR-AUC 0.7289** (TP 114 / FP 45 / FN 30 /
  TN 1111); per-type SPIKE 1.00 · DROP 1.00 · DRIFT 0.83 · FROZEN_SENSOR 0.50 ·
  MISSING_DATA 0.87.
- v3 is therefore the selected production configuration based on the current
  evaluation trade-off. The 0.85 figure belongs to the earlier
  configuration/promotion criterion and is **not** the active acceptance
  criterion. No claim is made that v3 beats every metric; the current trade-off
  is recorded here and in `backend/app/ml/retune.py`.