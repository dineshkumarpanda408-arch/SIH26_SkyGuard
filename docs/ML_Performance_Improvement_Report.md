# SkyGuard AI — ML Performance Improvement Report (v3 campaign)

> **HISTORICAL / LEGACY record.** This report documents the **Phase 3 campaign
> verdict at that time**: under the then-active validation precision ≥ 0.85
> promotion criterion, v3 was recorded as "not promoted" and v2 stayed production.
> Those figures describe the **pre-retune** v3 configuration. The verdict has since
> been **superseded**: after the decision-layer retune (ROC channel —
> `backend/app/ml/retune.py`), `skyguard-multivariate-v3` is the **current
> production** model (threshold 0.30; five-channel weights ml 0.35 / drift 0.05 /
> flat 0.20 / miss 0.30 / roc 0.10). Current measured production metrics (n=1,300):
> precision 0.717 · recall 0.7917 · F1 0.7525 · ROC-AUC 0.9046 · PR-AUC 0.7289.
> The 0.85 floor belongs to the earlier configuration/criterion and is no longer the
> active acceptance criterion. See `docs/ML_DATA_PROVENANCE.md` §9.

**Campaign outcome (historical):** The v3 mechanism-detector campaign was implemented,
trained, validated and evaluated honestly. Per the pre-registered Phase 3 decision rule,
v3 did **not** earn the production switch at that time. **Production remained
`skyguard_v2`**; the v3 artifact was retained intact for future experimentation.

**Date:** 2026-08-29 · **Author:** SkyGuard engineering (campaign log)

---

## 1. Executive summary

SkyGuard v2 (production at campaign time, now legacy, `skyguard-multivariate-v2`) has near-zero recall on sensor
**FROZEN** and **MISSING** anomalies (0.0000 / 0.0000). The campaign built **v3**:
the v2 ML fusion plus dedicated statistical **drift / flatline / missingness** detector
channels, fused with learned weights. v3 was trained on an expanded synthetic dataset,
tuned on a **chronological validation frame only**, then evaluated **strictly** on the
unchanged seed-7 benchmark and an unseen seed-19 frame.

| | seed 7 (unchanged benchmark, n=1300) | seed 19 (unseen, n=1300) |
|---|---|---|
| | v2 (prod at campaign time · legacy) | v3 (pre-retune, experimental) | v2 (prod at campaign time · legacy) | v3 (pre-retune, experimental) |
| Precision | **0.9483** | 0.8485 | **0.9667** | 0.8384 |
| Recall | 0.3819 | **0.5833** | 0.4028 | **0.5764** |
| F1 | 0.5446 | **0.6914** | 0.5686 | **0.6831** |
| ROC-AUC | 0.7506 | **0.8994** | 0.7606 | **0.9002** |
| PR-AUC | 0.5839 | **0.7230** | 0.6067 | **0.7193** |
| SPIKE recall | **0.8000** | 0.3333 | **0.6667** | 0.1333 |
| DROP recall | **0.6667** | 0.4000 | **0.6667** | 0.4000 |
| DRIFT recall | 0.6875 | **0.7083** | **0.7708** | 0.6667 |
| FROZEN recall | 0.0000 | **0.5000** | 0.0278 | **0.5000** |
| MISSING recall | 0.0000 | **0.7000** | 0.0000 | **0.8333** |

**Decision (historical):** v3 delivered a genuine, generalized improvement in overall
recall/AUC and, decisively, in FROZEN/MISSING detection — **but at a
reproducible cost**: precision fell ~10–13 points and SPIKE/DROP recall
regressed on **both** frames. That was a real trade-off, not an unqualified win.
Per the campaign rule ("switch only on genuine generalized improvement"),
the campaign concluded v3 was **not** promoted at that time. This verdict was
superseded by the ROC-channel retune (`backend/app/ml/retune.py`), after which
`skyguard-multivariate-v3` became current production (see the banner above and
`docs/ML_DATA_PROVENANCE.md` §9).

## 2. Decision rule (pre-registered, exercised honestly)

1. **No tuning on the evaluation frames.** The benchmark (seed 7) and unseen
   (seed 19) frames were used **never** during Phase 2 tuning — only the
   chronological validation window (seed 11) was. Tuning was frozen before any
   Phase 3 evaluation ran.
2. **Threshold-per-artifact.** Each model is scored at its own persisted
   threshold (v1/v2 = 0.72; v3 = 0.30). No threshold was re-fitted to the
   benchmark.
3. **Control integrity.** v2 must reproduce the frozen baseline exactly
   (0.9483 / 0.3819 / 0.5446 / 0.7506). It did, bit-for-bit, on the
   unchanged benchmark generator.
4. **Switch condition (as recorded in Phase 3).** Promote v3 only if benchmark
   **and** unseen frames show genuine generalized improvement. v3's validation
   numbers were targets, not resumes — they were non-decisive by design.
   Under this condition the campaign recorded "no promotion"; a later
   decision-layer retune changed the configuration and the promotion outcome
   (see banner / §7).

## 3. v2 baseline (control; production at campaign time, now legacy)

- Artifact: `backend/models/skyguard_v2/` (frozen; mtimes unchanged during campaign).
- Dataset `sample_weather_v2`, sha256 `69b56c7757035ee6…`, 24 stations /
  32,256 observations / 25 features, synthetic, uncalibrated, trained-once/offline.
- Control metrics on seed 7 (n=1300): precision **0.9483**, recall **0.3819**,
  F1 **0.5446**, ROC-AUC **0.7506**, PR-AUC **0.5839**.
- Per-type: SPIKE 0.8000 · DROP 0.6667 · DRIFT 0.6875 · FROZEN **0.0000** ·
  MISSING **0.0000**.

## 4. Phase 1 — architecture (v2-identical, v3-additive)

Changes in `backend/app/ml/pipeline.py` (all v3-additive; v2 path unchanged):
- `ml_fused_scores(df)` — extracted the exact v2 fusion
  (0.45 · baseline + 0.20 · temporal + 0.15 · multivariate + 0.20 · seasonal).
- `score_series_scores(df, raw=None)` — identical v2 output when no extra
  channels are configured; v3 fuses `ml + drift + flatline + missingness`
  with configured weights, missingness scored on the **raw (pre-imputation)** frame.
- `analyze_row()` uses `self.threshold`; exposes `evidence` for
  ml/drift/flatline/missingness; existing fields, `computed_by`,
  attribution and correction logic untouched.
- Detector semantics are labeled honestly: drift → `statistical/drift`,
  flatline → `statistical/flatline`, missingness → `data_quality/missingness`.
  No probability or calibration claims.
- `save()` / `load_versioned()` persist and restore threshold, fusion weights
  and extra channels; v2 artifacts restore exactly.

**Identity check:** the v2 probe reproduced 0.9483 / 0.3819 / 0.5446 / 0.7506
with all deltas 0.0.

## 5. Phase 2 — v3 training and validation tuning

- Dataset `data/sample_weather_v3.csv` — 24 stations × 60 days × 96 (15-min) =
  **138,240 observations**, 25 features; sha256
  `7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511`
  (verified before training; mismatch halts).
  Generator `backend/data_generator_v3.py`, seeds 91 / regime 4242 / episode 777.
- **Chronological split, never shuffled** (boundaries persisted):
  TRAIN 96,768 (2026-06-01 → 07-12) · VAL 20,736 (07-13 → 07-21) ·
  TEST 20,736 (07-22 → 07-30).
- **ML channels** (Isolation Forest, fit on TRAIN only): baseline
  (n_est 100, contamination 0.05, rs 42), temporal (0.04, rs 7),
  multivariate (0.04, rs 21), seasonal (diurnal-profile residuals).
  StandardScalers fit on TRAIN only. All hyperparameters recorded
  (`backend/models/skyguard_v3/metadata.json`).
- **Mechanism channels** (no TRAIN data leakage; config-only): drift
  (CUSUM/EWMA, `ewma_alpha 0.4`, `k 0.55`, `h 3.0`, slope win 8), flatline
  (window 5, ratio floor 0.4), missingness (interval tol 1.5×, window 12).
- **Tuning**: chronological **validation-only** frame (seed 11, n=600,
  anomaly rate 0.23). Grid: 5 detector-parameter configs × weights in
  {0, 0.1, 0.2, 0.3, 0.45} × threshold 0.28–0.78 (step 0.02) = **11,700
  experiments**, each logged to `ml/experiments/`. Selection rule:
  **max F1 subject to precision ≥ 0.85**, tie-break PR-AUC then
  min(DRIFT/FROZEN/MISSING recall). Benchmark (seed 7) / unseen (seed 19) /
  TEST split never entered the search.
- **Best (frozen) configuration:** threshold **0.30**; weights ml **0.30** /
  drift **0.10** / flat **0.30** / miss **0.30**.
  Validation: F1 0.7395, precision 0.88, recall 0.6377, ROC-AUC 0.8834,
  PR-AUC 0.7707 (TP 88 / FP 12 / FN 50 / TN 450); per-type SPIKE 0.3571,
  DROP 0.4667, DRIFT 0.7083, FROZEN 0.5484, MISSING 0.8333.
- **Diagnosed threshold-floor artifact:** the first pass used 0.40–0.775 and
  stopped on the grid floor with flat weight = 0 and FROZEN recall 0.0; the
  grid was widened to 0.28 and re-run — the frozen config beats it
  (F1 0.7395 vs 0.6953).
- **Reproducibility:** `verify_v3_train_load()` reloads the artifact from disk
  and reproduces metrics ≤ 1e-4. Two independent training runs produced
  identical selected config and identical 11,700-experiment logs; only
  run-id/timestamp differ.
- Training details: `docs/ML_V3_TRAINING_REPORT.md`.

## 6. Phase 3 — strict evaluation (read-only)

Runner: `ml/evaluation/run_phase3.py` — evaluation only; no tuning; no writes
to models/data; `services.py` untouched; `MODEL_SUBDIR` untouched.

### 6.1 Benchmark (seed 7, unchanged generator, n=1300, 144 anomalies, rate 0.1108)

| model | thr | Prec | Rec | F1 | ROC-AUC | PR-AUC | SPIKE | DROP | DRIFT | FROZEN | MISSING |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 legacy | 0.72 | 0.1229 | 0.7986 | 0.2130 | 0.6606 | 0.2871 | 1.0000 | 1.0000 | 0.9792 | 0.6944 | 0.4333 |
| **v2 (prod at campaign time)** | 0.72 | **0.9483** | 0.3819 | 0.5446 | 0.7506 | 0.5839 | 0.8000 | 0.6667 | 0.6875 | 0.0000 | 0.0000 |
| v3 (pre-retune) | 0.30 | 0.8485 | **0.5833** | **0.6914** | **0.8994** | **0.7230** | 0.3333 | 0.4000 | 0.7083 | **0.5000** | **0.7000** |

Confusion: v2 `[[1153,3],[89,55]]` · v3 `[[1141,15],[60,84]]`

### 6.2 Unseen frame (seed 19, same generator/framework, n=1300, 144 anomalies)

| model | thr | Prec | Rec | F1 | ROC-AUC | PR-AUC | SPIKE | DROP | DRIFT | FROZEN | MISSING |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v2 (prod at campaign time)** | 0.72 | **0.9667** | 0.4028 | 0.5686 | 0.7606 | 0.6067 | 0.6667 | 0.6667 | 0.7708 | 0.0278 | 0.0000 |
| v3 (pre-retune) | 0.30 | 0.8384 | **0.5764** | **0.6831** | **0.9002** | **0.7193** | 0.1333 | 0.4000 | 0.6667 | **0.5000** | **0.8333** |

Confusion: v2 `[[1154,2],[86,58]]` · v3 `[[1140,16],[61,83]]`

Results persisted: `ml/evaluation/phase3_seed7_versions.json`,
`ml/evaluation/phase3_seed19_v2v3.json`, `ml/experiments/v3_campaign_final_summary.json`.

## 7. Honest analysis

**v3's wins are real and reproducible across both frames:**
- FROZEN recall 0.0000 → **0.50 / 0.50**; MISSING recall 0.0000 → **0.70 / 0.83**;
  DRIFT up +0.02 on the benchmark (−0.10 on unseen).
- ROC-AUC **+0.149 / +0.140**; PR-AUC **+0.139 / +0.113**; F1 **+0.147 / +0.115**;
  recall **+0.20 / +0.17**.

**v3's regressions are also real and reproducible:**
- Precision 0.9483→0.8485 and 0.9667→0.8384 (unseen dips just below the 0.85
  campaign floor).
- **SPIKE recall 0.8000→0.3333 and 0.6667→0.1333** — SPIKE was v2's strongest
  mechanism. The low fused threshold (0.30) plus flat/miss weight (0.30/0.30)
  dilutes the ML spike/drop signal. DROP falls 0.6667→0.4000 on both frames.

**Net (historical):** v3 was an evidence/trade-off artifact — stronger on the new
mechanisms and overall rank metrics, weaker on precision and short-duration
spikes/drops. That is not a "v2 is better" claim or a "v3 is worse" claim; it is
a cost-benefit trade-off. Under the pre-registered rule, the campaign concluded
**no promotion at that time**. The later decision-layer retune rebalanced that
trade-off (SPIKE/DROP recall 1.00/1.00; precision 0.717, still below v2's
0.9483) and v3 became current production (see banner / `docs/ML_DATA_PROVENANCE.md`
§9).

## 8. Regression / compatibility verification (Option A closure)

*Historical point-in-time record, as of 2026-08-29 (production was v2 then).*
The current production state is v3 (see banner / `docs/ML_DATA_PROVENANCE.md` §9);
`MODEL_SUBDIR` now points at `skyguard_v3`.

- Backend suite: **29/29** OK at the time (`unittest` discover over `tests/`).
- Frontend: **3/3** vitest OK; **production build green** (`tsc -b && vite build`).
  Frontend source untouched by the campaign.
- v2 control bit-identical on the seed-7 benchmark.
- Demo smoke (live server, production = v2 at the time): S1 SPIKE AWS-023 mag14 →
  **0.8440** HIGH SPIKE LIKELY_SENSOR_FAULT (likelihood 0.130); S2 humidity DRIFT
  mag15 dur6 → **0.7983**; S3 cross-station AWS-002 **0.8454** / AWS-003 **0.8509** —
  values from that v2-era session. Under the current v3 production the same
  scenarios measure 0.422 / 0.349 / 0.422·0.423·0.424 (see `docs/SIH_Demo_Runbook.md`
  §8 re-verification).
- No changes were made by the campaign to: `services.py`, frontend,
  `data/sample_weather.csv`, `docs/SIH_Demo_Runbook.md`,
  `backend/models/skyguard_v1|v2`, the benchmark generator or seeds.

## 9. Deliverables

| Item | Location |
|---|---|
| v3 artifact (experimental at campaign time; now current production) | `backend/models/skyguard_v3/` |
| Experiment logs (11,700 / run) | `ml/experiments/v3_val_tuning_seed11_*.json` |
| Phase 3 results | `ml/evaluation/phase3_seed7_versions.json`, `phase3_seed19_v2v3.json` |
| Campaign summary | `ml/experiments/v3_campaign_final_summary.json` |
| Training report | `docs/ML_V3_TRAINING_REPORT.md` |
| This report | `docs/ML_Performance_Improvement_Report.md` |
| Data provenance | `docs/ML_DATA_PROVENANCE.md` |
| Decision record | §2 and §7 here; summary JSON `decision` field |

## 10. Limitations (stated, not hidden)

- All datasets are synthetic and weather-typical; `synthetic=true` is disclosed
  in artifact metadata and the UI. No claims about real weather are made.
- All scores are **uncalibrated** fusion output, not probabilities.
- v3 precision and SPIKE/DROP recall regressions were judged (at campaign time)
  to make promotion unsafe without further design (e.g., spike-preserving fusion).
  No further tuning was performed in the campaign. The later ROC-channel retune
  (`backend/app/ml/retune.py`) addressed the SPIKE/DROP regression (measured
  1.00/1.00) while keeping precision (0.717) below v2's (0.9483); the current
  trade-off is recorded in `docs/ML_DATA_PROVENANCE.md` §9.
- FROZEN/MISSING recall for **v2 remains 0.0000**; automated detection of
  stuck/cold and missing sensors remains an open problem in the v2-era legacy
  artifact. The current production v3 measures 0.50 / 0.87 (real, not perfect).