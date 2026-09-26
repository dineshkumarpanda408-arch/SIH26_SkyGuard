# Master Prompt — SkyGuard AI Real ML Performance Improvement (v3 campaign)

> **HISTORICAL record.** This is the written prompt that drove the v3 campaign.
> Its "baseline" (v2, threshold 0.72) describes the production state at campaign
> start, and its targets (e.g., precision ≥ 0.85) were the campaign's targets —
> not the current production configuration and not the active acceptance
> criterion. The campaign was implemented; after the later decision-layer retune
> (`backend/app/ml/retune.py`), `skyguard-multivariate-v3` is the current
> production model. See `docs/ML_DATA_PROVENANCE.md` §9.

Copy everything below to drive the ML improvement agent/engineer.

---

You are working on my existing SkyGuard AI project (Smart India Hackathon). It is
**already implemented and working** — v1 and v2 artifacts ship, the frontend is
frozen, the demo runbook exists. This campaign improves the **underlying detector**
only, by real ML/statistics. It adds, it does not replace.

## HARD RULES (never violate)

1. DO NOT rebuild, redesign the frontend, or change the theme/pages.
2. DO NOT fabricate, hard-code, or hide metrics. All numbers come from real
   detection runs on a fixed benchmark.
3. DO NOT randomly flip thresholds to inflate percentages.
4. DO NOT retune on the final test frame. Use chronology and validation only.
5. DO NOT introduce train/test leakage (no future info, no test-statistics).
6. DO NOT duplicate rows or fake "more data". Data grows by extending the
   weather generator with new regimes, longer span, and realistic anomaly shapes,
   each with explicit ground truth metadata.
7. Preserve v1 and v2 artifacts and the existing evaluation benchmark unchanged,
   and keep the demo scenarios working (they are a smoke check, not the goal).
8. If a change does not statistically improve the benchmark, reject it and record it.
9. Keep provenance labels (`computed_by`, model/dataset version, synthetic=true)
   and the SHAP/statistical attribution — never replace with fake text.

## BASELINE (measured, reproduced, do not lose)

Controlled hard evaluation frame — 1,300 windows, seed 7, threshold 0.72 —
scored by the persisted `skyguard_v2` artifact:

| Metric | v2 today |
|---|---|
| Precision | 0.9483 |
| Recall | 0.3819 |
| F1 | 0.5446 |
| ROC-AUC | 0.7506 |
| Confusion | TP 55 · FP 3 · FN 89 · TN 1153 |

Per-category recall (injected → detected):

- SPIKE 15 → 12 (0.800)  ·  DROP 15 → 10 (0.667)
- DRIFT 48 → 33 (0.688)  ·  FROZEN_SENSOR 36 → 0 (0.000)
- MISSING_DATA 30 → 0 (0.000)

Measured fused-score statistics on injected rows (why categories fail):

- FROZEN mean 0.503 / max 0.664  → plateau is not a forest outlier shape.
- MISSING mean 0.434 / max 0.514  → NaN is erased by `preprocessing.impute`
  (interpolate limit=3) then replaced by column median in `decision_scores` before
  scoring; there is no timestamp-gap signal either (eval injections keep the grid).

## GOAL

Fix the two 0% categories and raise DRIFT, while keeping precision high and
reporting the precision/recall trade-off honestly.

Acceptable, honest targets (illustrative, not guaranteed):

- MISSING_DATA recall ≥ 0.85   (dedicated missingness detector)
- FROZEN_SENSOR recall ≥ 0.60  (variance/flatline detector)
- DRIFT recall ≥ 0.78          (residual/CUSUM time-series signal)
- Precision ≥ 0.85 overall, F1 ≥ 0.65, ROC-AUC ≥ 0.80
- If targets are not reachable, report the real numbers. An honest 0.6 beats a fake 0.99.

## ARCHITECTURE STEP (the core change)

Do not try to make one isolation forest detect every anomaly mechanism.

Change:

    one detector -> every anomaly type

to:

    one detector per anomaly mechanism  ->  evidence fusion

- ML channel: isolation forest on physical + features (unchanged role).
- TS channel: residual-based drift (diurnal/seasonal-adjusted residual +
  EWMA/CUSUM/change-point persistence), SPIKE/DROP via robust z-score on residual.
- Data-quality channel: **dedicated missingness detector** on the RAW (pre-impute)
  stream — NaN runs, per-variable invalid values, timestamp gaps (> interval +
  tolerance), consecutive missing observations, single-variable outages.
- Flatline channel: **FROZEN detector** — rolling variance/std, unique-value count,
  zero-difference ratio over configurable persistence window, entropy.
- Fuse evidence transparently; keep `computed_by` provenance; keep score pure
  (flags may inform detection decision but must be disclosed as non-ML).

## STEPS

1. **Audit** (DONE — see `docs/ML_AUDIT.md`): inventory + reproduce baseline +
   root-cause. Baseline saved to
   `ml/evaluation/baseline_before_improvement.json`.
2. **Data v3**: extend `backend/data_generator.py` → longer span (e.g., 60 days),
   more synoptic regimes, station heterogeneity preserved, realistic sensor
   behaviors (gradual degradation, short/long freezes, dropout bursts, recovery)
   with explicit `injection_meta` ground truth. Synthetic stays `true`.
   Chronological TRAIN/VAL/TEST (70/15/15) on the NEW dataset.
3. **Features**: add residual-based features (observed − diurnal/seasonal/station
   expectation), drift slope/EWMA/CUSUM, flatline statistics, missingness window
   features. All causal; all available at inference.
4. **Detectors**: implement dedicated missingness + flatline + drift channels
   (new modules under `backend/app/ml/`), fit on TRAIN only.
5. **Fusion**: extend `ModelManager` scoring to the additional channels; tune
   weights + threshold ON VALIDATION ONLY; record every experiment under
   `ml/experiments/`.
6. **Evaluate v3 on the UNCHANGED hard frame** — same as v2/v1 — plus report
   confusion matrix, ROC-AUC, PR-AUC, per-category recall, and the thresholds used.
   If worse anywhere, keep v2 and say so.
7. **Version**: save as `skyguard_v3` (never overwrite v1/v2). Metadata: artifact
   paths, feature schema, dataset v3 hash, split bounds, hyperparameters, seed,
   train/val/test metrics, timestamp.
8. **Backward-compatible API**: new route(s)/fields additive; existing contracts
   intact. Evaluation page gains a clear "Production v3 · v2 · v1" honest block.
9. **Final validation**: train from scratch, save, load, run unseen frame, run
   the unchanged benchmark, run the 3 demo scenarios, verify attribution,
   correction, provenance, then backend tests (currently 29) + frontend build.
10. **Report**: `docs/ML_Performance_Improvement_Report.md` (baseline vs new,
    per-category, confusion, ROC/PR-AUC, leakage control, hyperparameters,
    limitations, reproducibility) + `docs/ML_DATA_PROVENANCE.md`.

## SUCCESS / STOP RULES

- Show the precision/recall trade-off explicitly; never hide it.
- Never present an ablation/standalone score as production performance.
- Never tune to station AWS-023 or the demo magnitudes.
- Judges must be able to trace each new number:
  DATA → FEATURES → TRAINING → VALIDATION → TEST → METRICS, no fake steps.