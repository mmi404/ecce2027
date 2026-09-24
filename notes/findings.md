# WP4 Findings (local analysis from saved predictions)

**Status:** Complete for the headline result. Point estimates validated (local
pycocotools-free AP50 implementation matches Kaggle's own pycocotools output exactly,
0.0000 diff across all 11 classes on a spot check). 95% bootstrap CIs (300 image-level
resamples per class, `src/run_wp4_bootstrap.py`) are done for all 11 classes;
see `results/transfer_gap.csv`.

## RQ1: Is there a transfer gap, and how big?

E1 (Dhaka-trained, mean of 3 folds) vs. E2 (Chattogram-trained, out-of-fold pooled),
both evaluated on the identical 1,121-image `ctg_pooled_eval` set:

- **Overall AP50: E1 ≈ 0.22-0.26 per fold, E2 (pooled) higher on every headline class
  except `motorbike` and `bus`.**
- Full per-class table: `results/headline_per_class_point.csv`.

## RQ2 (the headline result): the gap is concentrated in informal/non-motorised modes

Mode-group aggregation (weighted by each class's Chattogram instance count),
`results/mode_group_summary.csv`:

| Mode group | Classes (n instances) | E1 AP50 | E2 AP50 | **AP50 gap** | E1 Recall | E2 Recall | **Recall gap** |
|---|---|---|---|---|---|---|---|
| Formal motorised | car, bus, truck, motorbike (n=2621) | 0.410 | 0.471 | **+0.061** | 0.355 | 0.436 | **+0.081** |
| Pedestrian | person (n=2509) | 0.272 | 0.366 | +0.094 | 0.214 | 0.333 | +0.119 |
| Informal motorised | auto_rickshaw — CNG/battery (n=1796) | 0.558 | 0.755 | **+0.197** | 0.466 | 0.728 | **+0.263** |
| Non-motorised | three_wheeler (cycle-rickshaw), bicycle, cart_vehicle (n=1026) | 0.315 | 0.540 | **+0.225** | 0.245 | 0.521 | **+0.277** |

**The gap is not uniform.** It's smallest for globally-standard vehicle classes
(formal motorised: +0.06 AP50 / +0.08 recall) and 3-4x larger for the two
locally-specific informal/non-motorised modes (+0.20-0.23 AP50, +0.26-0.28 recall).
This matches RQ2's hypothesis directly, and the gap is measured almost entirely on
well-powered classes (`auto_rickshaw` n=1,796, `three_wheeler` n=1,037 within
non-motorised's 1,026 total) — not on the near-zero rare classes.

**Bootstrap CIs confirm this is real, not noise, for the two locally-specific classes
and for car/person/truck** (all exclude zero): `auto_rickshaw` +0.197 [0.180, 0.216],
`three_wheeler` +0.234 [0.203, 0.267], `car` +0.094 [0.069, 0.117], `person` +0.094
[0.077, 0.111], `truck` +0.098 [0.056, 0.137]. `bus` [-0.035, 0.070] and `motorbike`
[-0.096, 0.002] CIs span zero — per the stated protocol, reported as "no evidence of a
class-level difference," not folded into a claim that formal-motorised overall is
gap-free (its weighted point estimate is still carried by `car`/`truck`, both
individually significant).

**One genuine exception, reported straight (not hidden):** `motorbike` shows a small
*negative* gap (E1 AP50=0.414 > E2 AP50=0.369) — the Dhaka-trained model slightly
outperforms the Chattogram-trained reference on motorbikes in Chattogram. `bus` recall
is also flat-to-negative. So "formal motorised" isn't uniformly gap-free either; it's
just consistently the smallest and sometimes reversed, while the two informal/
non-motorised classes are consistently and substantially worse.

## Robustness check (leave-one-day-sequence-out, no temporal-block cuts)

Confirms the pattern is not an artifact of the temporal-block CV partitioning
(`results/per_class_ap.csv`, `rob_a`/`rob_b` vs. mean of the 3 E1 models on the same
held-out corridor):

| Class | Corridor A gap | Corridor B gap |
|---|---|---|
| `three_wheeler` | **+0.278** | **+0.162** |
| `auto_rickshaw` | +0.056 | +0.103 |
| `car` | +0.046 | +0.028 |
| `motorbike` | +0.071 | +0.019 |
| `truck` | -0.040 | -0.002 |
| `bus` | -0.047 | -0.161 |

`three_wheeler` (cycle-rickshaw) has by far the largest gap in both independently-held-out
day corridors — same conclusion as the main temporal-block design.

## Reverse transfer (CTG-trained → Dhaka): asymmetric, and asymmetric differently

E2 evaluated on its matching Dhaka fold is *always* worse than that fold's own
Dhaka-trained (E1) in-domain score (expected: E2 never saw Dhaka). But the classes hit
hardest are different in this direction — `truck` and `bus` degrade most
(-0.24 to -0.39 AP50), not the informal/non-motorised classes. So the transfer penalty's
class-concentration is direction-dependent, not just a fixed "hard class" ranking —
worth a sentence in Discussion.

## Mode-group bootstrap CI (`results/mode_group_ci.csv`)

All 4 groups' AP50/recall gap CIs exclude zero — every group has a real, non-zero gap.
But formal-motorised's CI is the narrowest and lowest by a wide margin, and does not
overlap the other three groups':

| Mode group | AP50 gap 95% CI | Recall gap 95% CI |
|---|---|---|
| Formal motorised | [+0.043, +0.081] | [+0.064, +0.100] |
| Pedestrian | [+0.073, +0.112] | [+0.100, +0.135] |
| Informal motorised | [+0.180, +0.217] | [+0.241, +0.282] |
| Non-motorised | [+0.195, +0.256] | [+0.245, +0.305] |

Correct framing: not "formal motorised has no gap" (it does, CI excludes zero), but
"its gap is consistently and substantially smaller than the two locally-specific
groups', 3-5x narrower/lower, non-overlapping intervals."

## Done since last update

- Day/night stratification (`results/day_night_stratified.csv`): pattern holds in both
  conditions; `bus`/`motorbike` flip to clearly negative at night specifically.
- Normalised confusion matrices (`results/confusion_e1.csv`, `confusion_e2.csv`,
  figures in `paper/figures/`): E1's background (miss) rate is visibly higher than
  E2's for every class, most for `auto_rickshaw`/`three_wheeler`. Largest single
  misclassification: `three_wheeler`→`auto_rickshaw` under E1 (6%).
- Auto-selected qualitative failure figure (`paper/figures/qualitative_failures.pdf`,
  `results/qualitative_failure_selection.csv`): top 12 by recall drop, several
  low-light/glare frames where E1 detects nothing. Known limitation: some selected
  frames are near-duplicate consecutive frames from the same sequence (unconditioned
  selection); noted in the paper's Limitations.

## Remaining gaps (not skipped, just not done)

- `bicycle`/`cart_vehicle`/`priority_vehicle`/`construction_vehicle` AP50 ≈ 0 everywhere
  — expected (below the ≥30-instance Gate A threshold), reported as a footnote only, no
  individual claims made.
- Qualitative figure selection doesn't deduplicate near-identical consecutive frames.
- No road-type-mix control between Dhaka and Chattogram splits.
