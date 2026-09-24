# Related work (WP4 §4 reading list — verified against arXiv abstracts, 2026-09-20)

## BadODD — arXiv 2401.10659
Baig, Hajong, Patwary, Rahman, Chowdhury (2024), "BadODD: Bangladeshi Autonomous
Driving Object Detection Dataset." Introduces the primary dataset this paper uses:
10,032 labelled images from dash-mounted smartphones across 9 Bangladeshi districts,
13 classes defined by physical characteristics (power source, wheel count, size)
rather than local names — e.g. `three_wheeler` = "3 wheeler, paddle, small" (cycle-
rickshaw) is explicitly split from `auto_rickshaw` = "3 wheeler, gas/electric, medium"
(CNG or battery/"tomtom"), and `cart_vehicle` = "human/animal, no paddle." **Cite for:**
dataset provenance, licence, and the exact class-definition table underpinning this
paper's mode-group taxonomy (`notes/decisions.md`).

## BMD-45 — arXiv 2604.24419
Sharma et al. (CVPR 2026 Findings), "BMD-45: A Large-Scale CCTV Vehicle Detection
Dataset for Urban Traffic in Developing Cities." 45K fixed CCTV images, 480K boxes,
14 region-specific vehicle classes. Reports a large domain gap between a
structured-traffic-trained detector and an in-domain one (33.6% vs. 83.8%
mAP@0.5:0.95, a 2.5x gap), attributed to viewpoint/elevation differences between
structured and unstructured traffic camera placements. **The abstract does not report
a class-wise breakdown of this transfer gap** — this paper's RQ2 (does the gap
concentrate on informal/non-motorised classes specifically) is not pre-answered by
BMD-45's headline number. **Cite for:** establishing that structured→unstructured
transfer degradation is already known (do not re-claim it); motivating why a
class-resolved breakdown is the open question.

## Premier University YOLO evaluation — arXiv 2509.05652
Hossain et al., "Evaluating YOLO Architectures: Implications for Real-Time Vehicle
Detection in Urban Environments of Bangladesh." Benchmarks 6 YOLO variants on a
29-class Bangladeshi vehicle dataset (best: YOLOv11x, 63.7% mAP@0.5). Single-domain
evaluation only — **no cross-city or cross-domain transfer experiment**, and rare
classes are reported to collapse to near-zero accuracy under class imbalance (the
same pattern this paper finds for `bicycle`/`cart_vehicle`/`priority_vehicle`/
`construction_vehicle`). **Cite for:** confirming YOLO-on-Bangladeshi-vehicles is
already benchmarked (do not re-claim it) and that rare-class collapse is a known,
independent phenomenon from the transfer gap.

## Katare et al., vulnerable-class bias — arXiv 2401.10397
Katare, Solans Noguero, Park, Kourtellis, Janssen, Ding (2024), "Analyzing and
Mitigating Bias for Vulnerable Classes: Towards Balanced Representation in Dataset."
Shows detection disparities for vulnerable road users (cyclists, pedestrians,
motorcyclists) on nuScenes, and that cost-sensitive resampling narrows them.
Structured-domain (nuScenes), not Bangladeshi/unstructured traffic, and about
class-imbalance-driven bias generally rather than cross-city transfer specifically.
**Cite for:** framing — vulnerable/minority-mode detection bias is a recognised
problem class in perception literature; this paper studies its cross-city-transfer
instance specifically.

## CityGen / CityTransfer-Bench — arXiv 2605.29935
Qian et al., "CityGen: Structure-Guided City-Style Synthesis for Cross-City Autonomous
Driving." Introduces **CityTransfer-Bench**, a benchmark for cross-city generalisation
spanning perception, segmentation and planning, plus CityGen, a diffusion method that
synthesises target-city-style training imagery from HD maps to close the gap without
labelled target data. Broader in scope (multi-task, synthesis-based mitigation) and not
focused on class-wise vehicle-detection breakdowns. **Cite for:** cross-city detection
generalisation is an active, separately-benchmarked topic (do not re-claim novelty of
"a benchmark exists"); this paper's contribution is the class-resolved analysis on a
real (not synthesised) same-protocol, same-viewpoint dataset.

## AI City Challenge 2026, Track 6
aicitychallenge.org — an active annual competition track on cross-domain vehicle
detection/tracking, evidence that cross-city generalisation is a live benchmark topic
in the wider CV community. Cited as context only, not for a specific numeric claim.

---

## What this confirms about novelty (do not re-litigate; see AGENT_BRIEF.md §2)

None of the above does a controlled, same-annotation-protocol, same-viewpoint,
class-resolved Dhaka→Chattogram transfer study with recall-focused, mode-grouped
reporting. BMD-45 establishes the *existence* of a structured/unstructured camera-
viewpoint gap (a different axis — this paper holds viewpoint fixed, since BadODD is
dashcam-only in both cities) but not its per-class shape. The Premier University paper
and other Bangladeshi-YOLO benchmarks don't do cross-city transfer at all. CityGen/
CityTransfer-Bench benchmarks cross-city generalisation broadly but not with this
paper's class-wise, recall-first, mode-group lens. This paper's RQ2 finding — that the
transfer gap is ~3-4x larger for informal/non-motorised modes than for globally-standard
vehicle classes, holding annotation protocol and viewpoint fixed — is not pre-answered
by any of these.
