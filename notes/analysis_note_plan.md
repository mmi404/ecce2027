# Analysis Note Plan: COCO-Covered vs. Non-COCO Transfer Loss Asymmetry

## 1. Research Question & Motivation
In cross-city vehicle detection transfer (Dhaka -> Chattogram), does the transfer performance drop disproportionately afflict local informal/non-motorised transport modes compared to global standardized vehicle classes?

Pretrained object detectors (e.g., YOLOv8s pretrained on MS COCO) possess rich prior visual representations for common global categories:
- **COCO-Covered Classes (5 headline classes):**
  - `car`
  - `bus`
  - `truck`
  - `motorbike` (COCO `motorcycle`)
  - `person`
- **Non-COCO / Informal Transport Classes (2 headline classes):**
  - `auto_rickshaw` (CNG, easy-bike / battery rickshaw)
  - `three_wheeler` (cycle-rickshaw, rickshaw-van)

## 2. Core Hypotheses to Test
- **Hypothesis 1 (Representation Anchor):** COCO-covered classes exhibit lower relative transfer loss ($\Delta \text{mAP}_{50}$ and $\Delta \text{Recall}$) because the backbone features are anchored by 118k diverse COCO pretraining images, buffering them against cross-city visual domain shifts.
- **Hypothesis 2 (Regional Variance Vulnerability):** Non-COCO vehicles (especially `auto_rickshaw` and `three_wheeler`) rely exclusively on features learned from the local training city (Dhaka). Because local paratransit body designs, paint schemes, passenger-hood structures, and operating environments vary sharply between Dhaka and Chattogram, these classes will experience severe cross-city transfer penalties.
- **Hypothesis 3 (Condition Interaction):** The transfer drop for informal modes is exacerbated at night, where distinctive silhouettes and retroreflective cues differ substantially across cities.

## 3. Planned Analytical Metrics & Methodology
1. **Transfer Gap Definition:**
   For each class $c$:
   $$\Delta \text{AP}(c) = \text{AP}_{\text{E2}}(c) - \text{AP}_{\text{E1}}(c)$$
   $$\text{Relative Transfer Loss}(c) = \frac{\text{AP}_{\text{E2}}(c) - \text{AP}_{\text{E1}}(c)}{\text{AP}_{\text{E2}}(c)}$$
   (Analogous metrics computed for Recall@0.5).
2. **Group Comparison:**
   - Compute mean relative transfer loss for **Group A (COCO-Covered)** vs. **Group B (Non-COCO)**.
   - Run a two-sample permutation test / bootstrap test across bootstrap iterations to evaluate statistical significance ($p < 0.05$).
3. **Error Type Breakdown (TIDE / Confusion Matrices):**
   - Classify false negatives into background misclassifications vs. inter-class confusions (e.g., auto_rickshaw confused with car or three_wheeler).
   - Evaluate whether non-COCO modes suffer more from localization error or classification confusion.
4. **Stratification:**
   - Repeat the group comparison across Day vs. Night subsets.

*Note:* This plan documents the framework for post-training analysis. Execution will take place once prediction JSONs are produced.
