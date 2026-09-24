# Threats to Validity and Dataset Limitations

**Paper Title:** Dhaka $\rightarrow$ Chattogram Object Detector Transfer, Class by Class  
**Venue:** ECCE 2027  
**Date Logged:** 19 September 2026

---

## 1. Geographic Scoping & Representative Limitations of Chattogram Subset

1. **Restricted Geographic Footprint:**
   - The Chattogram subset in BadODD consists of **exactly 3 dashcam drives** totaling **28.1 minutes** of driving footage (1,183 frames).
   - The daytime footage was captured exclusively in the **Bohoddarhat** area (`chittagong_bohoddarhat1` and `chittagong_bohoddarhat2`), a specific commercial/transport hub with heavy bus, truck, and rickshaw density.
   - The night footage (`chittagong_night1`, 11.8 minutes) captures a single corridor at night.
   - **Paper Scoping Rule:** All claims in the paper must be strictly scoped to **"BadODD's Chattogram subset"** and **never** generalized to "Chattogram traffic" as a whole. Major arterial corridors (e.g., Agrabad port access road, GEC circle, CDA Avenue) and other times of day/weather are not represented.

2. **Illumination and Sequence Asymmetry:**
   - Night driving represents **58.7%** of the Chattogram data (695 out of 1,183 frames).
   - In contrast, daytime driving represents 41.3% (488 frames).
   - Heavy vehicles (`bus` and `truck`) are >95% concentrated in the night run and the first Bohoddarhat run; `bohoddarhat2` has almost no buses (14) or trucks (29).
   - Cycle-rickshaws (`three_wheeler`) are daytime-dominant (77.5% during day, 22.5% at night).

3. **Viewpoint Confound:**
   - All imagery is captured from a dash-mounted smartphone inside a vehicle looking forward through the windshield/hood, **not** an elevated fixed traffic surveillance camera (CCTV / pole-mounted).
   - While valid for autonomous driving and mobile sensing, perspective distortion, vehicle-to-vehicle occlusion, and hood reflections differ from intersection monitoring deployments.

4. **Annotation Protocol & Taxonomy Limitations:**
   - Annotations are 2D bounding boxes based on the 13 vehicle/road user categories defined by Baig et al. (arXiv:2401.10659).
   - Microbuses and jeeps are subsumed under `car`; legunas (human haulers) are not distinctly separated from buses/trucks.
   - Ground truth represents the consensus of the original dataset annotators without subsequent re-labelling.

---

## 2. Experimental Controls Implemented to Mitigate Confounders

1. **Temporal-Block Folds with Dead-Zone Buffers:**
   - To eliminate video-frame correlation and near-duplicate leakage, consecutive video runs are partitioned into contiguous temporal blocks separated by **$\ge 15$-second buffer zones**. Buffer frames are completely excluded from both training and evaluation.
2. **Day/Night Ratio Matching:**
   - To prevent illumination from confounding the cross-city geographic transfer, Dhaka training sets are stratified to match Chattogram's day/night ratio (**~58.7% night**).
3. **Training Sample Volume Equalization:**
   - The Dhaka training volume for E1 is strictly subsampled to match the in-domain E2 fold training size, eliminating training set size as an explanatory factor for transfer degradation.
4. **Stratified Reporting:**
   - Metrics will be reported overall, and disaggregated by daytime vs. nighttime conditions, and disaggregated across the 5 transport mode groups.

5. **Robustness Folds Illumination Asymmetry (Conservative Bias):**
   - In the leave-one-day-sequence-out robustness splits, holding out an entire daytime sequence leaves only a single daytime corridor available for training (`bohoddarhat2` with 213 frames for Fold A, `bohoddarhat1` with 275 frames for Fold B).
   - Under size-matching (749 training images), the resulting night shares are **71.6% night** for Fold A (536 night, 213 day) and **63.3% night** for Fold B (474 night, 275 day), compared to **59.3% night** in the primary folds.
   - *Impact on Validity:* Because the robustness models are trained on substantially fewer daytime frames (213 and 275 vs. 305 in primary folds), the evaluation on the held-out daytime corridor is **conservative**; any observed performance drop reflects an adversarial evaluation biased *against* the Chattogram in-domain model, providing a robust lower-bound on in-city corridor transferability.
