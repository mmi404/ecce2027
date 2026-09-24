# BadODD Data Reconnaissance and Audit Report (WP1 — Zenodo Final)

**Date:** 19 September 2026  
**Primary Source:** Zenodo Record `13823722` (Concept `10532180`, DOI `10.5281/zenodo.13823687`, version `complete`)  
**Archive File:** `data/raw/zenodo/badodd.zip` (4,361,793,657 bytes)  
**Verified MD5 Checksum:** `6a0d3983b379302ab79a99baf7d47d74` (Exact match: `True`)  
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 1. Data Provenance & The Kaggle vs. Zenodo Comparison

### 1.1 Provenance and Why Zenodo Direct Download Failed Earlier
- **Zenodo Archive Verification:** The local archive `data/raw/zenodo/badodd.zip` matches the official MD5 checksum (`6a0d3983b379302ab79a99baf7d47d74`) exactly.
- **Root Cause of Prior Download Failure:**  
  Zenodo's Nginx frontend reverse proxy intermittently returned:  
  `HTTP/1.1 504 Gateway Time-out ("The server didn't respond in time")`.  
  When single-stream downloads connected, throughput was throttled to ~100–150 KiB/s, eventually raising:  
  `urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead')` and `ReadTimeoutError (timeout=30)`.  
  CERN/Zenodo's backend storage cluster experienced severe gateway timeouts on the multi-gigabyte archive.

### 1.2 Empirical Resolution of the 1,965-Image Gap
By inspecting and extracting the full Zenodo archive, the image count discrepancy between Kaggle and Zenodo is now **empirically resolved**:

| Source | Train Split | Val Split | Test Split | Total Images | Label Status |
|---|---|---|---|---|---|
| **Kaggle Competition Copy** | 5,896 | *Omitted* (0) | 1,964 | 7,860 | Train labeled (5,896); Test unlabelled (1,964) |
| **Zenodo Official Archive** | **5,967** | **2,032** | **2,033** | **10,032** | **100% of all 10,032 images have matching labels** |

- **Finding:** The Kaggle competition organizers distributed only the 60% training split and an unlabelled 20% test split, withholding the 20% validation split.
- **Zenodo Status:** The Zenodo archive contains **all three splits** (`train`, `val`, and `test`), with clean YOLO annotations for every single image ($10,032$ images with $10,032$ label files).
- **Paper Policy:** The Kaggle competition copy is discarded. The paper strictly uses the official **Zenodo CC BY 4.0 archive**.

---

## 2. District Image Distribution (Zenodo Complete Dataset)

All 10,032 images are 100% assignable to districts via filename prefixes:

| District | Clean Image Count | Share of Dataset | Number of Sequences |
|---|---|---|---|
| **Dhaka** | **1,941** | 19.3% | 9 sequences |
| **Mymensingh** | 1,646 | 16.4% | 6 sequences |
| **Sylhet** | 1,523 | 15.2% | 5 sequences |
| **Khulna** | 1,336 | 13.3% | 8 sequences |
| **Chattogram** | **1,183** | 11.8% | 3 sequences |
| **Maowa** | 1,030 | 10.3% | 4 sequences |
| **Sherpur** | 735 | 7.3% | 3 sequences |
| **Sirajganj** | 470 | 4.7% | 1 sequence |
| **Rajshahi** | 168 | 1.7% | 5 sequences |
| **Total** | **10,032** | **100.0%** | **44 sequences** |

---

## 3. Per-Class Ground-Truth Instance Counts (Dhaka vs. Chattogram)

Computed across all 10,032 labeled images:

| Class Name | Dhaka Instances | Chattogram Instances | Total Dataset Instances | CTG Status ($\ge 30$) |
|---|---|---|---|---|
| `person` | **3,646** | **2,630** | 31,016 | **PASS** |
| `auto_rickshaw` (CNG / easy-bike) | **875** | **1,896** | 17,820 | **PASS** |
| `car` | **2,214** | **1,200** | 6,336 | **PASS** |
| `three_wheeler` (cycle-rickshaw) | **1,606** | **1,037** | 9,664 | **PASS** |
| `truck` | **1,109** | **710** | 3,863 | **PASS** |
| `motorbike` | **948** | **447** | 6,272 | **PASS** |
| `bus` | **1,191** | **446** | 3,096 | **PASS** |
| `bicycle` | 139 | 19 | 1,195 | *Below threshold* ($< 30$) |
| `cart_vehicle` | 27 | 19 | 238 | *Below threshold* ($< 30$) |
| `construction_vehicle` | 6 | 12 | 41 | *Below threshold* ($< 30$) |
| `priority_vehicle` | 39 | 10 | 369 | *Below threshold* ($< 30$) |
| `train` | 41 | 0 | 42 | *Zero instances in CTG* |
| `wheelchair` | 0 | 0 | 84 | *Zero instances in both* |
| **Total Instances** | **11,841** | **8,426** | **80,036** | — |

---

## 4. Sequence Structure & Sequence-Level Statistics

### 4.1 Chattogram Sequences
Chattogram comprises **exactly 3 video driving runs** totaling 1,183 images:

| Sequence Name | Condition | Duration | Frames | Images | auto_rickshaw | bus | car | motorbike | person | three_wheeler | truck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `chittagong_night1` | Night | 11.8 min | 0–21,240 | **695** | 1,066 | 293 | 675 | 223 | 1,128 | 233 | 470 |
| `chittagong_bohoddarhat1` | Day | 9.4 min | 708–16,933 | **275** | 444 | 139 | 215 | 95 | 645 | 367 | 211 |
| `chittagong_bohoddarhat2` | Day | 6.9 min | 0–12,508 | **213** | 386 | 14 | 310 | 129 | 857 | 437 | 29 |
| **Chattogram Total** | — | **28.1 min** | — | **1,183** | **1,896** | **446** | **1,200** | **447** | **2,630** | **1,037** | **710** |

#### Class Concentration Findings in Chattogram:
- **`bus`:** 96.9% concentrated in `chittagong_night1` (293) and `chittagong_bohoddarhat1` (139). Only 14 instances in `bohoddarhat2`.
- **`truck`:** 95.9% concentrated in `chittagong_night1` (470) and `chittagong_bohoddarhat1` (211). Only 29 instances in `bohoddarhat2`.
- **`three_wheeler` (Cycle-rickshaw):** Strongly daytime-dominant (804 daytime instances vs. 233 at night).
- **`auto_rickshaw` (CNG/easy-bike), `car`, `person`, `motorbike`:** Well-represented across all three runs.

### 4.2 Dhaka Sequences
Dhaka comprises **9 sequences** totaling 1,941 images:

| Sequence Name | Condition | Images | auto_rickshaw | bus | car | motorbike | person | three_wheeler | truck |
|---|---|---|---|---|---|---|---|---|---|
| `dhaka_night3` | Night | 869 | 147 | 258 | 717 | 228 | 606 | 274 | 638 |
| `dhaka2` | Day | 542 | 383 | 434 | 968 | 323 | 1,223 | 541 | 362 |
| `dhaka3` | Day | 122 | 170 | 127 | 187 | 145 | 436 | 159 | 64 |
| `dhaka4` | Day | 112 | 31 | 0 | 118 | 102 | 629 | 356 | 2 |
| `dhaka5_khilkhet` | Day | 83 | 18 | 6 | 0 | 13 | 403 | 2 | 0 |
| `dhaka_night1` | Night | 81 | 12 | 180 | 84 | 24 | 138 | 136 | 19 |
| `dhaka1` | Day | 72 | 92 | 66 | 32 | 104 | 170 | 40 | 16 |
| `dhaka_night2` | Night | 50 | 22 | 120 | 108 | 8 | 41 | 98 | 7 |
| `dhaka_night4` | Night | 10 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| **Dhaka Total** | — | **1,941** | **875** | **1,191** | **2,214** | **948** | **3,646** | **1,606** | **1,109** |

---

## 5. Evaluation Design & Proposed 3-Fold Chattogram Partition

### 5.1 The 3 Chattogram Folds
Because Chattogram has exactly 3 macro-sequences, assigning each sequence to one fold yields:

| Fold | Assigned Sequence | Condition | Test Images | Train Images (Other 2 Folds) |
|---|---|---|---|---|
| **Fold 1** | `chittagong_night1` | Night | **695** | 488 (Day only: `bohoddarhat1` + `bohoddarhat2`) |
| **Fold 2** | `chittagong_bohoddarhat1` | Day | **275** | 908 (Mix: `night1` + `bohoddarhat2`) |
| **Fold 3** | `chittagong_bohoddarhat2` | Day | **213** | 970 (Mix: `night1` + `bohoddarhat1`) |
| **Total** | — | — | **1,183** | Pooled out-of-fold evaluations cover **100% of CTG** |

> [!WARNING]
> **Critical Methodological Note on Fold 1:**  
> If Fold 1 is trained on the other two sequences (`bohoddarhat1` + `bohoddarhat2`), its training set contains **zero night images**. Evaluating Fold 1 on `chittagong_night1` will introduce a pure **day-to-night domain shift**, causing the in-domain baseline for Fold 1 to drop due to illumination rather than geography.  
> **Alternative Partition (Recommended for consideration):** If temporal blocks within sequences are permitted (e.g. splitting `chittagong_night1` into 3 non-overlapping 4-minute continuous chunks, and splitting daytime runs similarly), each fold can contain ~394 images balanced across both day and night!

### 5.2 Controlled Training Set Size
- Average E2 fold training set size across the 3 folds: $\approx \mathbf{788}$ images (or exactly matching each fold's training size).
- The Dhaka training set for E1 will be subsampled to **exactly match the E2 fold training size** using sequence-grouped sampling.

---

## 6. ⛔ Gate A Final Assessment (Zenodo Numbers)

1. **Chattogram Evaluation Set Size:** **1,183 images** (Requirement: $\ge 300$). **PASSED (3.94× threshold).**
2. **Per-Class Chattogram Instance Threshold ($\ge 30$):**
   * **7 Headline Classes PASS with large margins:**
     - `person`: 2,630 instances
     - `auto_rickshaw`: 1,896 instances
     - `car`: 1,200 instances
     - `three_wheeler`: 1,037 instances
     - `truck`: 710 instances
     - `motorbike`: 447 instances
     - `bus`: 446 instances
   * **Minor classes below threshold in CTG:** `bicycle` (19), `cart_vehicle` (19), `construction_vehicle` (12), `priority_vehicle` (10), `train` (0), `wheelchair` (0).
3. **Taxonomy for Training:** Train on all 11 non-zero classes (drop only `wheelchair` and `train`). Report headline metrics on the 7 passing classes across the 5 agreed mode groups.
