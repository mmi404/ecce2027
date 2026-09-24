# Dataset Splits, Cross-Validation, & Evaluation Protocol (WP2 & WP3)

**Primary Dataset Source:** Zenodo Record `13823722` (10,032 fully labeled images, CC BY 4.0)  
**Methodology:** Temporal-Block 3-Fold Cross-Validation with 15-Second Buffer Zones & Symmetric Matching

---

## 1. Chattogram Primary Splits (3-Fold Temporal-Block CV)

Each of Chattogram's 3 continuous video drives is partitioned into 3 contiguous temporal blocks separated by $\ge 15$-second temporal buffers. Buffer frames are excluded from all training and evaluation sets to prevent near-duplicate leakage.

| Split Name | Total Images | Night Images | Day Images | Night Share | Description |
|---|---|---|---|---|---|
| `ctg_fold0_eval` | **372** | 221 | 151 | 59.4% | Fold 0 Test Set (Block 0 of all 3 drives) |
| `ctg_fold1_eval` | **372** | 221 | 151 | 59.4% | Fold 1 Test Set (Block 1 of all 3 drives) |
| `ctg_fold2_eval` | **377** | 223 | 154 | 59.2% | Fold 2 Test Set (Block 2 of all 3 drives) |
| `ctg_fold0_train` | **749** | 444 | 305 | 59.3% | Training Set for Fold 0 Model (Remaining 2 blocks) |
| `ctg_fold1_train` | **749** | 444 | 305 | 59.3% | Training Set for Fold 1 Model (Remaining 2 blocks) |
| `ctg_fold2_train` | **744** | 442 | 302 | 59.4% | Training Set for Fold 2 Model (Remaining 2 blocks) |
| **`ctg_pooled_eval`** | **1121** | 665 | 456 | 59.3% | **Combined 100% CTG Evaluation Set** (Evaluated by E1 and pooled E2) |
| `ctg_buffers_excluded` | **62** | 30 | 32 | 48.4% | Buffer frames dropped at cut boundaries (15s dead-zone) |

### Per-Class Instance Distribution in Chattogram Evaluation Folds

| Headline Class | Fold 0 Eval | Fold 1 Eval | Fold 2 Eval | Pooled Eval (All CTG) | Gate A Threshold ($\ge 30$) |
|---|---|---|---|---|---|
| `auto_rickshaw` | 592 | 636 | 568 | **1796** | PASS in all folds |
| `bus` | 144 | 100 | 167 | **411** | PASS in all folds |
| `car` | 261 | 445 | 416 | **1122** | PASS in all folds |
| `motorbike` | 117 | 64 | 220 | **401** | PASS in all folds |
| `person` | 898 | 677 | 934 | **2509** | PASS in all folds |
| `three_wheeler` | 366 | 371 | 251 | **988** | PASS in all folds |
| `truck` | 307 | 210 | 151 | **668** | PASS in all folds |

---

## 2. Dhaka Symmetric Protocol & Training Set Size Matching

Dhaka video sequences are partitioned into 3 temporal blocks with $\ge 15$-second buffers.  
*Sequence Handling Note:* Short/dense sequences (`dhaka5_khilkhet`: 83 consecutive frames, step 1, ~3s; and `dhaka_night4`: 10 frames) are kept intact without splitting and assigned entirely to Fold 0 to prevent temporal leakage.

To control for training set size and day/night illumination ratio, Dhaka training sets are subsampled to match Chattogram's exact training set size (~747 images) and day/night ratio (~59% night):

| Split Name | Total Images | Night Images | Day Images | Night Share | Matching Reference |
|---|---|---|---|---|---|
| `dhaka_fold0_train_matched` | **749** | 444 | 305 | 59.3% | Matches `ctg_fold0_train` (749) |
| `dhaka_fold0_eval` | **664** | 321 | 343 | 48.3% | Held-Out Dhaka Eval Fold 0 (includes khilkhet & night4) |
| `dhaka_fold1_train_matched` | **749** | 444 | 305 | 59.3% | Matches `ctg_fold1_train` (749) |
| `dhaka_fold1_eval` | **571** | 311 | 260 | 54.5% | Held-Out Dhaka Eval Fold 1 |
| `dhaka_fold2_train_matched` | **744** | 442 | 302 | 59.4% | Matches `ctg_fold2_train` (744) |
| `dhaka_fold2_eval` | **580** | 316 | 264 | 54.5% | Held-Out Dhaka Eval Fold 2 |
| `dhaka_buffers_excluded` | **126** | — | — | — | Buffer frames dropped at boundary cuts |

---

## 3. Robustness Splits (Leave-One-Day-Sequence-Out)

To verify that findings are not an artifact of temporal block partitioning, two leave-one-sequence-out models are trained on the two daytime sequences alongside the night sequence:

### Subsampling Analysis for Robustness Folds
- Total Chattogram data: `night1` (695 night), `bohoddarhat1` (275 day), `bohoddarhat2` (213 day).
- When holding out a daytime sequence, the remaining day data is limited to the single remaining day corridor (213 frames for Fold A, 275 frames for Fold B).

| Split Name | Held-Out Sequence | Subsampling Strategy | Total Images | Night Images | Day Images | Night Share |
|---|---|---|---|---|---|---|
| **Robustness Fold A** | `chittagong_bohoddarhat1` (275) | **Primary (Size-Matched)** | **749** | 536 | 213 (all day) | **71.6%** |
| *(Alternative A)* | `chittagong_bohoddarhat1` (275) | Ratio-Matched (59.3%) | 523 | 310 | 213 (all day) | 59.3% |
| **Robustness Fold B** | `chittagong_bohoddarhat2` (213) | **Primary (Size-Matched)** | **749** | 474 | 275 (all day) | **63.3%** |
| *(Alternative B)* | `chittagong_bohoddarhat2` (213) | Ratio-Matched (59.3%) | 676 | 401 | 275 (all day) | 59.3% |

*Primary Protocol:* Both robustness training sets are subsampled to exactly **749 images** (matching primary fold training size) with seed=42.

---

## 4. Training Protocol & Determinism Safeguards

1. **Epochs & Checkpoint Selection:**
   - **Fixed 100 Epochs** across all 8 training runs.
   - **No early stopping:** `patience=0` in YOLO training args.
   - **No model selection on evaluation data:** `last.pt` is strictly used for ALL evaluation. `best.pt` is ignored/deleted.
2. **Validation Monitoring Choice:**
   - The YAML `val:` points strictly to a small **10% monitoring slice (~75 images)** sampled *strictly from that run's own training set* (e.g., `_train_monitor.txt`).
   - This prevents validation crashes in Ultralytics YOLO while guaranteeing **zero exposure or leakage** of any test/evaluation split during training.
3. **Determinism & Hyperparameters:**
   - Exact pinned Ultralytics version: `ultralytics==8.4.155`.
   - `seed = 42`, `deterministic = True`.
   - Pinned hyperparameters logged in `configs/hyp.yaml`:
     - Architecture: `yolov8s.pt` (COCO pretrained)
     - Input resolution: `imgsz = 640`
     - Batch size: `batch = 16`
     - Optimizer: SGD (lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005)
     - Default augmentations: mosaic=1.0, fliplr=0.5.

---

## 5. Formal Evaluation Mapping

| Experiment / Evaluation | Evaluated Model(s) | Evaluation Target Split | Total Images | Metric Reporting & Aggregation |
|---|---|---|---|---|
| **E1: Cross-City Transfer (Dhaka $\rightarrow$ CTG)** | Dhaka Fold Models (E1.0, E1.1, E1.2) | `ctg_pooled_eval` | **1,121** | Mean mAP/Recall over the 3 models + 95% bootstrap CI. Stratified by All / Day / Night. |
| **E2: In-Domain CTG Benchmark** | CTG Fold Models (E2.0, E2.1, E2.2) | Out-of-fold pooled across `ctg_fold{k}_eval` | **1,121** | Out-of-fold pooled predictions over the exact same 1,121 images. Stratified by All / Day / Night. |
| **Dhaka In-Domain Benchmark** | Dhaka Fold Models (E1.0, E1.1, E1.2) | Corresponding `dhaka_fold{k}_eval` | 664 / 571 / 580 | In-domain baseline performance on native Dhaka test corridors. |
| **Reverse Transfer (CTG $\rightarrow$ Dhaka)** | CTG Fold Models (E2.0, E2.1, E2.2) | Dhaka eval folds (inference only) | 664 / 571 / 580 | Evaluates bidirectional transfer asymmetry. |
| **Robustness Check A** | Robustness Model A & all 3 Dhaka models | Held-out `chittagong_bohoddarhat1` | **275** | Corridor generalization without temporal block cuts. Day only. |
| **Robustness Check B** | Robustness Model B & all 3 Dhaka models | Held-out `chittagong_bohoddarhat2` | **213** | Corridor generalization without temporal block cuts. Day only. |

*Critical Guarantee:* E1 and E2 evaluate on the **identical 1,121 Chattogram images** (665 night, 456 day).
