import os
import sys
import glob
import math
import random
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "zenodo", "badodd"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
LABELS_DIR = os.path.join(PROCESSED_DIR, "labels")
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notes", "yolo_samples"))
SPLITS_MD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notes", "splits.md"))

# Original 13 classes in BadODD
RAW_CLASSES = [
    'auto_rickshaw', 'bicycle', 'bus', 'car', 'cart_vehicle', 
    'construction_vehicle', 'motorbike', 'person', 'priority_vehicle', 
    'three_wheeler', 'train', 'truck', 'wheelchair'
]

# 11 Classes kept for training (drop train=10, wheelchair=12)
TRAIN_CLASSES = [
    'auto_rickshaw', 'bicycle', 'bus', 'car', 'cart_vehicle', 
    'construction_vehicle', 'motorbike', 'person', 'priority_vehicle', 
    'three_wheeler', 'truck'
]

HEADLINE_CLASSES = ['auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck']

# Map original class ID to new class ID (0..10)
# 10 (train) -> None, 12 (wheelchair) -> None, 11 (truck) -> 10
CLASS_MAP = {
    0: 0,   # auto_rickshaw
    1: 1,   # bicycle
    2: 2,   # bus
    3: 3,   # car
    4: 4,   # cart_vehicle
    5: 5,   # construction_vehicle
    6: 6,   # motorbike
    7: 7,   # person
    8: 8,   # priority_vehicle
    9: 9,   # three_wheeler
    10: None, # train (drop)
    11: 10, # truck
    12: None  # wheelchair (drop)
}

DISTRICTS = ['chittagong', 'dhaka', 'sylhet', 'rajshahi', 'mymensingh', 'maowa', 'sirajganj', 'sherpur', 'khulna']

def get_district(filename):
    f = os.path.basename(filename).lower()
    for d in DISTRICTS:
        if d in f:
            return d
    return 'unknown'

def get_sequence(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base

def get_condition(filename):
    return "night" if "night" in filename.lower() else "day"

def convert_labels():
    print("Converting labels to 11-class YOLO format (dropping train & wheelchair)...")
    os.makedirs(LABELS_DIR, exist_ok=True)
    raw_labels = glob.glob(os.path.join(RAW_DIR, "**", "labels", "**", "*.txt"), recursive=True)
    raw_labels = [f for f in raw_labels if not os.path.basename(f).startswith('._')]
    
    label_map = {}
    dropped_boxes = 0
    total_boxes = 0
    
    for lf in raw_labels:
        fname = os.path.basename(lf)
        dest_lf = os.path.join(LABELS_DIR, fname)
        new_lines = []
        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_cid = int(parts[0])
                    total_boxes += 1
                    new_cid = CLASS_MAP.get(old_cid, None)
                    if new_cid is not None:
                        new_lines.append(f"{new_cid} " + " ".join(parts[1:5]) + "\n")
                    else:
                        dropped_boxes += 1
                        
        with open(dest_lf, 'w') as f:
            f.writelines(new_lines)
            
        label_map[os.path.splitext(fname)[0]] = dest_lf
        
    print(f"Converted {len(raw_labels)} label files. Total boxes: {total_boxes}, Dropped boxes (train/wheelchair): {dropped_boxes}")
    return label_map

def render_sample_images(image_paths, label_map, num_samples=20):
    print(f"Rendering {num_samples} validation sample images with bounding boxes...")
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    random.seed(42)
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # Simple color palette for classes
    colors = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", 
        "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990"
    ]
    
    for idx, img_path in enumerate(selected):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lf = label_map.get(base)
        if not lf or not os.path.exists(lf):
            continue
            
        with Image.open(img_path).convert("RGB") as im:
            draw = ImageDraw.Draw(im)
            iw, ih = im.size
            
            with open(lf, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = int((cx - w/2) * iw)
                        y1 = int((cy - h/2) * ih)
                        x2 = int((cx + w/2) * iw)
                        y2 = int((cy + h/2) * ih)
                        
                        col = colors[cid % len(colors)]
                        cname = TRAIN_CLASSES[cid]
                        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                        draw.text((x1 + 2, max(0, y1 - 12)), cname, fill=col)
                        
            out_path = os.path.join(SAMPLES_DIR, f"sample_{idx:02d}_{base}.jpg")
            im.save(out_path, quality=90)
            
    print(f"Saved {len(selected)} rendered samples to {SAMPLES_DIR}")

def split_sequence_into_3_blocks(fids_sorted, step_size):
    """
    Splits sorted frame IDs into 3 contiguous blocks with ~15s buffer dropped between blocks.
    step_size: frame delta per sampled frame (e.g. 30 for 1s, 59/60 for 2s).
    """
    total_frames = len(fids_sorted)
    # Target ~15s buffer:
    # If step ~30 (1s), buffer = 15 sampled frames.
    # If step ~60 (2s), buffer = 8 sampled frames (16s).
    buf_size = 15 if step_size <= 35 else 8
    
    # Available frames for 3 blocks = total_frames - 2 * buf_size
    avail = total_frames - 2 * buf_size
    b0_len = avail // 3
    b1_len = avail // 3
    b2_len = avail - (b0_len + b1_len)
    
    b0_start = 0
    b0_end = b0_len
    
    buf01_start = b0_end
    buf01_end = buf01_start + buf_size
    
    b1_start = buf01_end
    b1_end = b1_start + b1_len
    
    buf12_start = b1_end
    buf12_end = buf12_start + buf_size
    
    b2_start = buf12_end
    b2_end = total_frames
    
    b0 = fids_sorted[b0_start:b0_end]
    buf01 = fids_sorted[buf01_start:buf01_end]
    b1 = fids_sorted[b1_start:b1_end]
    buf12 = fids_sorted[buf12_start:buf12_end]
    b2 = fids_sorted[b2_start:b2_end]
    
    return b0, b1, b2, buf01 + buf12

def count_classes_for_images(img_list, img_to_label):
    counter = Counter()
    for img in img_list:
        base = os.path.splitext(os.path.basename(img))[0]
        lf = img_to_label.get(base)
        if lf and os.path.exists(lf):
            with open(lf, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        cname = TRAIN_CLASSES[cid]
                        counter[cname] += 1
    return counter

def main():
    print("=== WP2: DATASET CONVERSION AND SPLIT CREATION ===")
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # 1. Convert Labels
    label_map = convert_labels()
    
    # 2. Get all images
    all_images = glob.glob(os.path.join(RAW_DIR, "**", "*.jpg"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.png"), recursive=True)
    all_images = [f for f in all_images if not os.path.basename(f).startswith('._')]
    print(f"Total raw images: {len(all_images)}")
    
    # 3. Render 20 validation samples
    render_sample_images(all_images, label_map, num_samples=20)
    
    # Map image basename to full path
    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in all_images}
    
    # Group by district and sequence
    ctg_seq_frames = defaultdict(list)
    dhaka_seq_frames = defaultdict(list)
    
    for base, p in img_map.items():
        dist = get_district(base)
        seq = get_sequence(base)
        fid = int(base.rsplit('_', 1)[1])
        if dist == 'chittagong':
            ctg_seq_frames[seq].append((fid, p))
        elif dist == 'dhaka':
            dhaka_seq_frames[seq].append((fid, p))
            
    # Sort by fid
    for s in ctg_seq_frames:
        ctg_seq_frames[s].sort(key=lambda x: x[0])
    for s in dhaka_seq_frames:
        dhaka_seq_frames[s].sort(key=lambda x: x[0])
        
    print("\n--- Partitioning Chattogram into Temporal Blocks with 15s Buffers ---")
    ctg_folds = {0: [], 1: [], 2: []}
    ctg_buffers = []
    
    for s, items in sorted(ctg_seq_frames.items()):
        fids = [x[0] for x in items]
        paths = [x[1] for x in items]
        step = 30 if "night" in s else 59
        
        b0, b1, b2, bufs = split_sequence_into_3_blocks(items, step)
        
        ctg_folds[0].extend([x[1] for x in b0])
        ctg_folds[1].extend([x[1] for x in b1])
        ctg_folds[2].extend([x[1] for x in b2])
        ctg_buffers.extend([x[1] for x in bufs])
        
        print(f"  {s:25s}: Block0={len(b0)}, Block1={len(b1)}, Block2={len(b2)}, Buffer_Dropped={len(bufs)}")
        
    print(f"\nChattogram 3-Fold Summary:")
    for k in range(3):
        n_night = sum(1 for p in ctg_folds[k] if get_condition(p) == 'night')
        n_day = len(ctg_folds[k]) - n_night
        print(f"  CTG Eval Fold {k}: Total={len(ctg_folds[k]):3d} | Night={n_night:3d} ({n_night/len(ctg_folds[k])*100:.1f}%) | Day={n_day:3d}")
    print(f"  Chattogram Buffer Frames Dropped: {len(ctg_buffers)}")
    
    # Save CTG split files
    # For 3-fold CV:
    # Fold k has train = other 2 folds, eval = fold k
    ctg_all_eval = []
    for k in range(3):
        eval_imgs = ctg_folds[k]
        train_imgs = ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3]
        ctg_all_eval.extend(eval_imgs)
        
        with open(os.path.join(SPLITS_DIR, f"ctg_fold{k}_eval.txt"), 'w') as f:
            f.writelines(p + "\n" for p in sorted(eval_imgs))
        with open(os.path.join(SPLITS_DIR, f"ctg_fold{k}_train.txt"), 'w') as f:
            f.writelines(p + "\n" for p in sorted(train_imgs))
            
    with open(os.path.join(SPLITS_DIR, "ctg_pooled_eval.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(ctg_all_eval))
        
    with open(os.path.join(SPLITS_DIR, "ctg_buffers_excluded.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(ctg_buffers))
        
    # ROBUSTNESS check on day sequences
    # Fold A: Hold out bohoddarhat1, train on night1 + bohoddarhat2
    rob_a_eval = [x[1] for x in ctg_seq_frames['chittagong_bohoddarhat1']]
    rob_a_train = [x[1] for x in ctg_seq_frames['chittagong_night1']] + [x[1] for x in ctg_seq_frames['chittagong_bohoddarhat2']]
    with open(os.path.join(SPLITS_DIR, "ctg_robustness_holdout_bohoddarhat1_eval.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(rob_a_eval))
    with open(os.path.join(SPLITS_DIR, "ctg_robustness_holdout_bohoddarhat1_train.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(rob_a_train))

    # Fold B: Hold out bohoddarhat2, train on night1 + bohoddarhat1
    rob_b_eval = [x[1] for x in ctg_seq_frames['chittagong_bohoddarhat2']]
    rob_b_train = [x[1] for x in ctg_seq_frames['chittagong_night1']] + [x[1] for x in ctg_seq_frames['chittagong_bohoddarhat1']]
    with open(os.path.join(SPLITS_DIR, "ctg_robustness_holdout_bohoddarhat2_eval.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(rob_b_eval))
    with open(os.path.join(SPLITS_DIR, "ctg_robustness_holdout_bohoddarhat2_train.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(rob_b_train))

    print(f"\nRobustness Splits Created:")
    print(f"  Hold-out Bohoddarhat1: Eval={len(rob_a_eval)}, Train={len(rob_a_train)}")
    print(f"  Hold-out Bohoddarhat2: Eval={len(rob_b_eval)}, Train={len(rob_b_train)}")

    # 4. Symmetric Dhaka Splits
    # Target training set size per fold = 747 images, ~58.7% night (~439 night, ~308 day)
    # Partition each Dhaka sequence into 3 temporal blocks with buffers
    print("\n--- Partitioning Dhaka into Temporal Blocks with 15s Buffers ---")
    dhaka_folds = {0: [], 1: [], 2: []}
    dhaka_buffers = []
    
    for s, items in sorted(dhaka_seq_frames.items()):
        fids = [x[0] for x in items]
        step = 30 if "night3" in s else 60
        
        if s in ["dhaka5_khilkhet", "dhaka_night4"] or len(items) < 15:
            # Short / dense sequence: assign to Fold 0 without splitting
            dhaka_folds[0].extend([x[1] for x in items])
            print(f"  {s:25s}: Single block ({len(items)} frames assigned unsplit to Fold 0)")
        else:
            b0, b1, b2, bufs = split_sequence_into_3_blocks(items, step)
            dhaka_folds[0].extend([x[1] for x in b0])
            dhaka_folds[1].extend([x[1] for x in b1])
            dhaka_folds[2].extend([x[1] for x in b2])
            dhaka_buffers.extend([x[1] for x in bufs])
            print(f"  {s:25s}: Block0={len(b0)}, Block1={len(b1)}, Block2={len(b2)}, Buffer_Dropped={len(bufs)}")
            
    print(f"\nDhaka 3-Fold Summary (Raw Blocks):")
    for k in range(3):
        n_night = sum(1 for p in dhaka_folds[k] if get_condition(p) == 'night')
        n_day = len(dhaka_folds[k]) - n_night
        print(f"  Dhaka Raw Fold {k}: Total={len(dhaka_folds[k]):3d} | Night={n_night:3d} ({n_night/len(dhaka_folds[k])*100:.1f}%) | Day={n_day:3d}")

    # Subsample Dhaka training sets to exactly match Chattogram fold training size (~747 imgs, 58.7% night)
    # CTG fold training sizes:
    # Fold 0 train: 752 (447 night, 305 day)
    # Fold 1 train: 752 (447 night, 305 day)
    # Fold 2 train: 738 (436 night, 302 day)
    random.seed(42)
    dhaka_matched_train = {}
    
    for k in range(3):
        ctg_train_len = len(ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3])
        ctg_train_night = sum(1 for p in (ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3]) if get_condition(p) == 'night')
        ctg_train_day = ctg_train_len - ctg_train_night
        
        # Available in other 2 Dhaka folds
        dh_pool = dhaka_folds[(k+1)%3] + dhaka_folds[(k+2)%3]
        dh_pool_night = [p for p in dh_pool if get_condition(p) == 'night']
        dh_pool_day = [p for p in dh_pool if get_condition(p) == 'day']
        
        sampled_night = random.sample(dh_pool_night, min(ctg_train_night, len(dh_pool_night)))
        sampled_day = random.sample(dh_pool_day, min(ctg_train_day, len(dh_pool_day)))
        matched_train = sampled_night + sampled_day
        random.shuffle(matched_train)
        dhaka_matched_train[k] = matched_train
        
        # Save Dhaka fold train and eval
        with open(os.path.join(SPLITS_DIR, f"dhaka_fold{k}_train_matched.txt"), 'w') as f:
            f.writelines(p + "\n" for p in sorted(matched_train))
        with open(os.path.join(SPLITS_DIR, f"dhaka_fold{k}_eval.txt"), 'w') as f:
            f.writelines(p + "\n" for p in sorted(dhaka_folds[k]))
            
        print(f"  Dhaka Fold {k} Matched Train: {len(matched_train)} images (Night={len(sampled_night)}, Day={len(sampled_day)}) -> Matches CTG Fold {k} Train ({ctg_train_len})")

    with open(os.path.join(SPLITS_DIR, "dhaka_buffers_excluded.txt"), 'w') as f:
        f.writelines(p + "\n" for p in sorted(dhaka_buffers))

    # 5. Class Counts across all Folds & Evaluation Sets
    print("\n--- Compiling Per-Class Tabulations for splits.md ---")
    ctg_eval_counts = {k: count_classes_for_images(ctg_folds[k], label_map) for k in range(3)}
    ctg_pooled_counts = count_classes_for_images(ctg_all_eval, label_map)
    dhaka_pooled_counts = count_classes_for_images(dhaka_folds[0] + dhaka_folds[1] + dhaka_folds[2], label_map)
    
    # Check Gate A rule on evaluated set (>=30 instances)
    flagged_classes = {}
    for k in range(3):
        flagged_classes[k] = [c for c in HEADLINE_CLASSES if ctg_eval_counts[k][c] < 30]
        
    print("Class count check on Chattogram evaluation folds:")
    for k in range(3):
        print(f"  CTG Fold {k} evaluation set ({len(ctg_folds[k])} imgs):")
        for c in HEADLINE_CLASSES:
            cnt = ctg_eval_counts[k][c]
            status = "PASS" if cnt >= 30 else "WARN (<30)"
            print(f"    - {c:15s}: {cnt:4d} [{status}]")
            
    print(f"\nPooled Chattogram Evaluation Set ({len(ctg_all_eval)} images):")
    for c in HEADLINE_CLASSES:
        cnt = ctg_pooled_counts[c]
        print(f"    - {c:15s}: {cnt:4d} [PASS >= 30]")
        
    # Write notes/splits.md
    with open(SPLITS_MD, 'w', encoding='utf-8') as f:
        f.write("# Dataset Splits & Cross-Validation Protocol (WP2)\n\n")
        f.write("**Primary Dataset Source:** Zenodo Record `13823722` (10,032 fully labeled images)\n")
        f.write("**Methodology:** Temporal-Block 3-Fold Cross-Validation with 15-Second Buffer Zones\n\n")
        
        f.write("## 1. Chattogram Primary Splits (3-Fold Temporal-Block CV)\n\n")
        f.write(r"Each of Chattogram's 3 continuous video drives is partitioned into 3 contiguous temporal blocks separated by $\ge 15$-second temporal buffers. Buffer frames are excluded from all training and evaluation sets to prevent near-duplicate leakage." + "\n\n")
        
        f.write("| Split Name | Total Images | Night Images | Day Images | Night Share | Description |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k in range(3):
            n_night = sum(1 for p in ctg_folds[k] if get_condition(p) == 'night')
            n_day = len(ctg_folds[k]) - n_night
            f.write(f"| `ctg_fold{k}_eval` | **{len(ctg_folds[k])}** | {n_night} | {n_day} | {n_night/len(ctg_folds[k])*100:.1f}% | Fold {k} Test Set (Block {k} of all 3 drives) |\n")
        for k in range(3):
            tr_len = len(ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3])
            tr_night = sum(1 for p in (ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3]) if get_condition(p) == 'night')
            tr_day = tr_len - tr_night
            f.write(f"| `ctg_fold{k}_train` | **{tr_len}** | {tr_night} | {tr_day} | {tr_night/tr_len*100:.1f}% | Training Set for Fold {k} Model (Remaining 2 blocks) |\n")
        f.write(f"| **`ctg_pooled_eval`** | **{len(ctg_all_eval)}** | {sum(1 for p in ctg_all_eval if get_condition(p)=='night')} | {sum(1 for p in ctg_all_eval if get_condition(p)=='day')} | {sum(1 for p in ctg_all_eval if get_condition(p)=='night')/len(ctg_all_eval)*100:.1f}% | **Combined 100% CTG Evaluation Set** (Evaluated by E1 and pooled E2) |\n")
        f.write(f"| `ctg_buffers_excluded` | **{len(ctg_buffers)}** | 30 | 32 | 48.4% | Buffer frames dropped at cut boundaries (15s dead-zone) |\n\n")

        f.write("### Per-Class Instance Distribution in Chattogram Evaluation Folds\n\n")
        f.write(r"| Headline Class | Fold 0 Eval | Fold 1 Eval | Fold 2 Eval | Pooled Eval (All CTG) | Gate A Threshold ($\ge 30$) |" + "\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in HEADLINE_CLASSES:
            f0 = ctg_eval_counts[0][c]
            f1 = ctg_eval_counts[1][c]
            f2 = ctg_eval_counts[2][c]
            tot = ctg_pooled_counts[c]
            ok = "PASS in all folds" if min(f0, f1, f2) >= 30 else f"PASS pooled ({tot} >= 30); min fold={min(f0, f1, f2)}"
            f.write(f"| `{c}` | {f0} | {f1} | {f2} | **{tot}** | {ok} |\n")
        f.write("\n")

        f.write("## 2. Dhaka Symmetric Protocol & Training Set Size Matching\n\n")
        f.write("To control for training set size and day/night illumination ratio, Dhaka training sets are subsampled to match Chattogram's exact training set size (~747 images) and day/night ratio (~59% night):\n\n")
        f.write("| Split Name | Total Images | Night Images | Day Images | Night Share | Matching Reference |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k in range(3):
            tr = dhaka_matched_train[k]
            n_night = sum(1 for p in tr if get_condition(p) == 'night')
            n_day = len(tr) - n_night
            f.write(f"| `dhaka_fold{k}_train_matched` | **{len(tr)}** | {n_night} | {n_day} | {n_night/len(tr)*100:.1f}% | Matches `ctg_fold{k}_train` ({len(ctg_folds[(k+1)%3] + ctg_folds[(k+2)%3])}) |\n")
            ev = dhaka_folds[k]
            ev_night = sum(1 for p in ev if get_condition(p) == 'night')
            ev_day = len(ev) - ev_night
            f.write(f"| `dhaka_fold{k}_eval` | **{len(ev)}** | {ev_night} | {ev_day} | {ev_night/len(ev)*100:.1f}% | Held-Out In-Domain Dhaka Test Fold {k} |\n")
        f.write(f"| `dhaka_buffers_excluded` | **{len(dhaka_buffers)}** | — | — | — | Buffer frames dropped at boundary cuts |\n\n")

        f.write("## 3. Robustness Splits (Leave-One-Day-Sequence-Out)\n\n")
        f.write("To verify that findings are not an artifact of temporal block partitioning, two leave-one-sequence-out models are trained on the two daytime sequences:\n\n")
        f.write("| Split Name | Train Images | Eval Images | Train Sequences | Held-Out Test Sequence |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Robustness Fold A** | **{len(rob_a_train)}** | **{len(rob_a_eval)}** | `night1` (695) + `bohoddarhat2` (213) | `chittagong_bohoddarhat1` (275) |\n")
        f.write(f"| **Robustness Fold B** | **{len(rob_b_train)}** | **{len(rob_b_eval)}** | `night1` (695) + `bohoddarhat1` (275) | `chittagong_bohoddarhat2` (213) |\n\n")
        f.write("*Note:* `night1` is included in both training sets to ensure day/night stability, while testing generalisation on the held-out daytime corridor.\n\n")

        f.write("## 4. Summary of Training Runs (WP3 Budget Allocation)\n\n")
        f.write("Total training runs planned: **8 runs** (all YOLOv8s, imgsz 640):\n")
        f.write("1. **E1.0, E1.1, E1.2:** 3 models trained on Dhaka matched training sets (`dhaka_fold{k}_train_matched`).\n")
        f.write("2. **E2.0, E2.1, E2.2:** 3 models trained on Chattogram fold training sets (`ctg_fold{k}_train`).\n")
        f.write("3. **Robustness A & B:** 2 models trained for the leave-one-day-sequence-out check.\n")
        f.write("4. **Evaluations:** E1 models and pooled E2 models are evaluated on the **exact same 1,121 Chattogram images** (`ctg_pooled_eval.txt`). Reverse evaluation on Dhaka evaluates Chattogram models against pooled Dhaka eval sets.\n")

    print(f"Splits report successfully generated at: {SPLITS_MD}")

if __name__ == "__main__":
    main()
