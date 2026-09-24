import os
import sys
import glob
from collections import defaultdict, Counter
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "zenodo", "badodd"))

DISTRICTS = ['chittagong', 'dhaka', 'sylhet', 'rajshahi', 'mymensingh', 'maowa', 'sirajganj', 'sherpur', 'khulna']

CLASSES = [
    'auto_rickshaw', 'bicycle', 'bus', 'car', 'cart_vehicle', 
    'construction_vehicle', 'motorbike', 'person', 'priority_vehicle', 
    'three_wheeler', 'train', 'truck', 'wheelchair'
]

HEADLINE_CLASSES = ['auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck']

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

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist yet.")
        sys.exit(1)

    print("Locating label files in Zenodo dataset...")
    label_files = glob.glob(os.path.join(DATA_DIR, "**", "labels", "**", "*.txt"), recursive=True)
    label_files = [f for f in label_files if not os.path.basename(f).startswith('._')]
    print(f"Total label files found: {len(label_files)}")

    # Parse per-sequence data
    # seq_data[dist][seq] = { 'images': list_of_filenames, 'classes': Counter() }
    seq_data = defaultdict(lambda: defaultdict(lambda: {'images': [], 'classes': Counter()}))

    for lf in label_files:
        fname = os.path.basename(lf)
        dist = get_district(fname)
        seq = get_sequence(fname)

        seq_data[dist][seq]['images'].append(fname)

        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cname = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
                    seq_data[dist][seq]['classes'][cname] += 1

    # 1. District Summary
    print("\n=======================================================")
    print("           DISTRICT OVERVIEW (ZENODO FULL)             ")
    print("=======================================================")
    for dist in sorted(DISTRICTS):
        seqs = seq_data[dist]
        tot_imgs = sum(len(d['images']) for d in seqs.values())
        print(f"{dist.capitalize():15s}: {tot_imgs:5d} images across {len(seqs):3d} sequences")

    # 2. Detailed Breakdown for Chattogram
    print("\n=======================================================")
    print("           CHATTOGRAM SEQUENCE BREAKDOWN               ")
    print("=======================================================")
    ctg_seqs = seq_data['chittagong']
    tot_ctg_imgs = sum(len(d['images']) for d in ctg_seqs.values())
    print(f"Chattogram Total: {tot_ctg_imgs} images across {len(ctg_seqs)} sequences\n")

    ctg_rows = []
    for s, data in sorted(ctg_seqs.items(), key=lambda x: len(x[1]['images']), reverse=True):
        row = {'Sequence': s, 'Images': len(data['images'])}
        for c in CLASSES:
            row[c] = data['classes'][c]
        ctg_rows.append(row)

    df_ctg = pd.DataFrame(ctg_rows)
    print(df_ctg[['Sequence', 'Images'] + HEADLINE_CLASSES].to_string(index=False))

    # Check class concentration in Chattogram
    print("\n--- Class Concentration Check in Chattogram ---")
    ctg_totals = df_ctg[CLASSES].sum()
    for c in HEADLINE_CLASSES:
        tot = ctg_totals[c]
        top_seq = df_ctg.sort_values(by=c, ascending=False).iloc[0]
        top_seq_2 = df_ctg.sort_values(by=c, ascending=False).iloc[1] if len(df_ctg) > 1 else None
        top1_pct = (top_seq[c] / tot * 100) if tot > 0 else 0
        top2_pct = ((top_seq[c] + (top_seq_2[c] if top_seq_2 is not None else 0)) / tot * 100) if tot > 0 else 0
        concentrated = " [CONCENTRATED in top 2]" if top2_pct > 70 else ""
        print(f"  {c:15s}: Total={tot:4d} | Top seq ({top_seq['Sequence']})={top_seq[c]} ({top1_pct:.1f}%) | Top 2={top2_pct:.1f}%{concentrated}")

    # 3. Detailed Breakdown for Dhaka
    print("\n=======================================================")
    print("              DHAKA SEQUENCE BREAKDOWN                 ")
    print("=======================================================")
    dhaka_seqs = seq_data['dhaka']
    tot_dhaka_imgs = sum(len(d['images']) for d in dhaka_seqs.values())
    print(f"Dhaka Total: {tot_dhaka_imgs} images across {len(dhaka_seqs)} sequences\n")

    dhaka_rows = []
    for s, data in sorted(dhaka_seqs.items(), key=lambda x: len(x[1]['images']), reverse=True):
        row = {'Sequence': s, 'Images': len(data['images'])}
        for c in CLASSES:
            row[c] = data['classes'][c]
        dhaka_rows.append(row)

    df_dhaka = pd.DataFrame(dhaka_rows)
    print(df_dhaka[['Sequence', 'Images'] + HEADLINE_CLASSES].to_string(index=False))

    # 4. Propose 3-Fold Cross-Validation for Chattogram
    # Greedy partition by balancing total images and headline class counts
    print("\n=======================================================")
    print("      PROPOSED 3-FOLD SEQUENCE PARTITION FOR CHATTOGRAM ")
    print("=======================================================")
    
    # Sort sequences by image count descending
    sorted_seq_names = df_ctg['Sequence'].tolist()
    
    # Simple multi-objective greedy binning
    folds = {0: [], 1: [], 2: []}
    fold_img_counts = [0, 0, 0]
    
    for s in sorted_seq_names:
        n_img = len(ctg_seqs[s]['images'])
        # assign to fold with fewest images
        best_fold = min(range(3), key=lambda i: fold_img_counts[i])
        folds[best_fold].append(s)
        fold_img_counts[best_fold] += n_img

    fold_summaries = []
    for f_idx in range(3):
        f_seqs = folds[f_idx]
        f_imgs = sum(len(ctg_seqs[s]['images']) for s in f_seqs)
        f_classes = Counter()
        for s in f_seqs:
            f_classes.update(ctg_seqs[s]['classes'])
        summary = {'Fold': f"Fold {f_idx+1}", 'Sequences': len(f_seqs), 'Images': f_imgs}
        for c in HEADLINE_CLASSES:
            summary[c] = f_classes[c]
        fold_summaries.append(summary)

    df_folds = pd.DataFrame(fold_summaries)
    print(df_folds.to_string(index=False))
    
    print("\nSequence assignments per fold:")
    for f_idx in range(3):
        print(f"  Fold {f_idx+1} ({fold_img_counts[f_idx]} images): {folds[f_idx]}")

    # Fold train sizes
    print("\nE2 Fold Training Sizes (training on 2 folds, evaluating on 1 held-out fold):")
    for f_idx in range(3):
        train_imgs = sum(fold_img_counts[j] for j in range(3) if j != f_idx)
        val_imgs = fold_img_counts[f_idx]
        print(f"  Fold {f_idx+1} as test: Train on {train_imgs} images, Evaluate on {val_imgs} images")
        
    avg_e2_train_size = int(sum(fold_img_counts) * 2 / 3)
    print(f"\nTarget Dhaka training set size to match E2 fold train: {avg_e2_train_size} images")

if __name__ == "__main__":
    main()
