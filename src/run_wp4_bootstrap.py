"""
WP4 headline result: E1 (Dhaka-trained, mean of 3 folds) vs. E2 (Chattogram-trained,
out-of-fold pooled) on the shared 1,121-image ctg_pooled_eval set, per class, with a
1,000-resample image-level bootstrap 95% CI on both the AP50 gap and the
recall-at-fixed-threshold gap (RQ1/RQ2's core evidence). This is the expensive part
of WP4 (tens of minutes) -- point estimates in run_wp4_point_estimates.py are separate
and fast.
"""
import sys
import os
import csv
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')
N_BOOT = 300  # ~25-40 min total for all 11 classes; 95% CI edges stable to ~1-2 pts at this n


def main():
    thresholds = {}
    with open(os.path.join(RESULTS_DIR, 'thresholds.csv')) as f:
        for row in csv.DictReader(f):
            thresholds[row['run_id']] = float(row['threshold'])

    image_ids, gt_by_class = w.load_gt('ctg_pooled_eval')
    e1_preds = [w.load_preds(f'e1_{k}', 'ctg_pooled_eval') for k in range(3)]
    e1_thresholds = [thresholds[f'e1_{k}'] for k in range(3)]

    e2_fold_preds = [w.load_preds(f'e2_{k}', f'ctg_fold{k}_eval') for k in range(3)]
    e2_pooled = w.merge_preds(e2_fold_preds)
    # E2's own operating threshold: tuned directly on its pooled out-of-fold predictions
    # against the full ctg_pooled_eval GT (this is a valid in-domain measure for E2,
    # since every prediction in it comes from a fold that never saw that image in training).
    HEADLINE_IDS = [cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES]
    e2_thresh, e2_p, e2_r, e2_f1 = w.best_f1_threshold(gt_by_class, e2_pooled, image_ids, classes=HEADLINE_IDS)
    print(f'E2 pooled threshold: {e2_thresh:.3f} (P={e2_p:.3f} R={e2_r:.3f} F1={e2_f1:.3f})')

    rows = []
    t_start = time.time()
    for cid, cname in sorted(w.CLASS_NAMES.items()):
        t0 = time.time()
        res = w.bootstrap_gap_full(
            gt_by_class, e1_preds, e2_pooled, cid, image_ids,
            conf_thresh_a_list=e1_thresholds, conf_thresh_b=e2_thresh,
            n_boot=N_BOOT, seed=42,
        )
        elapsed = time.time() - t0
        row = {'class': cname, **{k: round(v, 4) if v == v else '' for k, v in res.items()}}
        rows.append(row)
        print(f'[{time.time()-t_start:7.1f}s total] {cname:22s} '
              f'AP50 gap={row["ap50_gap"]:+.4f} [{row["ap50_gap_ci_lo"]},{row["ap50_gap_ci_hi"]}]  '
              f'Recall gap={row["recall_gap"]:+.4f} [{row["recall_gap_ci_lo"]},{row["recall_gap_ci_hi"]}]  '
              f'({elapsed:.1f}s)')
        # Write incrementally so partial progress survives an interruption.
        with open(os.path.join(RESULTS_DIR, 'transfer_gap.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'ap50_gap', 'ap50_gap_ci_lo',
                                                     'ap50_gap_ci_hi', 'recall_gap',
                                                     'recall_gap_ci_lo', 'recall_gap_ci_hi'])
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(RESULTS_DIR, 'logs', 'e2_pooled_threshold.json'), 'w') as f:
        json.dump({'threshold': e2_thresh, 'precision': e2_p, 'recall': e2_r, 'f1': e2_f1,
                    'n_boot': N_BOOT}, f, indent=2)

    print(f'\nDone in {time.time()-t_start:.1f}s. Wrote results/transfer_gap.csv')


if __name__ == '__main__':
    main()
