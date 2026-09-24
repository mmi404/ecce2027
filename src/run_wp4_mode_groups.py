"""
WP4 mode-group aggregation (point estimates; CIs come from run_wp4_bootstrap.py
separately). Builds the E1 (Dhaka-trained, mean of 3 folds) vs E2 (Chattogram-trained,
out-of-fold pooled) per-class and per-mode-group comparison on ctg_pooled_eval,
weighted by each class's instance count -- this is the RQ1/RQ2 headline table and
the bridge to the thesis's mode-group framing.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')
HEADLINE_IDS = [cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES]


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
    e2_thresh, e2_p, e2_r, e2_f1 = w.best_f1_threshold(gt_by_class, e2_pooled, image_ids, classes=HEADLINE_IDS)

    per_class_rows = []
    for cid, cname in sorted(w.CLASS_NAMES.items()):
        ap_a = [w.compute_ap50_ap5095(gt_by_class, p, cid, image_ids)[0] for p in e1_preds]
        rec_a = [w.recall_at_threshold(gt_by_class, p, cid, image_ids, t)[0]
                  for p, t in zip(e1_preds, e1_thresholds)]
        ap_a_mean = sum(ap_a) / 3
        rec_a_mean = sum(r for r in rec_a if r == r) / len([r for r in rec_a if r == r]) if any(r == r for r in rec_a) else float('nan')

        ap_b, _, n_gt = w.compute_ap50_ap5095(gt_by_class, e2_pooled, cid, image_ids)
        rec_b, _ = w.recall_at_threshold(gt_by_class, e2_pooled, cid, image_ids, e2_thresh)

        per_class_rows.append({
            'class': cname, 'n_gt_ctg': n_gt, 'headline': cname in w.HEADLINE_CLASSES,
            'e1_ap50_mean': round(ap_a_mean, 4), 'e2_ap50_pooled': round(ap_b, 4),
            'ap50_gap_point': round(ap_b - ap_a_mean, 4),
            'e1_recall_mean': round(rec_a_mean, 4) if rec_a_mean == rec_a_mean else '',
            'e2_recall_pooled': round(rec_b, 4) if rec_b == rec_b else '',
            'recall_gap_point': round(rec_b - rec_a_mean, 4) if (rec_b == rec_b and rec_a_mean == rec_a_mean) else '',
        })

    with open(os.path.join(RESULTS_DIR, 'headline_per_class_point.csv'), 'w', newline='') as f:
        fieldnames = list(per_class_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)
    print('Wrote results/headline_per_class_point.csv')
    for r in per_class_rows:
        print(f"{r['class']:22s} n={r['n_gt_ctg']:5d}  E1={r['e1_ap50_mean']:.3f}  "
              f"E2={r['e2_ap50_pooled']:.3f}  gap={r['ap50_gap_point']:+.3f}  "
              f"| recall E1={r['e1_recall_mean']}  E2={r['e2_recall_pooled']}  gap={r['recall_gap_point']}")

    # Mode-group aggregation, weighted by each class's CTG instance count.
    group_rows = []
    by_class = {r['class']: r for r in per_class_rows}
    for group, classes in w.MODE_GROUPS.items():
        total_n = sum(by_class[c]['n_gt_ctg'] for c in classes)
        if total_n == 0:
            continue
        ap_a = sum(by_class[c]['e1_ap50_mean'] * by_class[c]['n_gt_ctg'] for c in classes) / total_n
        ap_b = sum(by_class[c]['e2_ap50_pooled'] * by_class[c]['n_gt_ctg'] for c in classes) / total_n
        rec_terms_a = [(by_class[c]['e1_recall_mean'], by_class[c]['n_gt_ctg']) for c in classes
                        if by_class[c]['e1_recall_mean'] != '']
        rec_terms_b = [(by_class[c]['e2_recall_pooled'], by_class[c]['n_gt_ctg']) for c in classes
                        if by_class[c]['e2_recall_pooled'] != '']
        rec_a = sum(v * n for v, n in rec_terms_a) / sum(n for _, n in rec_terms_a) if rec_terms_a else ''
        rec_b = sum(v * n for v, n in rec_terms_b) / sum(n for _, n in rec_terms_b) if rec_terms_b else ''
        group_rows.append({
            'mode_group': group, 'classes': ','.join(classes), 'total_n_gt_ctg': total_n,
            'e1_ap50_mean': round(ap_a, 4), 'e2_ap50_pooled': round(ap_b, 4),
            'ap50_gap': round(ap_b - ap_a, 4),
            'e1_recall_mean': round(rec_a, 4) if rec_a != '' else '',
            'e2_recall_pooled': round(rec_b, 4) if rec_b != '' else '',
            'recall_gap': round(rec_b - rec_a, 4) if (rec_a != '' and rec_b != '') else '',
        })

    with open(os.path.join(RESULTS_DIR, 'mode_group_summary.csv'), 'w', newline='') as f:
        fieldnames = list(group_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_rows)
    print('\nWrote results/mode_group_summary.csv')
    for r in group_rows:
        print(f"{r['mode_group']:22s} n={r['total_n_gt_ctg']:5d}  "
              f"AP50 E1={r['e1_ap50_mean']:.3f} E2={r['e2_ap50_pooled']:.3f} gap={r['ap50_gap']:+.3f}  "
              f"| Recall E1={r['e1_recall_mean']} E2={r['e2_recall_pooled']} gap={r['recall_gap']}")


if __name__ == '__main__':
    main()
