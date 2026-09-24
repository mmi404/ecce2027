"""
YOLO26s replicate of run_wp4_point_estimates.py: same 8 runs (E1 x3, E2 x3, rob_a,
rob_b), same eval targets, only the run_id gets a _y26 suffix so it reads the YOLO26s
predictions instead of the YOLOv8s ones. Writes to *_yolo26.csv so it never touches
the original YOLOv8s result files -- this is a parallel architecture-robustness check,
not a replacement of the headline result.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')

RUNS = {
    'e1_0_y26': ('dhaka_fold0_eval', ['ctg_pooled_eval', 'dhaka_fold0_eval',
                                       'ctg_robustness_holdout_bohoddarhat1_eval',
                                       'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e1_1_y26': ('dhaka_fold1_eval', ['ctg_pooled_eval', 'dhaka_fold1_eval',
                                       'ctg_robustness_holdout_bohoddarhat1_eval',
                                       'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e1_2_y26': ('dhaka_fold2_eval', ['ctg_pooled_eval', 'dhaka_fold2_eval',
                                       'ctg_robustness_holdout_bohoddarhat1_eval',
                                       'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e2_0_y26': ('ctg_fold0_eval', ['ctg_fold0_eval', 'dhaka_fold0_eval']),
    'e2_1_y26': ('ctg_fold1_eval', ['ctg_fold1_eval', 'dhaka_fold1_eval']),
    'e2_2_y26': ('ctg_fold2_eval', ['ctg_fold2_eval', 'dhaka_fold2_eval']),
    'rob_a_y26': (None, ['ctg_robustness_holdout_bohoddarhat1_eval']),
    'rob_b_y26': (None, ['ctg_robustness_holdout_bohoddarhat2_eval']),
}

HEADLINE_IDS = [cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES]


def main():
    gt_cache = {}

    def get_gt(target):
        if target not in gt_cache:
            gt_cache[target] = w.load_gt(target)
        return gt_cache[target]

    preds_cache = {}
    for run_id, (_, targets) in RUNS.items():
        for target in targets:
            preds_cache[(run_id, target)] = w.load_preds(run_id, target)

    ap_rows = []
    for run_id, (_, targets) in RUNS.items():
        for target in targets:
            image_ids, gt_by_class = get_gt(target)
            preds = preds_cache[(run_id, target)]
            for cid, cname in sorted(w.CLASS_NAMES.items()):
                ap50, ap5095, n_gt = w.compute_ap50_ap5095(gt_by_class, preds, cid, image_ids)
                ap_rows.append({
                    'run_id': run_id, 'target': target, 'class': cname,
                    'n_gt': n_gt, 'ap50': round(ap50, 4), 'ap50_95': round(ap5095, 4),
                })
            print(f'AP done: {run_id} / {target}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'per_class_ap_yolo26.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'target', 'class', 'n_gt', 'ap50', 'ap50_95'])
        writer.writeheader()
        writer.writerows(ap_rows)
    print(f'Wrote results/per_class_ap_yolo26.csv ({len(ap_rows)} rows)')

    thresholds = {}
    thr_rows = []
    for run_id, (in_domain, _) in RUNS.items():
        if in_domain is None:
            continue
        image_ids, gt_by_class = get_gt(in_domain)
        preds = preds_cache[(run_id, in_domain)]
        t, p, r, f1 = w.best_f1_threshold(gt_by_class, preds, image_ids, classes=HEADLINE_IDS)
        thresholds[run_id] = t
        thr_rows.append({'run_id': run_id, 'in_domain_target': in_domain,
                          'threshold': round(t, 3), 'precision': round(p, 4),
                          'recall': round(r, 4), 'f1': round(f1, 4)})
        print(f'Threshold {run_id}: t={t:.3f} (P={p:.3f} R={r:.3f} F1={f1:.3f}, tuned on {in_domain})')

    mean_e2_thresh = sum(thresholds[f'e2_{k}_y26'] for k in range(3)) / 3
    thresholds['rob_a_y26'] = mean_e2_thresh
    thresholds['rob_b_y26'] = mean_e2_thresh
    thr_rows.append({'run_id': 'rob_a_y26', 'in_domain_target': '(none; mean of E2 folds)',
                      'threshold': round(mean_e2_thresh, 3), 'precision': '', 'recall': '', 'f1': ''})
    thr_rows.append({'run_id': 'rob_b_y26', 'in_domain_target': '(none; mean of E2 folds)',
                      'threshold': round(mean_e2_thresh, 3), 'precision': '', 'recall': '', 'f1': ''})

    with open(os.path.join(RESULTS_DIR, 'thresholds_yolo26.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'in_domain_target', 'threshold',
                                                'precision', 'recall', 'f1'])
        writer.writeheader()
        writer.writerows(thr_rows)
    print('Wrote results/thresholds_yolo26.csv')

    recall_rows = []
    for run_id, (_, targets) in RUNS.items():
        t = thresholds[run_id]
        for target in targets:
            image_ids, gt_by_class = get_gt(target)
            preds = preds_cache[(run_id, target)]
            for cid, cname in sorted(w.CLASS_NAMES.items()):
                rec, n_gt = w.recall_at_threshold(gt_by_class, preds, cid, image_ids, t)
                recall_rows.append({
                    'run_id': run_id, 'target': target, 'class': cname,
                    'threshold': round(t, 3), 'n_gt': n_gt,
                    'recall': round(rec, 4) if rec == rec else '',
                })
            print(f'Recall done: {run_id} / {target}')

    with open(os.path.join(RESULTS_DIR, 'per_class_recall_yolo26.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'target', 'class', 'threshold', 'n_gt', 'recall'])
        writer.writeheader()
        writer.writerows(recall_rows)
    print(f'Wrote results/per_class_recall_yolo26.csv ({len(recall_rows)} rows)')


if __name__ == '__main__':
    main()
