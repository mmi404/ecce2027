"""
WP4 point estimates (no bootstrap): per-class AP50/AP50-95 for every run x target,
per-model best-F1 operating threshold (tuned on that model's own in-domain fold),
and per-class recall at that fixed threshold for every target the model covers.

Fast (~seconds); the expensive bootstrap CIs are computed separately in
run_wp4_bootstrap.py since they take much longer.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')

# run_id -> (in_domain_target_for_threshold_tuning, [all targets this run has predictions for])
RUNS = {
    'e1_0': ('dhaka_fold0_eval', ['ctg_pooled_eval', 'dhaka_fold0_eval',
                                   'ctg_robustness_holdout_bohoddarhat1_eval',
                                   'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e1_1': ('dhaka_fold1_eval', ['ctg_pooled_eval', 'dhaka_fold1_eval',
                                   'ctg_robustness_holdout_bohoddarhat1_eval',
                                   'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e1_2': ('dhaka_fold2_eval', ['ctg_pooled_eval', 'dhaka_fold2_eval',
                                   'ctg_robustness_holdout_bohoddarhat1_eval',
                                   'ctg_robustness_holdout_bohoddarhat2_eval']),
    'e2_0': ('ctg_fold0_eval', ['ctg_fold0_eval', 'dhaka_fold0_eval']),
    'e2_1': ('ctg_fold1_eval', ['ctg_fold1_eval', 'dhaka_fold1_eval']),
    'e2_2': ('ctg_fold2_eval', ['ctg_fold2_eval', 'dhaka_fold2_eval']),
    # rob_a/rob_b have no in-domain fold of their own (only ever evaluated on the
    # held-out corridor) -- threshold assigned later as the mean of the E2 fold
    # thresholds, since they share the same model family/training recipe as E2.
    'rob_a': (None, ['ctg_robustness_holdout_bohoddarhat1_eval']),
    'rob_b': (None, ['ctg_robustness_holdout_bohoddarhat2_eval']),
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

    # --- 1. Per-class AP50 / AP50-95 for every run x target ---
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
    with open(os.path.join(RESULTS_DIR, 'per_class_ap.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'target', 'class', 'n_gt', 'ap50', 'ap50_95'])
        writer.writeheader()
        writer.writerows(ap_rows)
    print(f'Wrote results/per_class_ap.csv ({len(ap_rows)} rows)')

    # --- 2. Per-model best-F1 threshold, tuned on the model's own in-domain fold ---
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

    mean_e2_thresh = sum(thresholds[f'e2_{k}'] for k in range(3)) / 3
    thresholds['rob_a'] = mean_e2_thresh
    thresholds['rob_b'] = mean_e2_thresh
    thr_rows.append({'run_id': 'rob_a', 'in_domain_target': '(none; mean of E2 folds)',
                      'threshold': round(mean_e2_thresh, 3), 'precision': '', 'recall': '', 'f1': ''})
    thr_rows.append({'run_id': 'rob_b', 'in_domain_target': '(none; mean of E2 folds)',
                      'threshold': round(mean_e2_thresh, 3), 'precision': '', 'recall': '', 'f1': ''})

    with open(os.path.join(RESULTS_DIR, 'thresholds.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'in_domain_target', 'threshold',
                                                'precision', 'recall', 'f1'])
        writer.writeheader()
        writer.writerows(thr_rows)
    print('Wrote results/thresholds.csv')

    # --- 3. Per-class recall at each model's fixed threshold, for every target it covers ---
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

    with open(os.path.join(RESULTS_DIR, 'per_class_recall.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'target', 'class', 'threshold', 'n_gt', 'recall'])
        writer.writeheader()
        writer.writerows(recall_rows)
    print(f'Wrote results/per_class_recall.csv ({len(recall_rows)} rows)')


if __name__ == '__main__':
    main()
