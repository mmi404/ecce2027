"""
WP4: mode-group-level bootstrap CI. Uses ONE shared image resample per iteration
across all classes (so the per-class draws inside a group are correlated the way a
real resample of the underlying 1,121 images would be), aggregates per-class AP50/
recall into each mode group with the same instance-count weights used in
run_wp4_mode_groups.py, and reports the 95% CI on each group's AP50 gap and recall gap.
"""
import sys, os, csv, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')
N_BOOT = 300
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
    e2_thresh = json.load(open(os.path.join(RESULTS_DIR, 'logs', 'e2_pooled_threshold.json')))['threshold']

    # Fixed weights = each class's real (non-resampled) CTG instance count.
    weights = {}
    for cid, cname in w.CLASS_NAMES.items():
        _, _, n_gt = w.compute_ap50_ap5095(gt_by_class, e2_pooled, cid, image_ids)
        weights[cname] = n_gt

    rng = np.random.default_rng(123)
    n = len(image_ids)
    image_ids_arr = np.array(image_ids, dtype=object)

    group_ap_gaps = {g: [] for g in w.MODE_GROUPS}
    group_rec_gaps = {g: [] for g in w.MODE_GROUPS}

    def eval_class(cid, preds, ids, conf_thresh):
        gp, pp = w.slots_from_resample(gt_by_class, preds, cid, ids)
        scores, is_tp, total_gt = w._greedy_match_curve(gp, pp, 0.5)
        ap = w._ap_from_curve(is_tp, total_gt)
        if total_gt == 0:
            rec = float('nan')
        else:
            keep = scores >= conf_thresh
            rec = float(is_tp[keep].sum()) / total_gt
        return ap, rec

    print(f'Running {N_BOOT} shared resamples over 11 classes x 4 models...')
    for i in range(N_BOOT):
        resample = image_ids_arr[rng.integers(0, n, n)]
        ap_a = {}
        rec_a = {}
        ap_b = {}
        rec_b = {}
        for cid, cname in w.CLASS_NAMES.items():
            a_vals = [eval_class(cid, p, resample, t) for p, t in zip(e1_preds, e1_thresholds)]
            ap_a[cname] = np.nanmean([v[0] for v in a_vals])
            rec_a[cname] = np.nanmean([v[1] for v in a_vals])
            ap_b[cname], rec_b[cname] = eval_class(cid, e2_pooled, resample, e2_thresh)

        for g, classes in w.MODE_GROUPS.items():
            wsum = sum(weights[c] for c in classes)
            if wsum == 0:
                continue
            ga = sum(ap_a[c] * weights[c] for c in classes) / wsum
            gb = sum(ap_b[c] * weights[c] for c in classes) / wsum
            group_ap_gaps[g].append(gb - ga)
            ra = sum(rec_a[c] * weights[c] for c in classes) / wsum
            rb = sum(rec_b[c] * weights[c] for c in classes) / wsum
            group_rec_gaps[g].append(rb - ra)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{N_BOOT} resamples done')

    rows = []
    for g in w.MODE_GROUPS:
        ap_arr = np.array(group_ap_gaps[g])
        rec_arr = np.array([x for x in group_rec_gaps[g] if x == x])
        ap_lo, ap_hi = np.percentile(ap_arr, [2.5, 97.5])
        rec_lo, rec_hi = np.percentile(rec_arr, [2.5, 97.5]) if len(rec_arr) > 0 else (float('nan'), float('nan'))
        rows.append({
            'mode_group': g,
            'ap50_gap_ci_lo': round(float(ap_lo), 4), 'ap50_gap_ci_hi': round(float(ap_hi), 4),
            'recall_gap_ci_lo': round(float(rec_lo), 4), 'recall_gap_ci_hi': round(float(rec_hi), 4),
        })
        print(f'{g:22s} AP50 gap CI [{ap_lo:.3f}, {ap_hi:.3f}]  Recall gap CI [{rec_lo:.3f}, {rec_hi:.3f}]')

    with open(os.path.join(RESULTS_DIR, 'mode_group_ci.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote results/mode_group_ci.csv')


if __name__ == '__main__':
    main()
