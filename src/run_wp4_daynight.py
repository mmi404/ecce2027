"""WP4: day/night stratified headline result (E1 mean vs E2 pooled, ctg_pooled_eval),
per mode group. Day/night inferred from filename ("night" substring), matching the
sequence-level day/night labels already used throughout notes/splits.md.
"""
import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')


def main():
    thresholds = {}
    with open(os.path.join(RESULTS_DIR, 'thresholds.csv')) as f:
        for row in csv.DictReader(f):
            thresholds[row['run_id']] = float(row['threshold'])

    image_ids, gt_by_class = w.load_gt('ctg_pooled_eval')
    night_ids = [i for i in image_ids if 'night' in i]
    day_ids = [i for i in image_ids if 'night' not in i]
    print(f'ctg_pooled_eval: {len(day_ids)} day, {len(night_ids)} night (of {len(image_ids)})')

    e1_preds = [w.load_preds(f'e1_{k}', 'ctg_pooled_eval') for k in range(3)]
    e1_thresholds = [thresholds[f'e1_{k}'] for k in range(3)]
    e2_fold_preds = [w.load_preds(f'e2_{k}', f'ctg_fold{k}_eval') for k in range(3)]
    e2_pooled = w.merge_preds(e2_fold_preds)
    HEADLINE_IDS = [cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES]
    e2_thresh, *_ = w.best_f1_threshold(gt_by_class, e2_pooled, image_ids, classes=HEADLINE_IDS)

    rows = []
    for cond_name, ids in [('day', day_ids), ('night', night_ids)]:
        for cid, cname in sorted(w.CLASS_NAMES.items()):
            if cname not in w.HEADLINE_CLASSES:
                continue
            ap_a = [w.compute_ap50_ap5095(gt_by_class, p, cid, ids)[0] for p in e1_preds]
            ap_a_mean = sum(ap_a) / 3
            rec_a = [w.recall_at_threshold(gt_by_class, p, cid, ids, t)[0]
                     for p, t in zip(e1_preds, e1_thresholds)]
            rec_a_valid = [r for r in rec_a if r == r]
            rec_a_mean = sum(rec_a_valid) / len(rec_a_valid) if rec_a_valid else float('nan')

            ap_b, _, n_gt = w.compute_ap50_ap5095(gt_by_class, e2_pooled, cid, ids)
            rec_b, _ = w.recall_at_threshold(gt_by_class, e2_pooled, cid, ids, e2_thresh)

            rows.append({
                'condition': cond_name, 'class': cname, 'n_gt': n_gt,
                'e1_ap50': round(ap_a_mean, 4), 'e2_ap50': round(ap_b, 4),
                'ap50_gap': round(ap_b - ap_a_mean, 4),
                'e1_recall': round(rec_a_mean, 4) if rec_a_mean == rec_a_mean else '',
                'e2_recall': round(rec_b, 4) if rec_b == rec_b else '',
                'recall_gap': round(rec_b - rec_a_mean, 4) if (rec_b == rec_b and rec_a_mean == rec_a_mean) else '',
            })

    with open(os.path.join(RESULTS_DIR, 'day_night_stratified.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote results/day_night_stratified.csv')
    for r in rows:
        print(f"{r['condition']:6s} {r['class']:16s} n={r['n_gt']:4d} "
              f"AP50 gap={r['ap50_gap']:+.3f}  recall gap={r['recall_gap']}")

    # Also: day/night image + instance counts per city, for the Limitations section.
    dhaka_ids = {}
    for k in range(3):
        ids_k, _ = w.load_gt(f'dhaka_fold{k}_eval')
        dhaka_ids[k] = ids_k
    dhaka_all = set()
    for k in range(3):
        dhaka_all.update(dhaka_ids[k])
    dhaka_night = sum(1 for i in dhaka_all if 'night' in i)
    print(f'\nDhaka eval pooled (3 folds, may double count buffer overlaps=0 by design): '
          f'{len(dhaka_all)} images, {dhaka_night} night ({100*dhaka_night/len(dhaka_all):.1f}%)')
    print(f'CTG pooled eval: {len(image_ids)} images, {len(night_ids)} night '
          f'({100*len(night_ids)/len(image_ids):.1f}%)')


if __name__ == '__main__':
    main()
