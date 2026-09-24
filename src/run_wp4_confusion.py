"""
WP4: normalised confusion matrices for E1 (Dhaka-trained, summed over the 3 folds'
independent evaluations of ctg_pooled_eval) and E2 (Chattogram-trained, out-of-fold
pooled), restricted to the 7 headline classes plus a background row/col for
misses (GT with no matching prediction) and false positives (prediction with no
matching GT). Rare classes' GT boxes are excluded from this analysis, consistent
with the rest of the paper (no individual claims on them).

Matching: for each image, predictions filtered to the model's own operating
threshold, sorted by score desc, each greedily matched to the best unmatched GT box
(any class, IoU>=0.5) in that image. Matched -> (gt_class, pred_class). Unmatched
prediction -> (background, pred_class). Unmatched GT after all predictions processed
-> (gt_class, background).
"""
import sys, os, csv, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')

HEADLINE_IDS = sorted(cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES)
LABELS = [w.CLASS_NAMES[c] for c in HEADLINE_IDS] + ['background']


def load_raw_gt(target):
    path = os.path.join(w.GT_DIR, f'{target}_coco_gt.json')
    d = json.load(open(path))
    gt_by_image = defaultdict(list)
    for ann in d['annotations']:
        if ann['category_id'] in HEADLINE_IDS:
            gt_by_image[ann['image_id']].append((ann['category_id'], ann['bbox']))
    return gt_by_image


def load_raw_preds(run_id, target, conf_thresh):
    path = os.path.join(w.PRED_DIR, f'{run_id}_{target}_preds_conf0001.json')
    d = json.load(open(path))
    preds_by_image = defaultdict(list)
    for p in d:
        if p['category_id'] in HEADLINE_IDS and p['score'] >= conf_thresh:
            preds_by_image[p['image_id']].append((p['score'], p['category_id'], p['bbox']))
    for img_id in preds_by_image:
        preds_by_image[img_id].sort(key=lambda x: -x[0])
    return preds_by_image


def accumulate_confusion(gt_by_image, preds_by_image, image_ids, matrix):
    for img_id in image_ids:
        gts = list(gt_by_image.get(img_id, []))
        matched = [False] * len(gts)
        for score, pred_cls, pbox in preds_by_image.get(img_id, []):
            pred_name = w.CLASS_NAMES[pred_cls]
            best_iou, best_j = 0.5, -1
            for j, (gt_cls, gbox) in enumerate(gts):
                if matched[j]:
                    continue
                iou = w._iou_xywh(pbox, gbox)
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                gt_name = w.CLASS_NAMES[gts[best_j][0]]
                matrix[gt_name][pred_name] += 1
                matched[best_j] = True
            else:
                matrix['background'][pred_name] += 1
        for j, (gt_cls, gbox) in enumerate(gts):
            if not matched[j]:
                matrix[w.CLASS_NAMES[gt_cls]]['background'] += 1


def new_matrix():
    return {r: {c: 0 for c in LABELS} for r in LABELS}


def save_matrix(matrix, name):
    # Row-normalisation (for src/make_confusion_tables.py, which renders the paper's
    # table) excludes the background row: it has no "true" GT to normalise against.
    with open(os.path.join(RESULTS_DIR, f'confusion_{name}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + LABELS)
        for r in LABELS:
            writer.writerow([r] + [matrix[r][c] for c in LABELS])
    print(f'Wrote results/confusion_{name}.csv')


def main():
    thresholds = {}
    with open(os.path.join(RESULTS_DIR, 'thresholds.csv')) as f:
        for row in csv.DictReader(f):
            thresholds[row['run_id']] = float(row['threshold'])

    image_ids, _ = w.load_gt('ctg_pooled_eval')
    gt_by_image = load_raw_gt('ctg_pooled_eval')

    # E1: sum confusion counts over the 3 independently-trained Dhaka folds, each
    # evaluated on the full ctg_pooled_eval at its own threshold.
    e1_matrix = new_matrix()
    for k in range(3):
        preds = load_raw_preds(f'e1_{k}', 'ctg_pooled_eval', thresholds[f'e1_{k}'])
        accumulate_confusion(gt_by_image, preds, image_ids, e1_matrix)
    save_matrix(e1_matrix, 'e1')

    # E2: pooled out-of-fold predictions, one evaluation over the full 1,121 images.
    e2_preds_by_image = defaultdict(list)
    e2_thresh = {}
    with open(os.path.join(RESULTS_DIR, 'logs', 'e2_pooled_threshold.json')) as f:
        e2_thresh_val = json.load(f)['threshold']
    for k in range(3):
        target = f'ctg_fold{k}_eval'
        fold_preds = load_raw_preds(f'e2_{k}', target, e2_thresh_val)
        for img_id, plist in fold_preds.items():
            e2_preds_by_image[img_id].extend(plist)
    for img_id in e2_preds_by_image:
        e2_preds_by_image[img_id].sort(key=lambda x: -x[0])
    e2_matrix = new_matrix()
    accumulate_confusion(gt_by_image, e2_preds_by_image, image_ids, e2_matrix)
    save_matrix(e2_matrix, 'e2')


if __name__ == '__main__':
    main()
