"""
WP4 local evaluation library: recomputes detection metrics from saved COCO-format
predictions (conf=0.001) and ground truth, entirely offline (no GPU, no Kaggle).

Implements its own greedy IoU-matching COCO-style AP (101-point interpolation) rather
than calling pycocotools directly, so that image-level bootstrap resampling (with
repeated images) is tractable -- pycocotools' COCO object doesn't support duplicate
image ids cleanly, so bootstrap resamples are built as synthetic "slots" that each
point back at one real image's boxes.

Validated against Kaggle's own pycocotools-computed per_class_ap50 in
results/logs/*_eval_summary.json -- see src/validate_wp4_lib.py.
"""
import json
import os
from collections import defaultdict

import numpy as np

CLASS_NAMES = {
    0: 'auto_rickshaw', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'cart_vehicle',
    5: 'construction_vehicle', 6: 'motorbike', 7: 'person', 8: 'priority_vehicle',
    9: 'three_wheeler', 10: 'truck',
}
NUM_CLASSES = len(CLASS_NAMES)

MODE_GROUPS = {
    'pedestrian': ['person'],
    'non_motorised': ['three_wheeler', 'bicycle', 'cart_vehicle'],
    'informal_motorised': ['auto_rickshaw'],
    'formal_motorised': ['car', 'bus', 'truck', 'motorbike', 'priority_vehicle', 'construction_vehicle'],
}
HEADLINE_CLASSES = [
    'auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck',
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'coco_gt')
PRED_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')

IOU_THRESHOLDS_COCO = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def load_gt(target):
    """Returns (image_ids: list[str], gt_by_class: {cat_id: {image_id: [xywh,...]}})."""
    path = os.path.join(GT_DIR, f'{target}_coco_gt.json')
    d = json.load(open(path))
    image_ids = [img['id'] for img in d['images']]
    gt_by_class = defaultdict(lambda: defaultdict(list))
    for ann in d['annotations']:
        gt_by_class[ann['category_id']][ann['image_id']].append(ann['bbox'])
    return image_ids, gt_by_class


def load_preds(run_id, target):
    """Returns preds_by_class: {cat_id: {image_id: [(score, xywh), ...]}}."""
    path = os.path.join(PRED_DIR, f'{run_id}_{target}_preds_conf0001.json')
    d = json.load(open(path))
    preds_by_class = defaultdict(lambda: defaultdict(list))
    for p in d:
        preds_by_class[p['category_id']][p['image_id']].append((p['score'], p['bbox']))
    return preds_by_class


def merge_preds(pred_dicts):
    """Pools several run's preds_by_class dicts covering disjoint image sets
    (e.g. E2's 3 out-of-fold models) into one preds_by_class dict."""
    merged = defaultdict(lambda: defaultdict(list))
    for pd_ in pred_dicts:
        for cat_id, by_img in pd_.items():
            for img_id, plist in by_img.items():
                merged[cat_id][img_id].extend(plist)
    return merged


def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match_curve(gt_per_slot, preds_per_slot, iou_thresh):
    """Sorts all predictions by score desc, greedily matches each to the best
    unmatched GT box (IoU >= iou_thresh) in the SAME slot. Returns
    (scores_sorted_desc, is_tp_array, total_gt)."""
    total_gt = sum(len(v) for v in gt_per_slot.values())
    all_preds = []
    for slot_id, plist in preds_per_slot.items():
        for score, bbox in plist:
            all_preds.append((score, slot_id, bbox))
    all_preds.sort(key=lambda x: -x[0])

    matched = {slot_id: [False] * len(boxes) for slot_id, boxes in gt_per_slot.items()}
    is_tp = np.zeros(len(all_preds), dtype=bool)
    scores = np.zeros(len(all_preds))
    for i, (score, slot_id, bbox) in enumerate(all_preds):
        scores[i] = score
        gts = gt_per_slot.get(slot_id)
        if not gts:
            continue
        best_iou, best_j = iou_thresh, -1
        slot_matched = matched[slot_id]
        for j, gbox in enumerate(gts):
            if slot_matched[j]:
                continue
            iou = _iou_xywh(bbox, gbox)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            is_tp[i] = True
            slot_matched[best_j] = True
    return scores, is_tp, total_gt


def _ap_from_curve(is_tp, total_gt):
    if total_gt == 0:
        return float('nan')
    if len(is_tp) == 0:
        return 0.0
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(~is_tp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # COCO 101-point interpolation: precision envelope (non-increasing from the right)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side='left')
        p = precision_envelope[idx] if idx < len(precision_envelope) else 0.0
        ap += p
    return ap / 101


def slots_from_image_ids(gt_by_class, preds_by_class, cat_id, image_ids):
    """Builds gt_per_slot / preds_per_slot keyed by real image_id (no resampling)."""
    gt_per_slot = {img_id: gt_by_class.get(cat_id, {}).get(img_id, []) for img_id in image_ids}
    preds_per_slot = {img_id: preds_by_class.get(cat_id, {}).get(img_id, []) for img_id in image_ids}
    return gt_per_slot, preds_per_slot


def slots_from_resample(gt_by_class, preds_by_class, cat_id, resampled_image_ids):
    """Builds synthetic slots for a bootstrap resample where the same real image_id
    may appear multiple times -- each occurrence gets its own unique slot id so
    repeated images don't share a matched-GT pool."""
    gt_per_slot = {}
    preds_per_slot = {}
    for slot_idx, img_id in enumerate(resampled_image_ids):
        gt_per_slot[slot_idx] = gt_by_class.get(cat_id, {}).get(img_id, [])
        preds_per_slot[slot_idx] = preds_by_class.get(cat_id, {}).get(img_id, [])
    return gt_per_slot, preds_per_slot


def compute_ap50_ap5095(gt_by_class, preds_by_class, cat_id, image_ids):
    gt_per_slot, preds_per_slot = slots_from_image_ids(gt_by_class, preds_by_class, cat_id, image_ids)
    _, is_tp50, total_gt = _greedy_match_curve(gt_per_slot, preds_per_slot, 0.5)
    ap50 = _ap_from_curve(is_tp50, total_gt)
    aps = [ap50]
    for t in IOU_THRESHOLDS_COCO[1:]:
        _, is_tp_t, _ = _greedy_match_curve(gt_per_slot, preds_per_slot, t)
        aps.append(_ap_from_curve(is_tp_t, total_gt))
    ap5095 = float(np.nanmean(aps))
    return ap50, ap5095, total_gt


def pr_curve_pooled(gt_by_class, preds_by_class, image_ids, classes=None):
    """Pooled (micro-averaged) precision/recall/F1 vs. confidence threshold across
    the given classes, for choosing an operating point. Returns
    (thresholds, precision, recall, f1)."""
    if classes is None:
        classes = list(CLASS_NAMES.keys())
    all_scores = []
    all_tp = []
    total_gt = 0
    for cat_id in classes:
        gt_per_slot, preds_per_slot = slots_from_image_ids(gt_by_class, preds_by_class, cat_id, image_ids)
        scores, is_tp, gt_n = _greedy_match_curve(gt_per_slot, preds_per_slot, 0.5)
        all_scores.append(scores)
        all_tp.append(is_tp)
        total_gt += gt_n
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    is_tp = np.concatenate(all_tp) if all_tp else np.array([], dtype=bool)

    grid = np.arange(0.01, 0.96, 0.01)
    precision = np.zeros_like(grid)
    recall = np.zeros_like(grid)
    f1 = np.zeros_like(grid)
    for i, t in enumerate(grid):
        keep = scores >= t
        n_pred = keep.sum()
        n_tp = is_tp[keep].sum()
        p = n_tp / n_pred if n_pred > 0 else 0.0
        r = n_tp / total_gt if total_gt > 0 else 0.0
        precision[i], recall[i] = p, r
        f1[i] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return grid, precision, recall, f1


def best_f1_threshold(gt_by_class, preds_by_class, image_ids, classes=None):
    grid, precision, recall, f1 = pr_curve_pooled(gt_by_class, preds_by_class, image_ids, classes)
    idx = int(np.argmax(f1))
    return float(grid[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def recall_at_threshold(gt_by_class, preds_by_class, cat_id, image_ids, conf_thresh):
    gt_per_slot, preds_per_slot = slots_from_image_ids(gt_by_class, preds_by_class, cat_id, image_ids)
    scores, is_tp, total_gt = _greedy_match_curve(gt_per_slot, preds_per_slot, 0.5)
    if total_gt == 0:
        return float('nan'), 0
    keep = scores >= conf_thresh
    tp = int(is_tp[keep].sum())
    return tp / total_gt, total_gt


def bootstrap_ap50_gap(gt_by_class, preds_a_list, preds_b, cat_id, image_ids, n_boot=1000, seed=42):
    """Paired image-level bootstrap for the gap AP50(b) - mean_AP50(a_list), where
    preds_a_list is a list of prediction dicts each evaluated on the SAME image_ids
    (e.g. the 3 E1 fold models) and preds_b is a single pooled prediction dict
    (e.g. E2 out-of-fold pooled). Returns (point_gap, ci_low, ci_high, boot_gaps)."""
    rng = np.random.default_rng(seed)
    n = len(image_ids)
    image_ids_arr = np.array(image_ids, dtype=object)

    # Point estimate
    ap_a_point = [compute_ap50_ap5095(gt_by_class, pa, cat_id, image_ids)[0] for pa in preds_a_list]
    ap_a_mean_point = float(np.nanmean(ap_a_point))
    ap_b_point, _, _ = compute_ap50_ap5095(gt_by_class, preds_b, cat_id, image_ids)
    point_gap = ap_b_point - ap_a_mean_point

    boot_gaps = np.zeros(n_boot)
    for i in range(n_boot):
        resample = image_ids_arr[rng.integers(0, n, n)]
        ap_a_vals = []
        for pa in preds_a_list:
            gp, pp = slots_from_resample(gt_by_class, pa, cat_id, resample)
            _, is_tp, total_gt = _greedy_match_curve(gp, pp, 0.5)
            ap_a_vals.append(_ap_from_curve(is_tp, total_gt))
        ap_a_mean = np.nanmean(ap_a_vals)
        gp, pp = slots_from_resample(gt_by_class, preds_b, cat_id, resample)
        _, is_tp, total_gt = _greedy_match_curve(gp, pp, 0.5)
        ap_b = _ap_from_curve(is_tp, total_gt)
        boot_gaps[i] = ap_b - ap_a_mean

    valid = boot_gaps[~np.isnan(boot_gaps)]
    if len(valid) < n_boot * 0.5:
        return point_gap, float('nan'), float('nan'), boot_gaps
    ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    return point_gap, float(ci_low), float(ci_high), boot_gaps


def bootstrap_gap_full(gt_by_class, preds_a_list, preds_b, cat_id, image_ids,
                        conf_thresh_a_list, conf_thresh_b, n_boot=1000, seed=42):
    """Like bootstrap_ap50_gap, but also returns the paired bootstrap for the
    recall-at-fixed-threshold gap, computed from the same resampled match curves
    (so AP and recall bootstraps are correlated draws, not independent runs).
    Each model in preds_a_list uses its own operating threshold
    (conf_thresh_a_list[i], tuned on that model's own in-domain fold); preds_b uses
    conf_thresh_b (tuned on its own in-domain/out-of-fold evaluation)."""
    rng = np.random.default_rng(seed)
    n = len(image_ids)
    image_ids_arr = np.array(image_ids, dtype=object)

    def ap_and_recall(scores, is_tp, total_gt, conf_thresh):
        ap = _ap_from_curve(is_tp, total_gt)
        if total_gt == 0:
            rec = float('nan')
        else:
            keep = scores >= conf_thresh
            rec = float(is_tp[keep].sum()) / total_gt
        return ap, rec

    def eval_preds(preds, ids, conf_thresh):
        gp, pp = slots_from_resample(gt_by_class, preds, cat_id, ids)
        scores, is_tp, total_gt = _greedy_match_curve(gp, pp, 0.5)
        return ap_and_recall(scores, is_tp, total_gt, conf_thresh)

    ap_a_pt, rec_a_pt = zip(*(eval_preds(pa, image_ids, ta)
                               for pa, ta in zip(preds_a_list, conf_thresh_a_list)))
    ap_b_pt, rec_b_pt = eval_preds(preds_b, image_ids, conf_thresh_b)
    point_ap_gap = ap_b_pt - float(np.nanmean(ap_a_pt))
    point_rec_gap = rec_b_pt - float(np.nanmean(rec_a_pt))

    boot_ap_gaps = np.zeros(n_boot)
    boot_rec_gaps = np.zeros(n_boot)
    for i in range(n_boot):
        resample = image_ids_arr[rng.integers(0, n, n)]
        ap_a_vals, rec_a_vals = zip(*(eval_preds(pa, resample, ta)
                                       for pa, ta in zip(preds_a_list, conf_thresh_a_list)))
        ap_b, rec_b = eval_preds(preds_b, resample, conf_thresh_b)
        boot_ap_gaps[i] = ap_b - np.nanmean(ap_a_vals)
        boot_rec_gaps[i] = rec_b - np.nanmean(rec_a_vals)

    def ci(arr, pt):
        valid = arr[~np.isnan(arr)]
        if len(valid) < n_boot * 0.5:
            return pt, float('nan'), float('nan')
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return pt, float(lo), float(hi)

    ap_gap, ap_lo, ap_hi = ci(boot_ap_gaps, point_ap_gap)
    rec_gap, rec_lo, rec_hi = ci(boot_rec_gaps, point_rec_gap)
    return {
        'ap50_gap': ap_gap, 'ap50_gap_ci_lo': ap_lo, 'ap50_gap_ci_hi': ap_hi,
        'recall_gap': rec_gap, 'recall_gap_ci_lo': rec_lo, 'recall_gap_ci_hi': rec_hi,
    }


def bootstrap_single_ap50(gt_by_class, preds, cat_id, image_ids, n_boot=1000, seed=42):
    """Image-level bootstrap CI for a single model's AP50 on one class/target."""
    rng = np.random.default_rng(seed)
    n = len(image_ids)
    image_ids_arr = np.array(image_ids, dtype=object)
    point, _, _ = compute_ap50_ap5095(gt_by_class, preds, cat_id, image_ids)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        resample = image_ids_arr[rng.integers(0, n, n)]
        gp, pp = slots_from_resample(gt_by_class, preds, cat_id, resample)
        _, is_tp, total_gt = _greedy_match_curve(gp, pp, 0.5)
        boots[i] = _ap_from_curve(is_tp, total_gt)
    valid = boots[~np.isnan(boots)]
    if len(valid) < n_boot * 0.5:
        return point, float('nan'), float('nan')
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return point, float(lo), float(hi)
