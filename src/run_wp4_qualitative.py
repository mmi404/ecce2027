"""
WP4: auto-selected qualitative failure figure. For every ctg_pooled_eval image with
>=2 headline-class GT boxes, computes per-image recall for E1 (mean of 3 Dhaka-trained
folds) and E2 (Chattogram-trained, pooled), ranks by E2-minus-E1 recall drop (largest
transfer failures), and renders the top 12 with GT boxes vs. e1_0's predicted boxes
overlaid -- selection is entirely by this script, not hand-picked.
"""
import sys, os, csv, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp4_lib as w

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

RESULTS_DIR = os.path.join(w.PROJECT_ROOT, 'results')
FIG_DIR = os.path.join(w.PROJECT_ROOT, 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

HEADLINE_IDS = sorted(cid for cid, name in w.CLASS_NAMES.items() if name in w.HEADLINE_CLASSES)
IMAGE_ROOTS = [
    os.path.join(w.PROJECT_ROOT, 'data', 'raw', 'zenodo', 'badodd', split, 'images')
    for split in ('train', 'val', 'test')
]
_image_path_cache = None


def find_image_path(img_id):
    global _image_path_cache
    if _image_path_cache is None:
        _image_path_cache = {}
        for root in IMAGE_ROOTS:
            if not os.path.isdir(root):
                continue
            for fn in os.listdir(root):
                if fn.endswith('.jpg') and not fn.startswith('.'):
                    _image_path_cache[fn[:-4]] = os.path.join(root, fn)
    return _image_path_cache.get(img_id)


def per_image_recall(gt_by_image, preds_by_image, img_id, conf_thresh):
    gts = gt_by_image.get(img_id, [])
    if not gts:
        return None, 0
    preds = [p for p in preds_by_image.get(img_id, []) if p[0] >= conf_thresh]
    preds.sort(key=lambda x: -x[0])
    matched = [False] * len(gts)
    for score, cls, pbox in preds:
        best_iou, best_j = 0.5, -1
        for j, (gt_cls, gbox) in enumerate(gts):
            if matched[j] or gt_cls != cls:
                continue
            iou = w._iou_xywh(pbox, gbox)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched[best_j] = True
    return sum(matched) / len(gts), len(gts)


def load_raw_gt(target):
    d = json.load(open(os.path.join(w.GT_DIR, f'{target}_coco_gt.json')))
    gt_by_image = defaultdict(list)
    for ann in d['annotations']:
        if ann['category_id'] in HEADLINE_IDS:
            gt_by_image[ann['image_id']].append((ann['category_id'], ann['bbox']))
    return gt_by_image


def load_raw_preds(run_id, target):
    d = json.load(open(os.path.join(w.PRED_DIR, f'{run_id}_{target}_preds_conf0001.json')))
    preds_by_image = defaultdict(list)
    for p in d:
        if p['category_id'] in HEADLINE_IDS:
            preds_by_image[p['image_id']].append((p['score'], p['category_id'], p['bbox']))
    return preds_by_image


def main():
    thresholds = {}
    with open(os.path.join(RESULTS_DIR, 'thresholds.csv')) as f:
        for row in csv.DictReader(f):
            thresholds[row['run_id']] = float(row['threshold'])
    e2_thresh = json.load(open(os.path.join(RESULTS_DIR, 'logs', 'e2_pooled_threshold.json')))['threshold']

    image_ids, _ = w.load_gt('ctg_pooled_eval')
    gt_by_image = load_raw_gt('ctg_pooled_eval')

    e1_preds = [load_raw_preds(f'e1_{k}', 'ctg_pooled_eval') for k in range(3)]
    e1_thresh = [thresholds[f'e1_{k}'] for k in range(3)]

    e2_preds_by_image = defaultdict(list)
    for k in range(3):
        for img_id, plist in load_raw_preds(f'e2_{k}', f'ctg_fold{k}_eval').items():
            e2_preds_by_image[img_id].extend(plist)

    candidates = []
    for img_id in image_ids:
        n_gt = len(gt_by_image.get(img_id, []))
        if n_gt < 2:
            continue
        e1_recalls = []
        for preds, t in zip(e1_preds, e1_thresh):
            r, _ = per_image_recall(gt_by_image, preds, img_id, t)
            if r is not None:
                e1_recalls.append(r)
        e1_mean = sum(e1_recalls) / len(e1_recalls) if e1_recalls else None
        e2_r, _ = per_image_recall(gt_by_image, e2_preds_by_image, img_id, e2_thresh)
        if e1_mean is None or e2_r is None:
            continue
        candidates.append((img_id, n_gt, e1_mean, e2_r, e2_r - e1_mean))

    candidates.sort(key=lambda x: -x[4])
    top = candidates[:12]

    with open(os.path.join(RESULTS_DIR, 'qualitative_failure_selection.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'n_gt_headline', 'e1_mean_recall', 'e2_recall', 'recall_drop'])
        for row in top:
            writer.writerow(row)
    print('Wrote results/qualitative_failure_selection.csv')
    for row in top:
        print(f'{row[0]:40s} n={row[1]:2d} e1_recall={row[2]:.2f} e2_recall={row[3]:.2f} drop={row[4]:.2f}')

    colors = {cid: c for cid, c in zip(HEADLINE_IDS, plt.cm.tab10.colors)}

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, (img_id, n_gt, e1_r, e2_r, drop) in zip(axes.flat, top):
        img_path = find_image_path(img_id)
        if img_path is None:
            ax.axis('off')
            continue
        img = Image.open(img_path)
        ax.imshow(img)
        for gt_cls, (x, y, bw, bh) in gt_by_image[img_id]:
            ax.add_patch(patches.Rectangle((x, y), bw, bh, linewidth=1.5,
                                            edgecolor=colors[gt_cls], facecolor='none'))
        # Overlay e1_0's detections (representative single model) above its own threshold.
        for score, cls, (x, y, bw, bh) in e1_preds[0].get(img_id, []):
            if score >= e1_thresh[0] and cls in HEADLINE_IDS:
                ax.add_patch(patches.Rectangle((x, y), bw, bh, linewidth=1.2,
                                                edgecolor=colors[cls], facecolor='none',
                                                linestyle='--'))
        ax.set_title(f'{img_id}\nrecall drop={drop:.2f} (E1={e1_r:.2f}, E2={e2_r:.2f})', fontsize=7)
        ax.axis('off')
    for ax in axes.flat[len(top):]:
        ax.axis('off')

    handles = [patches.Patch(edgecolor=colors[c], facecolor='none', label=w.CLASS_NAMES[c])
               for c in HEADLINE_IDS]
    handles += [patches.Patch(edgecolor='gray', facecolor='none', label='GT (solid)'),
                patches.Patch(edgecolor='gray', facecolor='none', linestyle='--', label='E1 pred (dashed)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=7)
    fig.suptitle('Auto-selected largest per-image recall drops (E2 recall $-$ E1 recall), '
                  'solid=ground truth, dashed=E1 (Dhaka-trained, e1_0) detections', fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(FIG_DIR, 'qualitative_failures.pdf')
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.replace('.pdf', '_preview.png'), dpi=120)
    plt.close(fig)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
