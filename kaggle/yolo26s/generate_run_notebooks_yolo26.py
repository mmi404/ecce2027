"""
Generates the 8 train+eval Kaggle notebook pairs for the YOLO26s
architecture-robustness follow-up: same 8-run matrix (E1 x3, E2 x3, 2
robustness folds), same manifests, same eval targets, same pinned
hyperparameters as the YOLOv8s study in generate_run_notebooks.py -- only
`model` differs (yolov8s.pt -> yolo26s.pt) -- so any AP50/recall gap between
the two architectures isolates the architecture effect, not a hyperparameter
or split difference.

Self-contained: hyperparameters are read once from configs/hyp_yolo26s.yaml
at *generation* time (locally) and baked into each notebook as a literal
dict, so the Kaggle-side `badodd-ecce2027-bundle` dataset does not need to be
touched or re-uploaded -- these notebooks attach to the exact same bundle the
YOLOv8s runs used.

Run once locally: python generate_run_notebooks_yolo26.py
Writes kaggle/yolo26s/train_run_<id>_y26.ipynb and kaggle/yolo26s/eval_run_<id>_y26.ipynb.
"""
import json
import os
import yaml

KAGGLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(KAGGLE_DIR))
HYP_PATH = os.path.join(REPO_ROOT, 'configs', 'hyp_yolo26s.yaml')

with open(HYP_PATH) as f:
    HYP = yaml.safe_load(f)
assert HYP['model'] == 'yolo26s.pt', f"Expected yolo26s.pt in {HYP_PATH}, got {HYP['model']}"

# ---------------------------------------------------------------------------
# Same run matrix as generate_run_notebooks.py (notes/splits.md section 5),
# only the output RUN_ID gets a _y26 suffix so weights/predictions/logs land
# under distinct filenames and never collide with the YOLOv8s run outputs.
# ---------------------------------------------------------------------------
RUNS = {
    'e1_0': dict(
        label='E1.0/YOLO26s -- Dhaka Fold 0 (train) -> transfer + in-domain + robustness eval',
        train_manifest='dhaka_fold0_train_matched',
        monitor_manifest='dhaka_fold0_train_matched_monitor',
        eval_targets=['ctg_pooled_eval', 'dhaka_fold0_eval',
                      'ctg_robustness_holdout_bohoddarhat1_eval',
                      'ctg_robustness_holdout_bohoddarhat2_eval'],
    ),
    'e1_1': dict(
        label='E1.1/YOLO26s -- Dhaka Fold 1 (train) -> transfer + in-domain + robustness eval',
        train_manifest='dhaka_fold1_train_matched',
        monitor_manifest='dhaka_fold1_train_matched_monitor',
        eval_targets=['ctg_pooled_eval', 'dhaka_fold1_eval',
                      'ctg_robustness_holdout_bohoddarhat1_eval',
                      'ctg_robustness_holdout_bohoddarhat2_eval'],
    ),
    'e1_2': dict(
        label='E1.2/YOLO26s -- Dhaka Fold 2 (train) -> transfer + in-domain + robustness eval',
        train_manifest='dhaka_fold2_train_matched',
        monitor_manifest='dhaka_fold2_train_matched_monitor',
        eval_targets=['ctg_pooled_eval', 'dhaka_fold2_eval',
                      'ctg_robustness_holdout_bohoddarhat1_eval',
                      'ctg_robustness_holdout_bohoddarhat2_eval'],
    ),
    'e2_0': dict(
        label='E2.0/YOLO26s -- Chattogram Fold 0 (train) -> out-of-fold + reverse-transfer eval',
        train_manifest='ctg_fold0_train',
        monitor_manifest='ctg_fold0_train_monitor',
        eval_targets=['ctg_fold0_eval', 'dhaka_fold0_eval'],
    ),
    'e2_1': dict(
        label='E2.1/YOLO26s -- Chattogram Fold 1 (train) -> out-of-fold + reverse-transfer eval',
        train_manifest='ctg_fold1_train',
        monitor_manifest='ctg_fold1_train_monitor',
        eval_targets=['ctg_fold1_eval', 'dhaka_fold1_eval'],
    ),
    'e2_2': dict(
        label='E2.2/YOLO26s -- Chattogram Fold 2 (train) -> out-of-fold + reverse-transfer eval',
        train_manifest='ctg_fold2_train',
        monitor_manifest='ctg_fold2_train_monitor',
        eval_targets=['ctg_fold2_eval', 'dhaka_fold2_eval'],
    ),
    'rob_a': dict(
        label='Robustness A/YOLO26s -- leave-bohoddarhat1-out (train) -> held-out corridor eval',
        train_manifest='ctg_robustness_holdout_bohoddarhat1_train_matched',
        monitor_manifest='ctg_robustness_holdout_bohoddarhat1_train_matched_monitor',
        eval_targets=['ctg_robustness_holdout_bohoddarhat1_eval'],
    ),
    'rob_b': dict(
        label='Robustness B/YOLO26s -- leave-bohoddarhat2-out (train) -> held-out corridor eval',
        train_manifest='ctg_robustness_holdout_bohoddarhat2_train_matched',
        monitor_manifest='ctg_robustness_holdout_bohoddarhat2_train_matched_monitor',
        eval_targets=['ctg_robustness_holdout_bohoddarhat2_eval'],
    ),
}

# ---------------------------------------------------------------------------
# TRAIN notebook cell templates
# ---------------------------------------------------------------------------

TRAIN_MD = """# ECCE 2027 Architecture Follow-up -- TRAIN ONLY (Kaggle): __LABEL__

YOLO26s replicate of the base run `__BASE_RUN_ID__`, identical manifests and
hyperparameters, only the model changes (yolov8s.pt -> yolo26s.pt), so the
AP50/recall gap between architectures isolates the architecture effect.

Split into train-only / eval-only notebooks for the same reason as the
YOLOv8s notebooks: inference on this Kaggle T4 x2 tier leaks GPU memory when
run in the same process as training.

**Attach one dataset before running** (Add Input, right sidebar) -- the exact
same bundle the YOLOv8s runs used, no changes needed:
- `badodd-ecce2027-bundle` (contains `badodd.zip` images and
  `badodd_ecce2027_overlay.zip` labels/splits/configs).

Settings: Accelerator = GPU T4 x2, Internet = ON (needed once, to download
the yolo26s.pt COCO-pretrained weights from Ultralytics on first run).

Train manifest: `splits/__TRAIN_MANIFEST__.txt`
Monitor manifest: `splits/__MONITOR_MANIFEST__.txt` (10% slice of the training set, val-only, never a real test set)

**When this finishes:** click **Save Version -> Save & Run All (Commit)**, then
attach this notebook as an input to `eval_run___RUN_ID__.ipynb`.
"""

TRAIN_CELL1 = """# ==============================================================================
# 1. HARDWARE & ENVIRONMENT VERIFICATION
# ==============================================================================
import os, sys, time, glob, json, shutil, subprocess

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from pathlib import Path
import torch

PINNED_ULTRALYTICS = "8.4.155"
RUN_ID = '__RUN_ID__'

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('CUDA device count (should be 1 after masking):', torch.cuda.device_count())

subprocess.run("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free "
               "--format=csv", shell=True)

assert torch.cuda.is_available(), (
    'FAIL: no GPU detected -- Accelerator is set to "None" in this notebook\\'s Settings tab. '
    'Cancel this run, set Accelerator = GPU T4 x2 in Settings, and re-run Save & Run All.'
)
device_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'GPU Device: {device_name} ({total_vram:.2f} GB VRAM)')
torch.cuda.reset_peak_memory_stats(0)

subprocess.run(f"pip install -q -U ultralytics=={PINNED_ULTRALYTICS} pycocotools", shell=True, check=True)
import ultralytics
assert ultralytics.__version__ == PINNED_ULTRALYTICS, (
    f"Ultralytics version drift: installed {ultralytics.__version__}, expected {PINNED_ULTRALYTICS}"
)
print('Ultralytics version:', ultralytics.__version__)
print('Run ID:', RUN_ID)

from ultralytics import YOLO

# Sanity check this ultralytics build actually ships YOLO26 before spending
# any GPU time -- fail loudly and immediately rather than partway into training.
try:
    _probe = YOLO('yolo26s.pt')
    print('YOLO26s weights loaded OK -- architecture is supported on this runtime.')
    del _probe
except Exception as e:
    raise RuntimeError(
        f"Could not load yolo26s.pt on ultralytics=={PINNED_ULTRALYTICS}: {e}. "
        f"If Ultralytics has not yet shipped YOLO26 in this pinned version, either bump "
        f"PINNED_ULTRALYTICS here (and re-verify hyp compatibility) or fall back to yolo11s.pt."
    ) from e"""

TRAIN_CELL2 = """# ==============================================================================
# 2. DATASET & OVERLAY DISCOVERY -- FAIL HARD on any missing image
# ==============================================================================
print('=== Scanning /kaggle/input for Dataset & Overlay ===')

badodd_zips = glob.glob('/kaggle/input/**/badodd.zip', recursive=True)
if badodd_zips and not glob.glob('/kaggle/input/**/*.jpg', recursive=True):
    print(f'Found badodd.zip at {badodd_zips[0]}. Extracting to /kaggle/working/badodd_images...')
    os.makedirs('/kaggle/working/badodd_images', exist_ok=True)
    subprocess.run(f"unzip -q {badodd_zips[0]} -d /kaggle/working/badodd_images", shell=True, check=True)
    IMAGE_SEARCH_ROOT = '/kaggle/working/badodd_images'
else:
    IMAGE_SEARCH_ROOT = '/kaggle/input'

all_input_imgs = [p for p in glob.glob(f'{IMAGE_SEARCH_ROOT}/**/*.jpg', recursive=True) +
                        glob.glob(f'{IMAGE_SEARCH_ROOT}/**/*.png', recursive=True)
                   if not os.path.basename(p).startswith('._')]
img_lookup = {os.path.basename(p): p for p in all_input_imgs}
print(f'Indexed {len(img_lookup)} images from {IMAGE_SEARCH_ROOT}.')
assert len(img_lookup) >= 10000, (
    f"Expected ~10,032 BadODD images, found only {len(img_lookup)} -- "
    f"is the bundle dataset attached?"
)

overlay_roots = []
for root, dirs, files in os.walk('/kaggle/input'):
    if 'splits' in dirs and 'labels' in dirs:
        overlay_roots.append(root)

overlay_zips = glob.glob('/kaggle/input/**/badodd_ecce2027_overlay*.zip', recursive=True)
overlay_src = None
if overlay_roots:
    overlay_src = overlay_roots[0]
    print(f'Found overlay directory at: {overlay_src}')
elif overlay_zips:
    print(f'Found overlay zip at: {overlay_zips[0]}. Unzipping to /kaggle/working/overlay...')
    subprocess.run(f"unzip -q {overlay_zips[0]} -d /kaggle/working/overlay", shell=True, check=True)
    overlay_src = '/kaggle/working/overlay'
else:
    raise FileNotFoundError(
        'Could not locate the ECCE 2027 overlay package in /kaggle/input! '
        'Attach the bundle dataset.'
    )

assert os.path.isdir(os.path.join(overlay_src, 'labels'))
assert os.path.isdir(os.path.join(overlay_src, 'splits'))

WORK_DATA = '/kaggle/working/data'
WORK_SPLITS = os.path.join(WORK_DATA, 'splits')
WORK_LABELS = os.path.join(WORK_DATA, 'labels')
WORK_IMAGES = os.path.join(WORK_DATA, 'images')
os.makedirs(WORK_SPLITS, exist_ok=True)
os.makedirs(WORK_LABELS, exist_ok=True)
os.makedirs(WORK_IMAGES, exist_ok=True)

label_files = glob.glob(os.path.join(overlay_src, 'labels', '*.txt'))
for lf in label_files:
    dest = os.path.join(WORK_LABELS, os.path.basename(lf))
    if not os.path.exists(dest):
        try:
            os.symlink(lf, dest)
        except OSError:
            shutil.copy(lf, dest)
print(f'Linked {len(label_files)} label files.')

for bname, src_p in img_lookup.items():
    dest = os.path.join(WORK_IMAGES, bname)
    if not os.path.exists(dest):
        try:
            os.symlink(src_p, dest)
        except OSError:
            shutil.copy(src_p, dest)
print(f'Linked {len(img_lookup)} image files.')

manifest_files = glob.glob(os.path.join(overlay_src, 'splits', '*.txt'))
resolved_counts = {}
for mf in manifest_files:
    mname = os.path.basename(mf)
    with open(mf, 'r') as f:
        bases = [os.path.basename(l.strip()) for l in f if l.strip()]
    resolved = []
    missing_here = []
    for b in bases:
        p = os.path.join(WORK_IMAGES, b)
        if os.path.exists(p):
            resolved.append(p)
        else:
            missing_here.append(b)
    if missing_here:
        raise FileNotFoundError(
            f"{mname}: {len(missing_here)} images referenced in the manifest are missing, "
            f"e.g. {missing_here[:5]}. Do not proceed -- check the attached dataset."
        )
    with open(os.path.join(WORK_SPLITS, mname), 'w') as f:
        f.writelines(p + '\\n' for p in resolved)
    resolved_counts[mname] = len(resolved)

print(f'Resolved {len(manifest_files)} manifests, ALL images present. This run uses:')
for k in ('__TRAIN_MANIFEST__.txt', '__MONITOR_MANIFEST__.txt'):
    print(f'  {k}: {resolved_counts.get(k)}')

print('Dataset & Overlay setup complete!')"""

TRAIN_CELL3 = """# ==============================================================================
# 3. YOLO26s HYPERPARAMETERS (baked in at generation time from
#    configs/hyp_yolo26s.yaml -- identical to the YOLOv8s hyp.yaml except
#    `model`, so this run differs from the base study by architecture only)
# ==============================================================================
hyp = __HYP_DICT_REPR__
print('Hyperparameters (baked in from configs/hyp_yolo26s.yaml at notebook-generation time):')
print(hyp)

CLASS_NAMES = {
    0: 'auto_rickshaw', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'cart_vehicle',
    5: 'construction_vehicle', 6: 'motorbike', 7: 'person', 8: 'priority_vehicle',
    9: 'three_wheeler', 10: 'truck',
}

run_yaml = {
    'path': WORK_DATA,
    'train': os.path.join(WORK_SPLITS, '__TRAIN_MANIFEST__.txt'),
    'val': os.path.join(WORK_SPLITS, '__MONITOR_MANIFEST__.txt'),
    'names': CLASS_NAMES,
}

import yaml
RUN_YAML_PATH = f'/kaggle/working/data_{RUN_ID}.yaml'
with open(RUN_YAML_PATH, 'w') as f:
    yaml.dump(run_yaml, f, sort_keys=False)

print(f'\\nWrote dataset configuration to: {RUN_YAML_PATH}')
print('Train manifest:', run_yaml['train'])
print('Val monitoring manifest:', run_yaml['val'])"""

TRAIN_CELL4 = """# ==============================================================================
# 3b. LABEL INTEGRITY AUDIT -- run BEFORE training. Catches silent background-only runs.
# ==============================================================================
train_manifest = run_yaml['train']
with open(train_manifest) as f:
    train_imgs = [l.strip() for l in f if l.strip()]

total_boxes_gt = 0
n_background = 0
for img_p in train_imgs:
    stem = os.path.splitext(os.path.basename(img_p))[0]
    lbl_p = os.path.join(WORK_LABELS, stem + '.txt')
    if not os.path.exists(lbl_p) or os.path.getsize(lbl_p) == 0:
        n_background += 1
        continue
    with open(lbl_p) as f:
        n = sum(1 for line in f if line.strip())
    total_boxes_gt += n
    if n == 0:
        n_background += 1

bg_rate = n_background / len(train_imgs)
print(f'Train Images in Manifest: {len(train_imgs)} | Total Ground Truth Boxes: {total_boxes_gt} | '
      f'Background Images: {n_background} ({bg_rate*100:.2f}%)')
assert total_boxes_gt > 0, 'FAIL: zero ground-truth boxes found -- labels did not link correctly.'
assert bg_rate < 0.05, f'FAIL: background image rate {bg_rate*100:.2f}% >= 5% -- label linkage is broken.'
print('Label integrity check PASSED.')"""

TRAIN_CELL5 = """# ==============================================================================
# 4. RUN FULL TRAINING (__LABEL__ -- 100 EPOCHS) & SAVE EVERYTHING TO OUTPUT
# ==============================================================================
PROJECT_DIR = '/kaggle/working/runs/train'
EXP_NAME = RUN_ID
EXP_DIR = os.path.join(PROJECT_DIR, EXP_NAME)
LAST_CKPT = os.path.join(EXP_DIR, 'weights', 'last.pt')

can_resume = os.path.exists(LAST_CKPT)
model = YOLO(LAST_CKPT if can_resume else hyp['model'])
print('Resuming from checkpoint.' if can_resume else f"Starting fresh from {hyp['model']}.")
if can_resume:
    print('NOTE: resumed from a prior partial run -- total_train_time_s below only covers this session.')

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(0)

EPOCHS = 100
t0_train = time.time()
train_results = model.train(
    data=RUN_YAML_PATH,
    epochs=EPOCHS,
    patience=hyp['patience'],
    batch=hyp['batch'],
    imgsz=hyp['imgsz'],
    device=0 if torch.cuda.is_available() else 'cpu',
    workers=4,
    optimizer=hyp['optimizer'],
    lr0=hyp['lr0'], lrf=hyp['lrf'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'],
    warmup_epochs=hyp['warmup_epochs'], warmup_momentum=hyp['warmup_momentum'], warmup_bias_lr=hyp['warmup_bias_lr'],
    box=hyp['box'], cls=hyp['cls'], dfl=hyp['dfl'],
    hsv_h=hyp['hsv_h'], hsv_s=hyp['hsv_s'], hsv_v=hyp['hsv_v'],
    degrees=hyp['degrees'], translate=hyp['translate'], scale=hyp['scale'], shear=hyp['shear'],
    perspective=hyp['perspective'], flipud=hyp['flipud'], fliplr=hyp['fliplr'],
    mosaic=hyp['mosaic'], mixup=hyp['mixup'], copy_paste=hyp['copy_paste'],
    seed=hyp['seed'], deterministic=hyp['deterministic'],
    project=PROJECT_DIR, name=EXP_NAME, exist_ok=True, save=True, resume=can_resume, verbose=True,
)

total_train_time = time.time() - t0_train
time_per_epoch = total_train_time / EPOCHS

peak_vram_gb = 0.0
if torch.cuda.is_available():
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

assert os.path.exists(LAST_CKPT), f'ERROR: {LAST_CKPT} was not created!'
print('\\n' + '='*60)
print(f'           {RUN_ID} TRAINING RESULTS ({EPOCHS} epochs)           ')
print('='*60)
print(f'Total Training Time: {total_train_time:.2f} s ({total_train_time/60:.1f} min)')
print(f'Approx Duration Per Epoch (this session): {time_per_epoch:.2f} s')
print(f'Resumed from a prior partial run: {can_resume}')
print(f'Peak GPU VRAM Allocated: {peak_vram_gb:.2f} GB')
print(f'Checkpoint Generated: {LAST_CKPT}')
print('='*60)

run_info = {
    'run_id': RUN_ID,
    'model': hyp['model'],
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'total_vram_gb': torch.cuda.get_device_properties(0).total_memory/(1024**3) if torch.cuda.is_available() else None,
    'torch_version': torch.__version__,
    'ultralytics_version': PINNED_ULTRALYTICS,
    'epochs': EPOCHS,
    'resumed': can_resume,
    'total_train_time_s': total_train_time,
    'seconds_per_epoch_this_session': time_per_epoch,
    'peak_vram_gb': peak_vram_gb,
}

# Save everything the matching eval notebook will need, under stable,
# unambiguous filenames so it can glob for them among whatever else is attached.
RESULTS_DIR = '/kaggle/working/results'
os.makedirs(f'{RESULTS_DIR}/logs', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/weights', exist_ok=True)
with open(f'{RESULTS_DIR}/logs/{RUN_ID}_run_info.json', 'w') as f:
    json.dump(run_info, f, indent=2)
shutil.copy(os.path.join(EXP_DIR, 'results.csv'), f'{RESULTS_DIR}/logs/{RUN_ID}_results.csv')
shutil.copy(os.path.join(EXP_DIR, 'args.yaml'), f'{RESULTS_DIR}/logs/{RUN_ID}_args.yaml')
shutil.copy(LAST_CKPT, f'{RESULTS_DIR}/weights/{RUN_ID}_last.pt')

print(json.dumps(run_info, indent=2))
print(f"\\nSaved checkpoint to {RESULTS_DIR}/weights/{RUN_ID}_last.pt -- this is what")
print(f"eval_run_{RUN_ID}.ipynb will load. Click Save Version -> Save & Run All (Commit) now.")"""

# ---------------------------------------------------------------------------
# EVAL notebook cell templates
# ---------------------------------------------------------------------------

EVAL_MD = """# ECCE 2027 Architecture Follow-up -- EVAL ONLY (Kaggle): __LABEL__

Runs in a brand-new container, separate from `train_run___RUN_ID__.ipynb`, so the
GPU is guaranteed clean.

**Attach two inputs before running** (Add Input, right sidebar):
1. `badodd-ecce2027-bundle` (images + overlay + coco_gt) -- same bundle as the YOLOv8s runs.
2. `train_run___RUN_ID__` -- your own training notebook, added as an input. It
   must have a completed, saved version first.

Evaluation targets for this run: __EVAL_TARGETS_STR__

Settings: Accelerator = GPU T4 x2, Internet = ON.
"""

EVAL_CELL1 = """# ==============================================================================
# 1. HARDWARE & ENVIRONMENT VERIFICATION (fresh container -- GPU should start clean)
# ==============================================================================
import os, sys, time, glob, json, shutil, subprocess

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from pathlib import Path
import torch

PINNED_ULTRALYTICS = "8.4.155"
RUN_ID = '__RUN_ID__'

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('CUDA device count (should be 1 after masking):', torch.cuda.device_count())

subprocess.run("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free "
               "--format=csv", shell=True)

assert torch.cuda.is_available(), (
    'FAIL: no GPU detected -- Accelerator is set to "None" in this notebook\\'s Settings tab. '
    'Cancel this run, set Accelerator = GPU T4 x2 in Settings, and re-run Save & Run All.'
)
device_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'GPU Device: {device_name} ({total_vram:.2f} GB VRAM)')
allocated_at_start = torch.cuda.memory_allocated(0) / 1e9
print(f'GPU memory allocated at notebook start: {allocated_at_start:.3f} GB (should be ~0)')

subprocess.run(f"pip install -q -U ultralytics=={PINNED_ULTRALYTICS} pycocotools", shell=True, check=True)
import ultralytics
assert ultralytics.__version__ == PINNED_ULTRALYTICS, (
    f"Ultralytics version drift: installed {ultralytics.__version__}, expected {PINNED_ULTRALYTICS}"
)
print('Ultralytics version:', ultralytics.__version__)
print('Run ID:', RUN_ID)

from ultralytics import YOLO"""

EVAL_CELL2 = """# ==============================================================================
# 2. DATASET & OVERLAY DISCOVERY -- FAIL HARD on any missing image
# ==============================================================================
print('=== Scanning /kaggle/input for Dataset & Overlay ===')

badodd_zips = glob.glob('/kaggle/input/**/badodd.zip', recursive=True)
if badodd_zips and not glob.glob('/kaggle/input/**/*.jpg', recursive=True):
    print(f'Found badodd.zip at {badodd_zips[0]}. Extracting to /kaggle/working/badodd_images...')
    os.makedirs('/kaggle/working/badodd_images', exist_ok=True)
    subprocess.run(f"unzip -q {badodd_zips[0]} -d /kaggle/working/badodd_images", shell=True, check=True)
    IMAGE_SEARCH_ROOT = '/kaggle/working/badodd_images'
else:
    IMAGE_SEARCH_ROOT = '/kaggle/input'

all_input_imgs = [p for p in glob.glob(f'{IMAGE_SEARCH_ROOT}/**/*.jpg', recursive=True) +
                        glob.glob(f'{IMAGE_SEARCH_ROOT}/**/*.png', recursive=True)
                   if not os.path.basename(p).startswith('._')]
img_lookup = {os.path.basename(p): p for p in all_input_imgs}
print(f'Indexed {len(img_lookup)} images from {IMAGE_SEARCH_ROOT}.')
assert len(img_lookup) >= 10000, (
    f"Expected ~10,032 BadODD images, found only {len(img_lookup)} -- "
    f"is the bundle dataset attached?"
)

overlay_roots = []
for root, dirs, files in os.walk('/kaggle/input'):
    if 'splits' in dirs and 'coco_gt' in dirs:
        overlay_roots.append(root)

overlay_zips = glob.glob('/kaggle/input/**/badodd_ecce2027_overlay*.zip', recursive=True)
overlay_src = None
if overlay_roots:
    overlay_src = overlay_roots[0]
    print(f'Found overlay directory at: {overlay_src}')
elif overlay_zips:
    print(f'Found overlay zip at: {overlay_zips[0]}. Unzipping to /kaggle/working/overlay...')
    subprocess.run(f"unzip -q {overlay_zips[0]} -d /kaggle/working/overlay", shell=True, check=True)
    overlay_src = '/kaggle/working/overlay'
else:
    raise FileNotFoundError(
        'Could not locate the ECCE 2027 overlay package in /kaggle/input! '
        'Attach the bundle dataset.'
    )

assert os.path.isdir(os.path.join(overlay_src, 'coco_gt'))
assert os.path.isdir(os.path.join(overlay_src, 'splits'))

WORK_DATA = '/kaggle/working/data'
WORK_IMAGES = os.path.join(WORK_DATA, 'images')
WORK_SPLITS = os.path.join(WORK_DATA, 'splits')
os.makedirs(WORK_IMAGES, exist_ok=True)
os.makedirs(WORK_SPLITS, exist_ok=True)

for bname, src_p in img_lookup.items():
    dest = os.path.join(WORK_IMAGES, bname)
    if not os.path.exists(dest):
        try:
            os.symlink(src_p, dest)
        except OSError:
            shutil.copy(src_p, dest)
print(f'Linked {len(img_lookup)} image files.')

EVAL_TARGETS = __EVAL_TARGETS_LIST__

resolved_counts = {}
for t in EVAL_TARGETS:
    manifest_src = os.path.join(overlay_src, 'splits', f'{t}.txt')
    with open(manifest_src) as f:
        bases = [os.path.basename(l.strip()) for l in f if l.strip()]
    resolved = []
    missing_here = []
    for b in bases:
        p = os.path.join(WORK_IMAGES, b)
        if os.path.exists(p):
            resolved.append(p)
        else:
            missing_here.append(b)
    if missing_here:
        raise FileNotFoundError(
            f"{t}.txt: {len(missing_here)} images missing, e.g. {missing_here[:5]}. "
            f"Do not proceed -- check the attached dataset."
        )
    gt_path = os.path.join(overlay_src, 'coco_gt', f'{t}_coco_gt.json')
    assert os.path.isfile(gt_path), f'Missing COCO ground truth for {t}: {gt_path}'
    with open(os.path.join(WORK_SPLITS, f'{t}.txt'), 'w') as f:
        f.writelines(p + '\\n' for p in resolved)
    resolved_counts[t] = len(resolved)

print('Resolved eval manifests (all images present):')
for t, n in resolved_counts.items():
    print(f'  {t}: {n} images')

CLASS_NAMES = {
    0: 'auto_rickshaw', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'cart_vehicle',
    5: 'construction_vehicle', 6: 'motorbike', 7: 'person', 8: 'priority_vehicle',
    9: 'three_wheeler', 10: 'truck',
}"""

EVAL_CELL3 = """# ==============================================================================
# 3. LOCATE THE TRAINED CHECKPOINT FROM the matching train notebook's OUTPUT
# ==============================================================================
ckpt_candidates = glob.glob(f'/kaggle/input/**/{RUN_ID}_last.pt', recursive=True)
if not ckpt_candidates:
    raise FileNotFoundError(
        f"Could not find {RUN_ID}_last.pt under /kaggle/input. Attach train_run_{RUN_ID}.ipynb "
        f"as an input (Add Input -> your notebooks), and make sure it has a completed, "
        f"saved version."
    )
print(f'Checkpoint candidates found ({len(ckpt_candidates)}):')
for c in ckpt_candidates:
    print(f'  {c}  ({os.path.getsize(c)/1e6:.2f} MB)')
if len(ckpt_candidates) > 1:
    print('WARNING: multiple matches -- using the first one. If the train notebook was '
          're-run multiple times, stale copies from earlier versions may be attached too.')
LAST_CKPT = ckpt_candidates[0]
print(f'\\nUsing checkpoint: {LAST_CKPT} ({os.path.getsize(LAST_CKPT)/1e6:.2f} MB)')
assert os.path.getsize(LAST_CKPT) < 300e6, (
    f'Checkpoint is {os.path.getsize(LAST_CKPT)/1e6:.1f} MB -- unexpectedly large for a '
    f'YOLO26s last.pt. Something may be wrong with this file -- do not proceed without checking.'
)

run_info_candidates = glob.glob(f'/kaggle/input/**/{RUN_ID}_run_info.json', recursive=True)
if run_info_candidates:
    with open(run_info_candidates[0]) as f:
        train_run_info = json.load(f)
    print('\\nTraining run info:')
    print(json.dumps(train_run_info, indent=2))
else:
    train_run_info = None
    print('WARNING: run_info.json not found alongside the checkpoint -- proceeding without it.')"""

EVAL_CELL4 = """# ==============================================================================
# 4. EXPORT PREDICTIONS + PYCOCOTOOLS CHECK FOR EVERY EVAL TARGET
# ==============================================================================
# Same chunked-reload pattern as the YOLOv8s eval notebooks: the streaming
# predictor leaks GPU memory per batch and never releases it mid-run, so each
# eval target is processed in small chunks, destroying and reloading the
# model between chunks so nothing accumulates past a safe ceiling.
import gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CHUNK_SIZE = 128

if torch.cuda.is_available():
    print(f'GPU memory at start of export: allocated={torch.cuda.memory_allocated(0)/1e9:.3f} GB '
          f'(should be ~0 in this fresh container)')

RESULTS_DIR = '/kaggle/working/results/predictions'
LOGS_DIR = '/kaggle/working/results/logs'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def export_predictions(images, ckpt_path):
    coco_predictions = []
    class_counts = {cid: 0 for cid in CLASS_NAMES}
    chunks = [images[i:i + CHUNK_SIZE] for i in range(0, len(images), CHUNK_SIZE)]
    for chunk_idx, chunk in enumerate(chunks):
        try:
            chunk_model = YOLO(ckpt_path)
            results_gen = chunk_model.predict(
                source=chunk, conf=0.001, iou=0.7, imgsz=640,
                device=0 if torch.cuda.is_available() else 'cpu',
                batch=16, stream=True, verbose=False, workers=0,
            )
            for r in results_gen:
                img_stem = Path(r.path).stem
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = xyxy[i]
                        cid = int(clss[i])
                        coco_predictions.append({
                            'image_id': img_stem, 'category_id': cid,
                            'bbox': [round(float(x1), 2), round(float(y1), 2),
                                     round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            'score': round(float(confs[i]), 5),
                        })
                        if cid in class_counts:
                            class_counts[cid] += 1
                del r
            del chunk_model, results_gen
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f'\\n*** OOM DURING CHUNK {chunk_idx + 1}/{len(chunks)} ***')
            print(torch.cuda.memory_summary(device=0, abbreviated=True))
            subprocess.run("nvidia-smi", shell=True)
            raise
    return coco_predictions, class_counts


def pycocotools_check(pred_json_path, gt_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    mAP50_95 = ev.stats[0]
    mAP50 = ev.stats[1]
    cat_ids = ev.params.catIds
    per_class_ap50 = {}
    for k, cid in enumerate(cat_ids):
        ap50_arr = ev.eval['precision'][0, :, k, 0, -1]
        ap50 = ap50_arr[ap50_arr > -1].mean() if (ap50_arr > -1).any() else float('nan')
        per_class_ap50[CLASS_NAMES[cid]] = ap50
    return mAP50, mAP50_95, per_class_ap50


results_summary = {}
for t in EVAL_TARGETS:
    print(f'\\n{"="*70}\\nEVAL TARGET: {t}\\n{"="*70}')
    with open(os.path.join(WORK_SPLITS, f'{t}.txt')) as f:
        images = [l.strip() for l in f if l.strip()]
    print(f'{len(images)} images.')

    t0 = time.time()
    coco_predictions, class_counts = export_predictions(images, LAST_CKPT)
    elapsed = time.time() - t0
    total_boxes = len(coco_predictions)
    print(f'Exported {total_boxes} detections in {elapsed:.1f}s ({len(images)/elapsed:.1f} FPS).')
    assert total_boxes > 0, f'FAIL: zero predictions for {t}.'

    pred_json_path = os.path.join(RESULTS_DIR, f'{RUN_ID}_{t}_preds_conf0001.json')
    with open(pred_json_path, 'w') as f:
        json.dump(coco_predictions, f)

    gt_json_path = os.path.join(overlay_src, 'coco_gt', f'{t}_coco_gt.json')
    mAP50, mAP50_95, per_class_ap50 = pycocotools_check(pred_json_path, gt_json_path)

    assert per_class_ap50.get('person', 0) > 0 or per_class_ap50.get('car', 0) > 0, (
        f'FAIL: AP50 is 0 for both person and car on {t} -- category ids or bbox coords '
        f'are almost certainly misaligned.'
    )
    print(f"mAP50={mAP50:.4f} mAP50-95={mAP50_95:.4f} "
          f"person={per_class_ap50.get('person', float('nan')):.4f} "
          f"car={per_class_ap50.get('car', float('nan')):.4f}")

    results_summary[t] = {
        'n_images': len(images), 'total_boxes': total_boxes,
        'mAP50': mAP50, 'mAP50_95': mAP50_95, 'per_class_ap50': per_class_ap50,
        'pred_json': pred_json_path,
    }

with open(os.path.join(LOGS_DIR, f'{RUN_ID}_eval_summary.json'), 'w') as f:
    json.dump({t: {k: v for k, v in d.items() if k != 'pred_json'} for t, d in results_summary.items()},
               f, indent=2)
print(f'\\nWrote eval summary to {LOGS_DIR}/{RUN_ID}_eval_summary.json')"""

EVAL_CELL5 = """# ==============================================================================
# 5. FINAL SUMMARY
# ==============================================================================
print('='*70)
print(f'                {RUN_ID.upper()} EVAL SUMMARY')
print('='*70)
if train_run_info:
    print(f"Training: {train_run_info.get('epochs', '?')} epochs, "
          f"{train_run_info.get('total_train_time_s', float('nan')):.1f}s total, "
          f"peak VRAM {train_run_info.get('peak_vram_gb', float('nan')):.2f} GB, "
          f"resumed={train_run_info.get('resumed')}")
for t, d in results_summary.items():
    print(f"\\n[{t}] n={d['n_images']} boxes={d['total_boxes']}")
    print(f"  mAP50={d['mAP50']:.4f} mAP50-95={d['mAP50_95']:.4f}")
    for cname, ap in d['per_class_ap50'].items():
        print(f"    {cname:22s}: AP50={ap:.4f}")
print('\\n' + '='*70)
print(f'>>> {RUN_ID} EVAL CHECKS PASSED SUCCESSFULLY <<<')
print('='*70)"""


def make_code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': source.splitlines(keepends=True)}


def make_md_cell(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


NB_METADATA = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
}


def write_notebook(path, cells):
    nb = {'cells': cells, 'metadata': NB_METADATA, 'nbformat': 4, 'nbformat_minor': 5}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'Wrote {path} ({len(cells)} cells)')


def hyp_dict_repr():
    # Deterministic key order matching configs/hyp_yolo26s.yaml, rendered as a
    # literal Python dict (NOT json.dumps -- that emits true/false/null, which
    # are not valid Python identifiers) so the notebook needs no external file
    # at runtime.
    import pprint
    return pprint.pformat(HYP, indent=4, sort_dicts=False, width=100)


def sub(template, base_run_id, run_id, cfg):
    eval_targets_repr = json.dumps(cfg.get('eval_targets', []))
    s = template
    s = s.replace('__RUN_ID__', run_id)
    s = s.replace('__BASE_RUN_ID__', base_run_id)
    s = s.replace('__LABEL__', cfg['label'])
    s = s.replace('__TRAIN_MANIFEST__', cfg.get('train_manifest', ''))
    s = s.replace('__MONITOR_MANIFEST__', cfg.get('monitor_manifest', ''))
    s = s.replace('__EVAL_TARGETS_LIST__', eval_targets_repr)
    s = s.replace('__EVAL_TARGETS_STR__', ', '.join(cfg.get('eval_targets', [])))
    s = s.replace('__HYP_DICT_REPR__', hyp_dict_repr())
    return s


for base_run_id, cfg in RUNS.items():
    run_id = f'{base_run_id}_y26'
    train_cells = [
        make_md_cell(sub(TRAIN_MD, base_run_id, run_id, cfg)),
        make_code_cell(sub(TRAIN_CELL1, base_run_id, run_id, cfg)),
        make_code_cell(sub(TRAIN_CELL2, base_run_id, run_id, cfg)),
        make_code_cell(sub(TRAIN_CELL3, base_run_id, run_id, cfg)),
        make_code_cell(sub(TRAIN_CELL4, base_run_id, run_id, cfg)),
        make_code_cell(sub(TRAIN_CELL5, base_run_id, run_id, cfg)),
    ]
    write_notebook(os.path.join(KAGGLE_DIR, f'train_run_{run_id}.ipynb'), train_cells)

    eval_cells = [
        make_md_cell(sub(EVAL_MD, base_run_id, run_id, cfg)),
        make_code_cell(sub(EVAL_CELL1, base_run_id, run_id, cfg)),
        make_code_cell(sub(EVAL_CELL2, base_run_id, run_id, cfg)),
        make_code_cell(sub(EVAL_CELL3, base_run_id, run_id, cfg)),
        make_code_cell(sub(EVAL_CELL4, base_run_id, run_id, cfg)),
        make_code_cell(sub(EVAL_CELL5, base_run_id, run_id, cfg)),
    ]
    write_notebook(os.path.join(KAGGLE_DIR, f'eval_run_{run_id}.ipynb'), eval_cells)

print('\\nDone: 8 train + 8 eval YOLO26s notebooks generated.')
