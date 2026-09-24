# Class-Resolved Cross-City Transfer of YOLO Vehicle Detectors

Code and results for a conference paper measuring how object detectors trained on
Dhaka dash-camera imagery (BadODD) transfer to Chattogram, class by class, and
whether the transfer gap is uniform across vehicle types or concentrated in specific
ones (informal and non-motorised modes). The result is checked against two detector
architectures, YOLOv8s and YOLO26s, under identical splits, seed and hyperparameters.

The paper is in [`paper/main.tex`](paper/main.tex) (compiled PDF: `paper/main.pdf`).

## Pipeline

1. **Data.** [BadODD](https://doi.org/10.5281/zenodo.13823687) (Zenodo, CC BY 4.0),
   10,032 labelled dash-camera images across 9 Bangladeshi districts.
   `src/download_badodd*.py` fetch it; `src/convert_and_split.py` builds the
   temporal-block and robustness splits described in `notes/splits.md`.
2. **Training/eval, on Kaggle.** `kaggle/yolov8s/` and `kaggle/yolo26s/` each hold a
   `generate_run_notebooks*.py` script that renders one train + one eval notebook per
   run (8 runs per architecture: 3-fold Dhaka-trained, 3-fold Chattogram-trained, 2
   leave-one-corridor-out robustness folds) from `configs/hyp*.yaml`. Notebooks are
   pushed to Kaggle and run there (GPU); each eval notebook exports raw COCO-format
   predictions at confidence 0.001.
3. **Pull results.** `kaggle/pull_outputs*.ps1` downloads each run's predictions,
   logs and weights from Kaggle into `results/` (requires your own
   `kaggle auth login`; the scripts never touch credentials).
4. **Offline analysis.** `src/wp4_lib.py` recomputes every metric (AP50, AP50:95,
   recall at a tuned threshold, bootstrap CIs) directly from the saved predictions and
   ground truth in `data/processed/coco_gt/` -- no GPU needed. Entry points:
   `run_wp4_point_estimates*.py`, `run_wp4_mode_groups*.py`, `run_wp4_bootstrap*.py`,
   `run_wp4_confusion.py`, `run_wp4_daynight.py`, `run_wp4_qualitative.py`.
5. **Paper tables/macros.** `src/make_tables.py` and `src/make_yolo26_macros.py`
   write every LaTeX table and inline number the paper uses from `results/*.csv` --
   nothing in `main.tex` is hand-typed.

## Repo layout

- `paper/` -- LaTeX source, generated tables/macros, figures.
- `src/` -- offline analysis (metrics, bootstrap CIs, table/figure generation).
- `kaggle/yolov8s/`, `kaggle/yolo26s/` -- per-architecture notebook generators and
  output-pull scripts.
- `configs/` -- pinned hyperparameters and per-split data YAMLs.
- `notes/` -- dataset audit, split design and other research notes referenced from
  the paper.
- `results/` -- per-run predictions (COCO format, conf$\geq$0.001), logs, and every
  derived CSV/table the paper's numbers come from. Model weights are not included
  here (see below).

`results/weights/` and `results/_kaggle_raw/` are git-ignored (large binaries /
unprocessed staging dumps); `data/` (the raw BadODD images) is git-ignored too --
re-download it from the Zenodo DOI above.

## Reproducing a result end to end

```
python src/run_wp4_point_estimates_yolo26.py   # AP50/recall/thresholds -> results/*_yolo26.csv
python src/run_wp4_mode_groups_yolo26.py       # headline + mode-group comparison
python src/make_yolo26_macros.py               # -> paper/tables/macros_yolo26.tex
```

reproduces every YOLO26s number in the paper from the predictions already checked
into `results/predictions/`, with no Kaggle access required. The equivalent
non-suffixed scripts do the same for the YOLOv8s runs.
