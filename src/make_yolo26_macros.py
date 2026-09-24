"""
Writes the YOLO26s architecture-robustness-check macros to paper/tables/macros_yolo26.tex
(its own file, so make_tables.py rewriting macros.tex never drops them), from
results/mode_group_summary_yolo26.csv, results/headline_per_class_point[_yolo26].csv and
results/per_class_ap_yolo26.csv. Same "nothing hand-typed in main.tex" convention as
make_tables.py -- run this any time the YOLO26s results change
(src/run_wp4_point_estimates_yolo26.py, src/run_wp4_mode_groups_yolo26.py first).

LaTeX macro names cannot contain digits, so the YOLO26s variants of the existing
\\GapFormalMotorised-style macros are suffixed "AltArch" (alternative architecture)
rather than "Y26".
"""
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'paper', 'tables')


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))


def get_ap(rows, run_id, target, cls):
    for r in rows:
        if r['run_id'] == run_id and r['target'] == target and r['class'] == cls:
            return float(r['ap50'])
    raise KeyError((run_id, target, cls))


def macro(name, value):
    return f'\\newcommand{{\\{name}}}{{{value}}}'


def main():
    mode_groups = read_csv('mode_group_summary_yolo26.csv')
    by_group = {r['mode_group']: r for r in mode_groups}

    macros = []
    name_map = {
        'formal_motorised': 'FormalMotorised',
        'pedestrian': 'Pedestrian',
        'informal_motorised': 'InformalMotorised',
        'non_motorised': 'NonMotorised',
    }
    for group, key in name_map.items():
        r = by_group[group]
        macros.append(macro(f'Gap{key}AltArch', f"{float(r['ap50_gap']):.3f}"))
        macros.append(macro(f'RecallGap{key}AltArch', f"{float(r['recall_gap']):.3f}" if r['recall_gap'] != '' else 'TODO'))

    ap_rows = read_csv('per_class_ap_yolo26.csv')
    e1_mean_h1 = sum(get_ap(ap_rows, f'e1_{k}_y26', 'ctg_robustness_holdout_bohoddarhat1_eval', 'three_wheeler')
                      for k in range(3)) / 3
    e1_mean_h2 = sum(get_ap(ap_rows, f'e1_{k}_y26', 'ctg_robustness_holdout_bohoddarhat2_eval', 'three_wheeler')
                      for k in range(3)) / 3
    rob_a = get_ap(ap_rows, 'rob_a_y26', 'ctg_robustness_holdout_bohoddarhat1_eval', 'three_wheeler')
    rob_b = get_ap(ap_rows, 'rob_b_y26', 'ctg_robustness_holdout_bohoddarhat2_eval', 'three_wheeler')
    macros.append(macro('RobAGapAltArch', f'{rob_a - e1_mean_h1:+.3f}'))
    macros.append(macro('RobBGapAltArch', f'{rob_b - e1_mean_h2:+.3f}'))

    headline = ['auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck']
    for suffix, fname in [('', 'headline_per_class_point.csv'), ('AltArch', 'headline_per_class_point_yolo26.csv')]:
        by_class = {r['class']: r for r in read_csv(fname)}
        e1 = sum(float(by_class[c]['e1_ap50_mean']) for c in headline) / len(headline)
        e2 = sum(float(by_class[c]['e2_ap50_pooled']) for c in headline) / len(headline)
        macros.append(macro(f'HeadlineMeanEOne{suffix}', f'{e1:.3f}'))
        macros.append(macro(f'HeadlineMeanETwo{suffix}', f'{e2:.3f}'))

    out_path = os.path.join(TABLES_DIR, 'macros_yolo26.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(macros) + '\n')
    print(f'Wrote {len(macros)} YOLO26s macros to {out_path}')
    for m in macros:
        print(' -', m)


if __name__ == '__main__':
    main()
