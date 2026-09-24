"""
Generates every LaTeX table and inline-number macro the paper uses, directly from
results/*.csv. Nothing in paper/main.tex is hand-typed; it only \input{}s the files
this script writes. Re-run any time results/ changes (e.g. once the bootstrap CI job
in run_wp4_bootstrap.py finishes covering more classes).
"""
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'paper', 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

PRETTY = {
    'auto_rickshaw': 'Auto-rickshaw', 'bus': 'Bus', 'car': 'Car', 'motorbike': 'Motorbike',
    'person': 'Person', 'three_wheeler': 'Three-wheeler (cycle-rick.)', 'truck': 'Truck',
    'bicycle': 'Bicycle', 'cart_vehicle': 'Cart vehicle', 'priority_vehicle': 'Priority vehicle',
    'construction_vehicle': 'Construction vehicle',
}
GROUP_PRETTY = {
    'formal_motorised': 'Formal motorised', 'pedestrian': 'Pedestrian',
    'informal_motorised': 'Informal motorised', 'non_motorised': 'Non-motorised',
}
HEADLINE = ['auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck']


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))


def fmt_ci(gap, lo, hi):
    if lo == '' or hi == '' or lo is None:
        return f'{float(gap):+.3f} (CI pending)'
    return f'{float(gap):+.3f} [{float(lo):+.3f}, {float(hi):+.3f}]'


def main():
    headline_pt = {r['class']: r for r in read_csv('headline_per_class_point.csv')}
    gap_ci = {r['class']: r for r in read_csv('transfer_gap.csv')}
    mode_groups = read_csv('mode_group_summary.csv')

    # --- Table 1: per-class headline result (AP50 + recall, E1 vs E2, on ctg_pooled_eval) ---
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Dhaka-trained (E1, mean of 3 folds) vs.\ Chattogram-trained reference (E2, '
        r'out-of-fold pooled), evaluated on the identical 1{,}121-image Chattogram test set. '
        r'AP50 gap = E2 $-$ E1, with 95\% image-level bootstrap CI (300 resamples).}',
        r'\label{tab:headline}',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{lrrrl}',
        r'\toprule',
        r'Class & $n$ & E1 AP50 & E2 AP50 & AP50 gap [95\% CI] \\',
        r'\midrule',
    ]
    for c in HEADLINE:
        r = headline_pt[c]
        g = gap_ci.get(c)
        ci_str = fmt_ci(r['ap50_gap_point'], g['ap50_gap_ci_lo'] if g else '', g['ap50_gap_ci_hi'] if g else '')
        lines.append(f"{PRETTY[c]} & {r['n_gt_ctg']} & {float(r['e1_ap50_mean']):.3f} & "
                     f"{float(r['e2_ap50_pooled']):.3f} & {ci_str} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
    with open(os.path.join(TABLES_DIR, 'headline_table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # --- Table 2: mode-group aggregation ---
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r"\caption{Transfer gap aggregated by mode group, weighted by each class's "
        r'Chattogram instance count.}',
        r'\label{tab:modegroup}',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Mode group & $n$ & E1 AP50 & E2 AP50 & AP50 gap \\',
        r'\midrule',
    ]
    order = ['formal_motorised', 'pedestrian', 'informal_motorised', 'non_motorised']
    by_group = {r['mode_group']: r for r in mode_groups}
    for g in order:
        r = by_group[g]
        lines.append(f"{GROUP_PRETTY[g]} & {r['total_n_gt_ctg']} & {float(r['e1_ap50_mean']):.3f} & "
                     f"{float(r['e2_ap50_pooled']):.3f} & {float(r['ap50_gap']):+.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(TABLES_DIR, 'mode_group_table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # --- Inline macros for prose ---
    def macro(name, value):
        return f'\\newcommand{{\\{name}}}{{{value}}}'

    macros = []
    for g in order:
        r = by_group[g]
        key = ''.join(w.capitalize() for w in g.split('_'))
        macros.append(macro(f'Gap{key}', f"{float(r['ap50_gap']):.3f}"))
        macros.append(macro(f'RecallGap{key}', f"{float(r['recall_gap']):.3f}" if r['recall_gap'] != '' else 'TODO'))
        macros.append(macro(f'N{key}', r['total_n_gt_ctg']))
    ar = gap_ci.get('auto_rickshaw')
    if ar:
        macros.append(macro('ArAPgapCILo', f"{float(ar['ap50_gap_ci_lo']):.3f}"))
        macros.append(macro('ArAPgapCIHi', f"{float(ar['ap50_gap_ci_hi']):.3f}"))
    with open(os.path.join(TABLES_DIR, 'macros.tex'), 'w') as f:
        f.write('\n'.join(macros) + '\n')

    # --- Table 3: mode-group aggregation with bootstrap CI, if available ---
    ci_path = os.path.join(RESULTS_DIR, 'mode_group_ci.csv')
    if os.path.exists(ci_path):
        ci_by_group = {r['mode_group']: r for r in read_csv('mode_group_ci.csv')}
        lines = [
            r'\begin{table}[t]',
            r'\centering',
            r'\caption{Mode-group AP50 and recall transfer gaps with 95\% bootstrap CI '
            r'(300 shared image resamples across all 11 classes).}',
            r'\label{tab:modegroupci}',
            r'\resizebox{\columnwidth}{!}{%',
            r'\begin{tabular}{lll}',
            r'\toprule',
            r'Mode group & AP50 gap [95\% CI] & Recall gap [95\% CI] \\',
            r'\midrule',
        ]
        for g in order:
            r = by_group[g]
            c = ci_by_group[g]
            ap_ci = fmt_ci(r['ap50_gap'], c['ap50_gap_ci_lo'], c['ap50_gap_ci_hi'])
            rec_ci = fmt_ci(r['recall_gap'], c['recall_gap_ci_lo'], c['recall_gap_ci_hi'])
            lines.append(f"{GROUP_PRETTY[g]} & {ap_ci} & {rec_ci} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
        with open(os.path.join(TABLES_DIR, 'mode_group_ci_table.tex'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        for g in order:
            c = ci_by_group[g]
            key = ''.join(w.capitalize() for w in g.split('_'))
            with open(os.path.join(TABLES_DIR, 'macros.tex'), 'a') as f:
                f.write(macro(f'{key}APgapCILo', f"{float(c['ap50_gap_ci_lo']):.3f}") + '\n')
                f.write(macro(f'{key}APgapCIHi', f"{float(c['ap50_gap_ci_hi']):.3f}") + '\n')

    # --- Table 4: day/night stratified mode-group gaps ---
    dn_rows = read_csv('day_night_stratified.csv')
    dn_by_cond_class = {(r['condition'], r['class']): r for r in dn_rows}
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{AP50 transfer gap by mode group, stratified by day vs.\ night. '
        r'Restricted to headline classes only (below-threshold classes not included: '
        r'non-motorised = \texttt{three\_wheeler} only here, vs.\ + bicycle/cart in Table 2).}',
        r'\label{tab:daynight}',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Mode group & Day AP50 gap & Night AP50 gap \\',
        r'\midrule',
    ]
    for g in order:
        classes = [c for c in {
            'formal_motorised': ['car', 'bus', 'truck', 'motorbike'],
            'pedestrian': ['person'],
            'informal_motorised': ['auto_rickshaw'],
            'non_motorised': ['three_wheeler'],
        }[g]]
        day_num = sum(float(dn_by_cond_class[('day', c)]['ap50_gap']) * float(dn_by_cond_class[('day', c)]['n_gt']) for c in classes)
        day_den = sum(float(dn_by_cond_class[('day', c)]['n_gt']) for c in classes)
        night_num = sum(float(dn_by_cond_class[('night', c)]['ap50_gap']) * float(dn_by_cond_class[('night', c)]['n_gt']) for c in classes)
        night_den = sum(float(dn_by_cond_class[('night', c)]['n_gt']) for c in classes)
        lines.append(f"{GROUP_PRETTY[g]} & {day_num/day_den:+.3f} & {night_num/night_den:+.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(TABLES_DIR, 'day_night_table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote {len(os.listdir(TABLES_DIR))} files to {TABLES_DIR}')
    for fn in os.listdir(TABLES_DIR):
        print(' -', fn)


if __name__ == '__main__':
    main()
