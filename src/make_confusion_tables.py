"""
Renders the E1/E2 confusion matrices as native LaTeX (gridded tables with
\\cellcolor-shaded cells) instead of matplotlib-generated PDFs, from
results/confusion_e1.csv and results/confusion_e2.csv (written by
run_wp4_confusion.py). Writes paper/tables/confusion_tables.tex.

Row-normalised the same way run_wp4_confusion.py's plot did: each of the 7
headline-class rows divided by that row's total (background row excluded, since
it has no "true" GT to normalise against). Cell shown as a rounded percentage,
blank if <1%, shaded blue!<percentage> so the diagonal reads as the darkest cells
-- same information as the old heatmap PDF, same colour semantics, now typeset
directly by LaTeX so it matches the paper's fonts exactly.
"""
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'paper', 'tables')

ABBREV = {
    'auto_rickshaw': 'AR', 'bus': 'BU', 'car': 'CA', 'motorbike': 'MB',
    'person': 'PE', 'three_wheeler': 'TW', 'truck': 'TR', 'background': 'BG',
}
ROWS = ['auto_rickshaw', 'bus', 'car', 'motorbike', 'person', 'three_wheeler', 'truck']
COLS = ROWS + ['background']


def read_matrix(name):
    with open(os.path.join(RESULTS_DIR, f'confusion_{name}.csv')) as f:
        rows = list(csv.reader(f))
    header = rows[0][1:]
    matrix = {}
    for r in rows[1:]:
        matrix[r[0]] = {c: int(v) for c, v in zip(header, r[1:])}
    return matrix


def render_table(name, title):
    matrix = read_matrix(name)
    lines = [
        r'\begin{minipage}{0.48\linewidth}',
        r'\centering',
        r'\scriptsize',
        f'{title}\\\\[2pt]',
        r'\setlength{\tabcolsep}{2pt}',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\begin{tabular}{|l|' + 'c|' * len(COLS) + '}',
        r'\hline',
        r'\textbf{True}$\backslash$\textbf{Pred} & ' + ' & '.join(f'\\textbf{{{ABBREV[c]}}}' for c in COLS) + r' \\',
        r'\hline',
    ]
    for r in ROWS:
        row_sum = sum(matrix[r][c] for c in COLS)
        cells = []
        for c in COLS:
            pct = round(100 * matrix[r][c] / row_sum) if row_sum > 0 else 0
            if pct < 1:
                cells.append('')
            else:
                cells.append(f'\\cellcolor{{blue!{pct}}}{pct}')
        lines.append(f'\\textbf{{{ABBREV[r]}}} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines += [
        r'\end{tabular}',
        r'\end{minipage}',
    ]
    return lines


def main():
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
    ]
    lines += render_table('e1', r'\textbf{E1} (Dhaka-trained, 3 folds summed)')
    lines.append(r'\hfill')
    lines += render_table('e2', r'\textbf{E2} (Chattogram-trained, out-of-fold pooled)')
    lines += [
        r'\caption{Row-normalised confusion matrices (headline classes + background; '
        r'IoU $\geq$0.5 greedy match) for E1 (left) and E2 (right), both on '
        r'\texttt{ctg\_pooled\_eval}. Cells are row percentages (blank $<$1\%); a GT '
        r'box with no matching prediction counts as background in that row (a miss), '
        r'a prediction with no matching GT counts as background in that column (a '
        r'false positive). AR=auto-rickshaw, BU=bus, CA=car, MB=motorbike, PE=person, '
        r'TW=three-wheeler, TR=truck, BG=background.}',
        r'\label{tab:confusion}',
        r'\end{table*}',
    ]
    out_path = os.path.join(TABLES_DIR, 'confusion_tables.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
