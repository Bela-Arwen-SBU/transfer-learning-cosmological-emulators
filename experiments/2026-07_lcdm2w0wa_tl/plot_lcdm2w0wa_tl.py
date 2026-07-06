"""
plot_lcdm2w0wa_tl.py
LCDM->w0wa transfer learning vs w0wa scratch (Taka->Taka, T500 run1).
Four figures:
  1. median Delta chi2 vs N_train  (log-log; early_1 control sits at ~800)
  2. f(Delta chi2 > 0.2) vs N_train
  3. freeze-depth ladder, median (all resnet_* strategies)
  4. freeze-depth ladder, f(Delta chi2 > 0.2)
Scratch curve includes its existing 250k/500k points.
Usage (on NvWulf login node): python plot_lcdm2w0wa_tl.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path('/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real')
TL_DIR      = BASE / 'chains/taka_lcdm2w0wa_tl_T500_run1_clip90_intdim362/metrics'
SCRATCH_DIR = BASE / 'chains/taka_w0wa_scratch_T500_run1_clip90_intdim362/metrics'

CURVES = [
    {'label': 'w0wa scratch',            'dir': SCRATCH_DIR, 'pattern': 'metrics_N{n}.txt',
     'color': 'black',   'marker': 'o', 'linestyle': '-'},
    {'label': 'TL none (full finetune)', 'dir': TL_DIR, 'pattern': 'metrics_none_N{n}.txt',
     'color': '#F18F01', 'marker': 's', 'linestyle': '-'},
    {'label': 'TL late_1',               'dir': TL_DIR, 'pattern': 'metrics_late_1_N{n}.txt',
     'color': '#2E86AB', 'marker': '^', 'linestyle': '-'},
    {'label': 'TL resnet_123',           'dir': TL_DIR, 'pattern': 'metrics_resnet_123_N{n}.txt',
     'color': '#7B2D8E', 'marker': 'D', 'linestyle': '--'},
    {'label': r'TL early_1 ($w_0 w_a$-blind control)', 'dir': TL_DIR, 'pattern': 'metrics_early_1_N{n}.txt',
     'color': 'gray',    'marker': 'x', 'linestyle': ':'},
]

LADDER_CURVES = [
    {'label': 'none',       'dir': TL_DIR, 'pattern': 'metrics_none_N{n}.txt',
     'color': '#F18F01', 'marker': 's', 'linestyle': '-'},
    {'label': 'resnet_1',   'dir': TL_DIR, 'pattern': 'metrics_resnet_1_N{n}.txt',
     'color': '#2E86AB', 'marker': '^', 'linestyle': '-'},
    {'label': 'resnet_3',   'dir': TL_DIR, 'pattern': 'metrics_resnet_3_N{n}.txt',
     'color': '#2E86AB', 'marker': 'v', 'linestyle': '--'},
    {'label': 'resnet_12',  'dir': TL_DIR, 'pattern': 'metrics_resnet_12_N{n}.txt',
     'color': '#7B2D8E', 'marker': '^', 'linestyle': '-'},
    {'label': 'resnet_23',  'dir': TL_DIR, 'pattern': 'metrics_resnet_23_N{n}.txt',
     'color': '#7B2D8E', 'marker': 'v', 'linestyle': '--'},
    {'label': 'resnet_123', 'dir': TL_DIR, 'pattern': 'metrics_resnet_123_N{n}.txt',
     'color': 'gray',    'marker': 'D', 'linestyle': ':'},
    {'label': 'w0wa scratch', 'dir': SCRATCH_DIR, 'pattern': 'metrics_N{n}.txt',
     'color': 'black',   'marker': 'o', 'linestyle': '-'},
]

N_TRAIN_SIZES = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]

# ── plot settings ─────────────────────────────────────────────────────────────
plt.rcParams['mathtext.fontset']    = 'stix'
plt.rcParams['font.family']         = 'STIXGeneral'
plt.rcParams['text.usetex']         = False
plt.rcParams['xtick.bottom']        = True
plt.rcParams['xtick.top']           = False
plt.rcParams['ytick.right']         = False
plt.rcParams['axes.edgecolor']      = 'black'
plt.rcParams['axes.linewidth']      = 1.0
plt.rcParams['axes.grid']           = True
plt.rcParams['grid.linewidth']      = 0.0
plt.rcParams['grid.alpha']          = 0.18
plt.rcParams['grid.color']          = 'lightgray'
plt.rcParams['legend.labelspacing'] = 0.77
plt.rcParams.update({
    'font.size':       18,
    'axes.labelsize':  26,
    'axes.titlesize':  28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 15,
})

# ── load metrics ──────────────────────────────────────────────────────────────
def load_metrics(metrics_dir, pattern, sizes):
    ns, medians, fracs_02 = [], [], []
    for n in sizes:
        fpath = metrics_dir / pattern.format(n=n)
        if not fpath.exists():
            continue
        data = {}
        for line in fpath.read_text().splitlines():
            k, v = line.split()
            data[k] = float(v)
        ns.append(n)
        medians.append(data['median_chi2'])
        fracs_02.append(data['n_outliers_0p2'] / data['n_test'])
    return np.array(ns), np.array(medians), np.array(fracs_02)

def style_xaxis(ax):
    ax.set_xscale('log')
    ax.set_xlabel(r'$N_\mathrm{train}$')
    xtick_vals   = [n / 1e3 for n in N_TRAIN_SIZES]
    xtick_labels = ['10k', '25k', '50k', '100k', '250k', '500k']
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

# ── figure 1: median chi2 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for curve in CURVES:
    ns, medians, _ = load_metrics(curve['dir'], curve['pattern'], N_TRAIN_SIZES)
    if len(ns) == 0:
        print(f'[skip] no data found for {curve["label"]}')
        continue
    print(f'\n{curve["label"]}')
    for n, m in zip(ns, medians):
        print(f'  N={n:>9,}  median={m:.4g}')
    ax.plot(ns / 1e3, medians,
            label=curve['label'],
            color=curve['color'], marker=curve['marker'], linestyle=curve['linestyle'],
            linewidth=2.5, markersize=9)

ax.set_yscale('log')
style_xaxis(ax)
ax.set_ylabel(r'median $\Delta\chi^2$')
ax.set_title('LCDM$\\to$w0wa TL vs scratch — Taka, T500 run1')
ax.legend(loc='center right', fontsize=13)
plt.tight_layout()
outpath = BASE / 'chains/lcdm2w0wa_tl_vs_scratch_median.pdf'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'\nSaved: {outpath}')

# ── figure 2: outlier fraction ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for curve in CURVES:
    ns, _, fracs_02 = load_metrics(curve['dir'], curve['pattern'], N_TRAIN_SIZES)
    if len(ns) == 0:
        continue
    ax.plot(ns / 1e3, fracs_02,
            label=curve['label'],
            color=curve['color'], marker=curve['marker'], linestyle=curve['linestyle'],
            linewidth=2.5, markersize=9)

ax.axhline(0.10, color='gray', linestyle='--', linewidth=1.2, label='f = 0.10')
style_xaxis(ax)
ax.set_ylabel(r'$f(\Delta\chi^2 > 0.2)$')
ax.set_title('LCDM$\\to$w0wa TL vs scratch — Taka, T500 run1')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.legend(loc='upper right', fontsize=13)
plt.tight_layout()
outpath = BASE / 'chains/lcdm2w0wa_tl_vs_scratch_outlier_frac.pdf'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Saved: {outpath}')

# ── figure 3: freeze-depth ladder ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for curve in LADDER_CURVES:
    ns, medians, _ = load_metrics(curve['dir'], curve['pattern'], N_TRAIN_SIZES)
    if len(ns) == 0:
        continue
    ax.plot(ns / 1e3, medians,
            label=curve['label'],
            color=curve['color'], marker=curve['marker'], linestyle=curve['linestyle'],
            linewidth=2.5, markersize=9)

ax.set_yscale('log')
style_xaxis(ax)
ax.set_ylabel(r'median $\Delta\chi^2$')
ax.set_title('Freeze-depth ladder — Taka, T500 run1')
ax.legend(loc='upper right', fontsize=13)
plt.tight_layout()
outpath = BASE / 'chains/lcdm2w0wa_tl_ladder_median.pdf'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Saved: {outpath}')

# ── figure 4: freeze-depth ladder, outlier fraction ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for curve in LADDER_CURVES:
    ns, _, fracs_02 = load_metrics(curve['dir'], curve['pattern'], N_TRAIN_SIZES)
    if len(ns) == 0:
        continue
    ax.plot(ns / 1e3, fracs_02,
            label=curve['label'],
            color=curve['color'], marker=curve['marker'], linestyle=curve['linestyle'],
            linewidth=2.5, markersize=9)

ax.axhline(0.10, color='gray', linestyle='--', linewidth=1.2, label='f = 0.10')
style_xaxis(ax)
ax.set_ylabel(r'$f(\Delta\chi^2 > 0.2)$')
ax.set_title('Freeze-depth ladder — Taka, T500 run1')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.legend(loc='upper right', fontsize=13)
plt.tight_layout()
outpath = BASE / 'chains/lcdm2w0wa_tl_ladder_outlier_frac.pdf'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Saved: {outpath}')
