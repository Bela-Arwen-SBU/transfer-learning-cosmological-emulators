# Experiment: LCDM → w0wa Transfer Learning (Takahashi)

## Overview
Tests whether a pretrained LCDM cosmic shear emulator reduces the training
data needed for a w0wa emulator, versus training from scratch. Both arms use
Takahashi halofit throughout, so the only change is the cosmology extension.

Run 2026-07-04/05 on NvWulf.

---

## Setup

| Setting | Value |
|---------|-------|
| Base model | LCDM taka run4, N=500k, ResMLP, INT_DIM_RES=362, bs32 |
| Base checkpoint | `chains/taka_scratch_T500_run4_clip90_intdim362_bs32/models/N500000/` |
| Target data | w0wa taka run1 (train T500; valid/test T250 cut2p5pct) |
| Baseline | w0wa taka scratch sweep (same data, same test set) |
| Training | lr 1e-3, batch 32, 1500 epochs |
| N_train | 10k, 25k, 50k, 100k |
| Strategies | none, late_1, early_1, resnet_1/3/12/23/123 |

### Input padding (15 → 17)
The base model takes 15 inputs; w0wa adds w and w0pwa. Model input order
follows the yaml `ord` list, which appends w, w0pwa at positions 15, 16.
`pad_lcdm_to_w0wa.py` appends two zero columns to `model.0.weight`, so the
padded model initially reproduces the LCDM prediction exactly and learns the
w0/wa response during fine-tuning.

The companion `.h5` must also be padded (`pad_h5_lcdm_to_w0wa.py`): in TL
mode the trainer loads `sample_mean`/`sample_std` from the pretrained h5.
Note the stored `sample_std` carries a 5x factor from the save step; the
appended entries replicate it.

---

## Results

Median Δχ² on the shared test set (n=41,063):

| N_train | TL none | late_1 | early_1 | scratch |
|---------|---------|--------|---------|---------|
| 10k  | 0.217 | 0.235 | 889 | 0.977 |
| 25k  | 0.073 | 0.125 | 758 | 0.303 |
| 50k  | 0.056 | 0.096 | 786 | 0.106 |
| 100k | 0.042 | 0.076 | 783 | 0.063 |

f(Δχ² > 0.2), TL none vs scratch: 0.52/0.94 (10k), 0.25/0.63 (25k),
0.20/0.33 (50k), 0.16/0.22 (100k).

Freeze-depth ladder, median Δχ²:

| N_train | none | resnet_1 | resnet_3 | resnet_12 | resnet_23 | resnet_123 |
|---------|------|----------|----------|-----------|-----------|------------|
| 10k  | 0.217 | 0.340 | 0.248 | 1.19  | 0.458 | 6.94 |
| 25k  | 0.073 | 0.115 | 0.137 | 0.339 | 0.296 | 3.17 |
| 50k  | 0.056 | 0.071 | 0.087 | 0.181 | 0.218 | 2.55 |
| 100k | 0.042 | 0.048 | 0.066 | 0.124 | 0.172 | 2.13 |

### Findings
- Full fine-tuning beats scratch at every N tested. The LCDM base is worth
  roughly 2.5-4x in w0wa training data, largest at small N.
- Freezing degrades performance monotonically with the number of frozen
  blocks. All three blocks frozen fails outright: input/output surgery alone
  cannot express the new physics.
- Depth matters, and flips with N: at 10k freezing the late block(s) is
  cheaper; from 25-50k up freezing the early block(s) is cheaper. The
  crossover N grows with the amount frozen. resnet_12 at 10k is worse than
  scratch.
- early_1 (frozen input) is the blind control: the w0/wa weights stay at
  zero, so it measures the cost of ignoring dark energy entirely (Δχ² ~ 800,
  independent of N). Caveat: a frozen input layer also cannot absorb the
  normalization rescale, so this number bundles both effects.

### Caveats
- Single seed per configuration.
- Fixed 1500 epochs at all N (identical in both arms, so the comparison is
  internally consistent).
- Full run1 prior; no prior softening or high-Ωm cut applied yet.
- mean_chi2 is dominated by a catastrophic outlier tail in all arms; use
  medians and outlier fractions.

---

## Files

```
pad_lcdm_to_w0wa.py         pad the .pt checkpoint 15 -> 17 inputs
pad_h5_lcdm_to_w0wa.py      pad sample_mean/sample_std in the companion .h5
run_tl_sweep_lcdm2w0wa_taka_T500_run1_clip90_intdim362_lr3_bs32.sh
submit_tl_sweep_lcdm2w0wa.sh    sbatch wrapper; takes strategy as argument
plot_lcdm2w0wa_tl.py        three figures: median, outlier fraction, ladder
lcdm2w0wa_tl_vs_scratch_median.pdf
lcdm2w0wa_tl_vs_scratch_outlier_frac.pdf
```

Metrics and models on NvWulf:
`chains/taka_lcdm2w0wa_tl_T500_run1_clip90_intdim362/`
