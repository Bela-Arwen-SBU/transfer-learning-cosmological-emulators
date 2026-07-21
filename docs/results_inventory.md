# Results Inventory — snapshot 2026-07-15

Every known result, its source, and confidence that it is correctly understood.
Annotate freely: mark ✓ (confirmed), ✗ (wrong), or fill in blanks.
Compiled by Claude from repo READMEs, saved memory, notebooks, and the era
folders in `~/Desktop/Research/Projects/Transfer Learning/`.

Thesis outline this feeds (Ch 5 = three transfer axes):
5.1 common setup · 5.2 accuracy transfer (class paper) · 5.3 prescription
transfer (taka→mead) · 5.4 parameter-space transfer (ΛCDM→w0wa) ·
5.5 cross-axis discussion (incl. correction-network negative result) ·
5.6 limitations.

---

## Axis 3 — ΛCDM→w0wa (July 2026) — HIGH confidence
Source: `experiments/2026-07_lcdm2w0wa_tl/README.md` + saved memory.

- TL (full fine-tune) vs scratch, median Δχ², shared test set (n=41,063):
  | N | TL none | scratch |
  |---|---|---|
  | 10k | 0.217 | 0.977 |
  | 25k | 0.073 | 0.303 |
  | 50k | 0.056 | 0.106 |
  | 100k | 0.042 | 0.063 |
  → TL wins at every N; base worth ~2.5–4× training data.
- f(Δχ²>0.2) TL/scratch: 0.52/0.94 (10k) → 0.16/0.22 (100k).
- Freeze ladder (median): degradation monotone in frozen depth;
  resnet_123 fails outright (2–7); crossover: resnet_3 better at 10k,
  resnet_1 better ≥25k, widening with N.
- early_1 = blind control, Δχ²≈800 flat in N (caveat: bundles
  w0wa-blindness + inability to absorb 5× normalization rescale).
- Padding surgery: 15→17 inputs, zero cols appended per yaml `ord`
  (w, w0pwa at 15,16); companion .h5 sample_mean/std padded (5× bake-in).
- w0wa scratch baseline complete since April (N=10k–500k),
  `chains/taka_w0wa_scratch_T500_run1_clip90_intdim362/metrics/`.
- Caveats (state in thesis): single seed (seed rep 2 of the champion in
  flight, job 48728), fixed 1500 epochs, mean χ² dominated by outlier
  tail → medians + outlier fractions; full run1 prior.

### ✓ COLMASK: HARVESTED (Claude, 2026-07-18) — supersedes the box above
- **46258 (early_1+mask) DONE** 2026-07-14, 4/4. Median Δχ² / f(>0.2) / f(>1):
  10k 0.184/0.475/0.171 · 25k 0.087/0.282/0.101 · 50k 0.064/0.221/0.082 ·
  100k 0.052/0.184/0.068. Beat `none` at 10k, trailed ~20–25% above.
- **46259 (late_4+mask) crashed at preflight** (torch import; concurrent
  stop/start_cocoa collision — stagger paired submissions). Resubmit
  completed 2026-07-15 but result CONFOUNDED: median ~7e4 at all N = the
  5× normalization rescale with nothing trainable to absorb it. Not physics;
  do not quote except as the methods anecdote (predicted in advance).
- **RESCALED reruns (jobs 47364/65, `rescale_padded_ckpt.py` ×5 on the 15
  locked cols), output `chains/taka_lcdm2w0wa_tl_colmask_rescaled_*`:**
  - **early_1+mask rescaled = CHAMPION at every N** (thesis headline for 5.4,
    pending one control): 10k **0.083**/0.284/0.107 · 25k **0.053**/0.197/0.075 ·
    50k **0.041**/0.159/0.064 · 100k **0.035**/0.133/0.055. Beats `none`
    everywhere (2.6× at 10k); 100k point matches scratch@250k (0.037).
  - **late_4+mask rescaled (726 params)**: median 1411/1238/1156/1066
    (10k→100k), 100% outliers, deconfounded. Physics probe answer: dark
    energy cannot ride frozen ΛCDM feature directions; worse than the
    blind control (which at least adapts its trunk).
- **θ-corr r=55 on this axis** (job 47373, frozen padded17 base, e500,
  `chains/taka_lcdm2w0wa_corr_thetar55_*`): median 14.1/7.9/7.0/6.3,
  f(>0.2)=1.00 at all N. Third frozen-base failure → **extension law**:
  resnet_123 (2–7) + late_4+mask (~1.2e3) + θ-corr (6–14) all fail;
  every base-adapting method succeeds. (Loophole not yet closed: single
  r=55; optional r≈256 run would rule out capacity.)
- ✓ **CONTROL LANDED (job 48745, harvested 2026-07-20) — 5.4 headline
  gate LIFTED, verdict = the "tie" branch, refined.** `none` from the
  rescaled ckpt: 0.0830 / 0.0559 / 0.0459 (10k/25k/50k; N100k finishing,
  harvest `chains/taka_lcdm2w0wa_tl_rescaledbase_*`). Vs colmask-rescaled
  0.0835 / 0.0527 / 0.0415: tie at 10k, colmask ahead 6% (25k) and
  10% (50k), edge GROWING with N. Canonical 5.4 framing (three nested
  results): (1) warm start beats scratch 2.5–4×; (2) **exact-scaled init
  is the dominant effect** (~2.6× over naive padding at 10k, for any
  strategy — the 5× bake-in fix is the program's biggest single win);
  (3) locking the embedding costs nothing, adds a small deterministic
  edge at scale, and uniquely guarantees the base is byte-preserved.
  Given determinism (below), the 6–10% gaps are exact, not seed noise —
  but caveat: deterministic-under-rerun ≠ robust-under-data-reordering.
  - Harvest the N100k rescaledbase number when the job finishes
    (expected ~0.036–0.040 vs colmask 0.0349).
mean_chi2 2.380766e+05
median_chi2 3.873045e-02
n_outliers_0p2 5898
n_outliers_1 2380
n_test 41063
  - ~~Job 48728 seed rep~~ ✓ → became the DETERMINISM finding (Axis 2
    section): identical to 7 sig figs; no seed scatter exists for TL runs.
  - Experiment A pair (paired scratch + direct-TL, N=1k–50k, 12 metrics,
    `chains/mead_{scratch,tl_none}_paired_smallN_*`) — scripts generated
    2026-07-18; verify with `sq` that both sbatches went in.

## Axis 2 — taka→mead (Feb–Jun 2026) — MEDIUM confidence
Sources: `experiments/2026-02_roman_baseline/` notebooks + era folders 06, 07.

- 16-strategy TL sweep (Feb notebook `roman_cs_analysis (5).ipynb`):
  none 0.0219 · early_1 0.0346 · early_2 0.064 · early_3 0.106 ·
  early_4 0.60 · early_5 24.4 · early_6 110 · late_1 0.082 ·
  late_2 0.092 · late_3 0.331 · late_4 0.527 · late_5 0.92.
  Same never-freeze-the-extremes story as w0wa.
  - Sorry for the formatting:
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_5sigmafix_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_5sigmafix_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 2.212128e-02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_5sigmafix_from_taka_T500_run4_clip90_intdim362_none_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_5sigmafix_from_taka_T500_run4_clip90_intdim362_none_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 1.542845e-02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_constfiddv_from_taka_T500_run4_clip90_intdim362_early_2_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_constfiddv_from_taka_T500_run4_clip90_intdim362_early_2_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 5.708181e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200/metrics/metrics_fs_early_1.txt:median_chi2 7.590894e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200/metrics/metrics_fs_early_2.txt:median_chi2 2.497289e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200/metrics/metrics_fs_late_1.txt:median_chi2 1.129769e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200/metrics/metrics_fs_late_2.txt:median_chi2 5.639463e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200/metrics/metrics_fs_none.txt:median_chi2 5.129703e-02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128/metrics/metrics_fs_early_1.txt:median_chi2 4.743422e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128/metrics/metrics_fs_early_2.txt:median_chi2 2.026723e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128/metrics/metrics_fs_late_1.txt:median_chi2 4.914090e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128/metrics/metrics_fs_late_2.txt:median_chi2 1.272118e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T200_lr3_bs128/metrics/metrics_fs_none.txt:median_chi2 3.519744e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/metrics/metrics_fs_early_1.txt:median_chi2 2.658606e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/metrics/metrics_fs_early_2.txt:median_chi2 6.387195e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/metrics/metrics_fs_late_1.txt:median_chi2 2.461699e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/metrics/metrics_fs_late_2.txt:median_chi2 5.891093e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/metrics/metrics_fs_none.txt:median_chi2 2.299008e+01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N100000.tx:median_chi2 2.262742e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N10000.txtmedian_chi2 3.575857e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N250000.tx:median_chi2 9.126334e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N25000.txtmedian_chi2 1.264477e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N500000.tx:median_chi2 6.217253e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N50000.txtmedian_chi2 4.888520e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362/metrics/metrics_N5000.txt:median_chi2 5.228573e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N100000.txt:median_chi2 1.050424e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N10000.txt:median_chi2 1.181813e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N250000.txt:median_chi2 5.531264e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N25000.txt:median_chi2 3.935615e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N500000.txt:median_chi2 3.810272e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 2.212128e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_1_lr3_bs128/metrics/metrics_N5000.txt:median_chi2 1.727967e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_2_lr4_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_2_lr4_bs128/metrics/metrics_N50000.txt:median_chi2 4.487474e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_2_lr4_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_2_lr4_bs32/metrics/metrics_N50000.txt:median_chi2 9.280413e-02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 3.160999e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr4_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr4_bs128/metrics/metrics_N50000.txt:median_chi2 1.629880e+00
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr4_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_early_3_lr4_bs32/metrics/metrics_N50000.txt:median_chi2 6.604403e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_fz_sweep
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_fz_sweep/metrics/metrics_fz_early_1.txt:median_chi2 2.212128e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_fz_sweep/metrics/metrics_fz_early_2.txt:median_chi2 4.888520e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_fz_sweep/metrics/metrics_fz_early_3.txt:median_chi2 3.160999e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_fz_sweep/metrics/metrics_fz_none.txt:median_chi2 1.542845e-02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N100000.txt:median_chi2 2.831411e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N10000.txt:median_chi2 1.621227e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N250000.txt:median_chi2 1.816944e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N25000.txt:median_chi2 7.520501e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N500000.txt:median_chi2 1.507713e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 5.247297e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_input_output_lr3_bs128/metrics/metrics_N5000.txt:median_chi2 3.592134e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N100000.txt:median_chi2 2.664948e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N10000.txt:median_chi2 1.991913e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N250000.txt:median_chi2 1.942218e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N25000.txt:median_chi2 8.018874e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N500000.txt:median_chi2 1.549657e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 4.369996e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_late_1_lr3_bs128/metrics/metrics_N5000.txt:median_chi2 3.739147e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N100000.txt:median_chi2 9.499181e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N10000.txt:median_chi2 1.322435e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N250000.txt:median_chi2 5.496860e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N25000.txt:median_chi2 3.856695e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N500000.txt:median_chi2 4.000239e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N50000.txt:median_chi2 1.542845e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr3_bs128/metrics/metrics_N5000.txt:median_chi2 2.202451e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N100000.txt:median_chi2 1.722214e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N10000.txt:median_chi2 1.884321e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N250000.txt:median_chi2 1.041788e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N25000.txt:median_chi2 8.570914e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N500000.txt:median_chi2 6.284783e-03
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N50000.txt:median_chi2 3.060237e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N5000.txt:median_chi2 3.499289e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N100000.txt:median_chi2 5.158191e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N10000.txt:median_chi2 5.402493e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N250000.txt:median_chi2 4.211750e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N25000.txt:median_chi2 2.163164e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N500000.txt:median_chi2 1.550603e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N50000.txt:median_chi2 8.576976e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500_run4_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N5000.txt:median_chi2 1.095527e+00
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N10000.txt:median_chi2 6.034803e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N1000.txt:median_chi2 1.469639e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N25000.txt:median_chi2 2.777501e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N2500.txt:median_chi2 3.485886e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N50000.txt:median_chi2 1.903970e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_none_paired_smallN_T500_run4_clip90_intdim362/metrics/metrics_tl_none_N5000.txt:median_chi2 1.354220e-01
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N100000.txt:median_chi2 1.757767e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N10000.txt:median_chi2 1.244193e+02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N250000.txt:median_chi2 1.285899e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N25000.txt:median_chi2 8.260348e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N500000.txt:median_chi2 3.915972e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N50000.txt:median_chi2 6.553763e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362/metrics/metrics_N5000.txt:median_chi2 4.543031e+02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N100000.txt:median_chi2 3.635957e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N10000.txt:median_chi2 7.665166e+01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N250000.txt:median_chi2 8.436448e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N25000.txt:median_chi2 3.113848e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N500000.txt:median_chi2 5.500559e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N50000.txt:median_chi2 6.401746e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr4_bs32/metrics/metrics_N5000.txt:median_chi2 1.082853e+02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N100000.txt:median_chi2 1.925317e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N10000.txt:median_chi2 2.478918e+02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N250000.txt:median_chi2 8.250790e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N25000.txt:median_chi2 7.509054e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N500000.txt:median_chi2 5.165335e-01
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N50000.txt:median_chi2 5.829997e+00
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_w0wa_tl_from_taka_T500_run1_clip90_intdim362_nofz_lr5_bs32/metrics/metrics_N5000.txt:median_chi2 3.084485e+02
== /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_early_1_N100000.txt:median_chi2 2.472543e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_early_1_N10000.txt:median_chi2 7.149951e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_early_1_N25000.txt:median_chi2 3.839094e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_early_1_N50000.txt:median_chi2 2.842763e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_input_output_N100000.txt:median_chi2 2.496004e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_input_output_N10000.txt:median_chi2 8.696069e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_input_output_N25000.txt:median_chi2 3.800940e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_input_output_N50000.txt:median_chi2 3.084944e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_late_1_N100000.txt:median_chi2 2.260921e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_late_1_N10000.txt:median_chi2 5.759022e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_late_1_N25000.txt:median_chi2 3.920752e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_late_1_N50000.txt:median_chi2 2.896200e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_none_N100000.txt:median_chi2 2.306886e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_none_N10000.txt:median_chi2 6.156417e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_none_N25000.txt:median_chi2 3.158332e-02
/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_tl_taka2taka_T500_run4_clip90_intdim362_bs32/metrics/metrics_none_N50000.txt:median_chi2 2.667511e-02
(.local) (cocoa_bela) [barwen@login1 bela]$ 
  - ✗→exploratory (Claude, 2026-07-18): **NOT thesis-canon regardless of
    settings.** The Feb sweep predates the run4 datasets; per
    `docs/datasets/README_lcdm_taka_run4.md`, all pre-run4 training data
    used consecutive-MCMC sampling with documented autocorrelation and
    poor prior coverage. Present (if at all) as exploratory/appendix with
    that caveat; do not mix its numbers into 5.3 tables.
- Scratch scaling (same notebook, cached):
  taka median 0.1439 (10k) → 0.0012 (1M); mead similar (0.1530 at 10k).
- Era 06 (`06_tl_sweeps_2026-04-05/`): run4-scale TL sweep metrics:
  median Δχ² 0.350 (N=5k), 0.058 (N=50k), 0.018 (N=500k).
  Cache keys tl_taka50k / tl_taka250k suggest base-size comparison exists.
  - ✓ CANON = era-06 run4-scale sweeps (Claude, 2026-07-18). Reasons:
    run4 corrected sampling; same architecture (INT_DIM_RES=362, clip90)
    and dataset lineage as every other canon result; bases from
    `chains/taka_scratch_T500_run4_clip90_intdim362*` (Apr sweeps used the
    N500k base; May taka2taka used the `_bs32` sibling). Test sets: the
    Apr-19 taka2mead sweep script used T250 **cut5pct**; taka2taka (May)
    used **cut2p5pct** — per-sweep, so label each table with its cut.
  - [ ] STILL NEEDS HARVEST for exact per-N canon table: metrics dirs under
    `chains/mead_tl_from_taka_T500_run1_*` / `taka_tl_taka2taka_*` etc.
    Known reference points: best TL 0.015 vs mead scratch 0.018 at N=50k
    (June slides; possibly paired-test numbers — "0.0168 vs 0.0192" flagged
    in Béla's own notes as within possible seed noise, now testable with
    the s2-style reps). Era-06 cached 0.058@50k is a different test set
    than the slide 0.015 — reconcile cuts before quoting both.
- Era 07 (`07_corrections_matched_kmax_2026-06/`):
  - ✓ Matched-kmax diagnostic — SETTINGS + PRECISE NUMBERS (Claude, 2026-07-18):
    200 paired DVs, identical cosmologies, taka AND mead, T=500, at
    k_max ∈ {100 (default), 10 (CAMB boundary), 3.2} h/Mpc.
    Self-comparison (DV change when lowering k_max): k100→k10 ~1e-17
    (machine zero), k100→k3.2 ~1e-11. Cross-comparison (taka↔mead gap,
    median SSD): 1.66e-8 @ k_max=100, 1.66e-8 @ 10, 1.84e-8 @ 3.2.
    → Ch3 §3.3 P8 "gap persists as k_max is lowered" is CONFIRMED as
    written; phrase precisely as "does not shrink" (it is identical at
    k_max=10 and ~10% LARGER at 3.2 — do not write "unchanged").
    Cosmic shear is numerically insensitive to k>10; the taka–mead
    difference is intrinsic to k<10 where both are computed: modeling
    difference, not numerics or extrapolation.
  - ✓ Correction r55: EXPLORATORY as originally run (Claude, 2026-07-18).
    June-8 number (median 3.9, Taka baseline 370 → 94× reduction) was
    e100, explicitly unconverged (LR step at ~epoch 48), and on the
    paired test vs cut2.5 elsewhere — do not quote as final. Its
    diagnostic sequel IS canon: e1500 + flat r-sweep established the
    blind-correction ceiling ≈0.9 median (capacity + convergence ruled
    out). [ ] harvest exact ceiling value from its chains dir if Ch3
    needs a precise number rather than "≈0.9".
    **FINAL correction-family result for the thesis = θ-corr r55, e1500,
    paired test (job 47374, `chains/mead_corr_thetar55_paired_*`):**
    median 3.11/1.95/1.45/1.03/0.685/**0.673** at N=1k/2.5k/5k/10k/25k/50k;
    f(>0.2) 0.98→0.79; f(>1) 0.73→0.42. **Breaks the blind ceiling**
    (0.9→0.67): θ-blindness was the wall, as hypothesized (slide 227).
    Still far above direct TL — the proposal's decision logic completes
    when the Experiment A comparators land.
    → Ch3/ch4 prose promise: present as "ceiling, diagnosed; ceiling,
    broken" — a two-act result, not a bare failure.
  - emulator_film.py (FiLM variant): drafted 2026-06-08, never run in any
    recorded session — exploratory dead code; ignore for Ch5.
- Experiments A/B status: **BOTH DONE.** B = θ-corr above.
  ✓ **A harvested 2026-07-20.** Paired test (n=10k), median Δχ²,
  N = 1k/2.5k/5k/10k/25k/50k:
  scratch 41.9 / 2.15 / 0.360 / 0.109 / 0.0445 / 0.0234;
  direct TL (none, taka N50k base) 1.47 / 0.349 / 0.135 / 0.060 /
  0.0278 / 0.0190. → **Direct TL beats scratch at every N (28× at 1k).**
  The old "TL never beats scratch on taka→mead" was an N=50k-saturation
  artifact, exactly as the proposal slide hypothesized. θ-corr (0.673
  floor) beats scratch only at ≤2.5k and is dominated by direct TL
  everywhere → corrections never competitive with fine-tuning.
  Chains: `mead_{scratch,tl_none}_paired_smallN_T500_run4_clip90_intdim362`.
  This paired small-N table is the natural 5.3 canon figure.
- **DETERMINISM FINDING (s2 rep, 2026-07-20):** colmask-rescaled rerun
  reproduced run 1 to 7 significant figures in every metric — TL-from-
  checkpoint runs are bit-deterministic (no random init; fixed per-epoch
  data ordering, cf. Vivian's 6/15 remark). For 5.6: TL numbers carry no
  run-to-run variance by construction; seed scatter applies only to
  random-init (scratch) runs — a scratch rerun is the right error bar.
- Figures in repo: freeze_sweep_{all,early,late,other,best5}.pdf,
  baseline_results.pdf, chi2_distributions.pdf.

## Axis 1 — AccuracyBoost transfer (class paper era)
## STATUS: EARLY PROOF-OF-CONCEPT. Deprioritize. (Béla, 2026-07-18)
This was the FIRST test: a proof of concept run on a dummy training setup
Evan had built in the LSST-Y1 project folder, asking only whether accuracy
settings changed things and whether TL worked at all. It is the least
interesting of the three axes and should NOT be presented as a headline
result. Béla has slides covering this era if it is kept.
→ Strong argument for cutting or heavily fencing 5.2 (the in/out decision
  with Vivian). If kept, frame explicitly as "proof of concept, different
  survey, inherited scaffolding" and keep it short.

Sources: Béla's outline comments + `configs/xi/*.yaml` (verified 2026-07-18)
+ Vivian's CPIP overview slides, slide 18 (the TL roadmap slide — that
preliminary result is BÉLA'S OWN WORK, done for Vivian, presented by her).

- Claimed: pretrain AccuracyBoost=1.0 (10k) → fine-tune 4.0;
  Early Freezing 2 optimal (22.6% frozen); 2500 samples matches
  10000-sample baseline (median 0.067 vs 0.054, 88.8% vs 89.1% pass)
  → 4× reduction; low-acc baseline insufficient (0.133, 69.8%);
  EF2 converges 2–3× faster (~epoch 150 vs 400–500).

### The 1k vs 2.5k question — RESOLVED (Béla, 2026-07-18)
Not a contradiction; two different criteria against the same 10k baseline:
- **1k** = enough to reach the Δχ² < 0.2 threshold (→ 10× data reduction).
  This is the number on CPIP slide 18 ("only 1k high acc data needed").
- **2.5k** = enough to *fully recover the χ² distribution* (median AND
  pass fraction match the 10k scratch baseline) (→ 4× data reduction).
  This is the number in the outline claim above.
→ For 5.2, headline the 2.5k/4× figure (stronger criterion) and mention
  1k/10× as the weaker threshold-crossing result. State which is which.

### CONFIRMED from `configs/xi/` (three configs, read directly)
| | likelihood `accuracyboost` | `integration_accuracy` | CAMB `AccuracyBoost` | n_train |
|---|---|---|---|---|
| low  | 1.0 | 0   | 1.0 | 10000 |
| high | 4.0 | 1.0 | 2.0 | 10000 |
| TL   | 4.0 | 1.0 | 2.0 | 1000  |

- Architecture: `MLA: MLP` (= ResMLP), `INT_DIM_RES: 256`, 12 inputs,
  780-element DV → **601,626 params** (matches the old figure exactly).

### THREE FLAGS before writing 5.2
1. **This axis is LSST-Y1, NOT Roman.** Configs use
   `lsst_y1.cosmic_shear`, `./projects/lsst_y1/data`, 12 params,
   780-element DV (15 pairs × 2 × 26 θ-bins), intdim 256. The rest of the
   thesis is Roman (17 params, 1080 elements, intdim 362). 5.2 MUST say so
   or the reader will assume Roman throughout. This also bears on the
   5.2 in/out decision with Vivian.
2. **"AccuracyBoost 1→4" is imprecise.** TWO knobs move together: the
   CosmoLike likelihood `accuracyboost` (1.0→4.0, plus
   `integration_accuracy` 0→1.0) and CAMB's own `AccuracyBoost` (1.0→2.0).
   The briefing's "AccuracyBoost 1→4" conflates them. Name both.
3. **Possible config bug:** `xi_emulator_low_accuracy.yaml` sets
   `train_datavectors_file: 'train_datavectors_high_accuracy_200.npy'` —
   a HIGH-accuracy DV file inside the LOW-accuracy config. Either a
   copy-paste artifact or deliberate. Check before quoting that config.
- [ ] Still worth locating the class paper PDF to confirm the median /
      pass-fraction numbers, which are NOT in any repo file.

## Supporting material (methods chapter fodder)
- Era 04: run4 dataset generation (taka+mead READMEs), intdim362 +
  clip90 selection, outlier-fraction figures.
- Era 05: w0wa run1 datasets (READMEs in repo docs/), ΛCDM-vs-w0wa
  scratch difficulty comparison figure.
- Eras 00–01: xi + CMB emulator eras (background/history only).
- Dataset docs already in repo: `docs/datasets/README_lcdm_taka_run4.md`,
  `docs/datasets/README_w0wa_taka_run1.md`.

## Open items (updated 2026-07-18)
1. ~~Harvest colmask~~ ✓ done (incl. rescaled + θ-corr, see Axis 3).
2. Class paper PDF → verify Axis 1 numbers. (unchanged)
3. taka→mead: canon RULED (era 06); still harvest the exact per-N canon
   table from the era-06 chains dirs and reconcile test-set cuts.
4. Decide 5.2 in/out — with Vivian. (unchanged)
5. Await + harvest: 48727 (rescaled-base control — GATES the 5.4 headline
   sentence), 48728 (seed rep), Experiment A pair. Placeholder-macro
   numbers for Ch5: 4+4 medians/fractions from 48727/28, 12 metrics from
   Exp A, exact blind-corr ceiling value.
6. Optional loophole-closer: θ-corr r≈256 on the w0wa axis (30-second
   script variant) before claiming the extension law is capacity-proof.

## Editorial: five most defense-valuable results (Claude, 2026-07-18)
1. **The rescaled warm start + colmask pair (5.4).** [REVISED 2026-07-20
   after the control landed] Headline = exact-scaled initialization
   (~2.6× over naive padding, any strategy, pipeline-level fix); colmask
   keeps a small deterministic edge (6–10%, growing with N) plus the
   byte-preserved-base guarantee. Presented together they are stronger
   than either alone: a mechanism story with its own control experiment.
2. **Padded full-FT vs scratch (5.4).** The primal result: 2.5–4× data
   savings, simplest to state, hardest to attack — survives even if the
   champion's framing shifts.
3. **The extension law (5.5).** Three architecturally independent
   frozen-base methods fail on parameter-space extension while every
   base-adapting method succeeds — a qualitative, control-backed claim
   that organizes the whole cross-axis discussion.
4. **Freeze-depth crossover (5.4).** Parameter-matched pairs, reproduced
   at two freeze depths, with an N-dependent flip — the kind of clean
   ablation design committees reward, independent of any headline.
5. **Blind ceiling broken by θ-conditioning (5.3→5.5).** Hypothesis
   stated in advance (θ-blindness), tested with one variable changed,
   ceiling falls 0.9→0.67 — textbook falsifiable-prediction structure.
   (Runner-up: the kmax diagnostic — small, airtight, feeds Ch3.)

Cut as junk (one sentence each):
- **Feb 16-strategy sweep numbers**: pre-run4 data with documented
  sampling flaws — superseded, appendix at most.
- **Unrescaled late_4+mask (~7e4) and the first colmask run's large-N
  deficits**: normalization-confound artifacts; keep only as the
  one-paragraph methods anecdote about the 5× bake-in.
- **June-8 r55 = 3.9 as a quotable result**: unconverged e100 on a
  mismatched test set; the e1500 ceiling replaces it.
- **emulator_film.py / FiLM**: never run — delete from the narrative.
- **Infrastructure sweeps (lr_ic, intdim selection, gpu-pin, concurrency,
  batch-size comparisons)**: methods-appendix material, not results.
