# Dataset: ΛCDM Takahashi — Run 4

## Overview
Roman Space Telescope cosmic shear data vectors generated under ΛCDM cosmology
using the Takahashi (2012) halofit prescription.

Run 4 is the canonical corrected dataset. Prior runs used consecutive MCMC
chain sampling which produced highly autocorrelated samples with poor prior
coverage, especially at tails. Run 4 fixes this with random index selection
and a 5M step floor.

---

## Generation

| Setting          | Value |
|------------------|-------|
| Cluster          | SeaWulf (milan nodes, `hbm-long-96core` partition) |
| MPI tasks        | 192 (2 nodes × 96 tasks/node) |
| Cocoa YAML       | `lcdm_takahashi_cs_CNN.yaml` |
| Halofit          | Takahashi 2012 (`takahashi`) |
| Cosmology        | ΛCDM |
| Probe            | Full 3×2pt (2115 elements generated); training uses indices 0:1080 |
| Data file        | `ones.dataset` — no masking, all DV elements included |
| Sampling method  | Tempered MCMC, random index selection (`--unif 0`) |
| Max correlation  | 0.15 (`--maxcorr 0.15`) |
| Prior boundary   | 1.0 (`--boundary 1.0`) |
| Checkpoint freq  | Every 2000 samples |

### Training set
| Setting         | Value |
|-----------------|-------|
| Temperature     | T=500 |
| Size            | 6,000,000 samples |
| Min chain steps | 5,000,000 |

### Validation / Test sets
| Setting         | Value |
|-----------------|-------|
| Val temperature | T=250 |
| Test temperatures | T=250 and T=125 |
| Size (each)     | 50,000 samples |
| Prior cut       | Post-hoc via `clip_prior.py`; 2.5%, 5%, 10% variants available |

---

## Parameters (15 total)

| Column | Parameter     | Prior type | Prior range / σ |
|--------|---------------|------------|-----------------|
| 0  | `As_1e9`      | Uniform  | [0.5, 5.0] |
| 1  | `ns`          | Uniform  | [0.87, 1.07] |
| 2  | `H0`          | Uniform  | [55.0, 91.0] |
| 3  | `omegab`      | Uniform  | [0.03, 0.07] |
| 4  | `omegam`      | Uniform  | [0.1, 0.9] |
| 5  | `roman_DZ_S1` | Gaussian | loc=0, σ=0.002 |
| 6  | `roman_DZ_S2` | Gaussian | loc=0, σ=0.002 |
| 7  | `roman_DZ_S3` | Gaussian | loc=0, σ=0.002 |
| 8  | `roman_DZ_S4` | Gaussian | loc=0, σ=0.002 |
| 9  | `roman_DZ_S5` | Gaussian | loc=0, σ=0.002 |
| 10 | `roman_DZ_S6` | Gaussian | loc=0, σ=0.002 |
| 11 | `roman_DZ_S7` | Gaussian | loc=0, σ=0.002 |
| 12 | `roman_DZ_S8` | Gaussian | loc=0, σ=0.002 |
| 13 | `roman_A1_1`  | Uniform  | [-5.0, 5.0] |
| 14 | `roman_A1_2`  | Uniform  | [-5.0, 5.0] |

Fixed (not sampled): `mnu=0.06`, `tau=0.06`, `w0pwa=-1`, `w=-1`,
all `roman_M*=0`, `roman_A2_*=0`, `roman_BTA_1=0`.

---

## Files

```
roman_real_lcdm_taka_train_T500_run4_cs_500.npy
roman_real_lcdm_b_taka_train_parameters_T500_run4.txt

roman_real_lcdm_taka_valid_T250_run4_cs_250.npy
roman_real_lcdm_taka_valid_T250_run4_cs_250_cut2p5pct.npy
roman_real_lcdm_taka_valid_T250_run4_cs_250_cut5pct.npy
roman_real_lcdm_taka_valid_T250_run4_cs_250_cut10pct.npy
roman_real_lcdm_b_taka_valid_parameters_T250_run4.txt
roman_real_lcdm_b_taka_valid_parameters_T250_run4_cut2p5pct.txt
roman_real_lcdm_b_taka_valid_parameters_T250_run4_cut5pct.txt
roman_real_lcdm_b_taka_valid_parameters_T250_run4_cut10pct.txt

roman_real_lcdm_taka_test_T250_run4_cs_250.npy
roman_real_lcdm_taka_test_T250_run4_cs_250_cut2p5pct.npy
roman_real_lcdm_taka_test_T250_run4_cs_250_cut5pct.npy
roman_real_lcdm_taka_test_T250_run4_cs_250_cut10pct.npy
roman_real_lcdm_b_taka_test_parameters_T250_run4.txt
roman_real_lcdm_b_taka_test_parameters_T250_run4_cut2p5pct.txt
roman_real_lcdm_b_taka_test_parameters_T250_run4_cut5pct.txt
roman_real_lcdm_b_taka_test_parameters_T250_run4_cut10pct.txt

roman_real_lcdm_taka_test_T125_run4_cs_125.npy
roman_real_lcdm_taka_test_T125_run4_cs_125_cut2p5pct.npy
roman_real_lcdm_taka_test_T125_run4_cs_125_cut5pct.npy
roman_real_lcdm_taka_test_T125_run4_cs_125_cut10pct.npy
roman_real_lcdm_b_taka_test_parameters_T125_run4.txt
roman_real_lcdm_b_taka_test_parameters_T125_run4_cut2p5pct.txt
roman_real_lcdm_b_taka_test_parameters_T125_run4_cut5pct.txt
roman_real_lcdm_b_taka_test_parameters_T125_run4_cut10pct.txt
```

