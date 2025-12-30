# ΛCDM CMB Training Data Generation

This folder contains scripts and data for generating high-accuracy CMB training data for neural network emulators.

## Directory Structure

### `input/`
Cosmological parameters (Gaussian-sampled from covariance matrix).

**Historical**: Previously required pre-generation of parameter files. Now we use on-the-fly generation (see `CMBunidv_gen.py`), but parameters are still saved here for safety and reproducibility. 

### `output/`
Generated CMB power spectra data vectors.

**Structure**:
- `base_truth/`: Raw output files (_0_cmb.npy through _514_cmb.npy)
  - Each file: 2000 cosmologies × 4998 ℓ values × 4 columns [TT, TE, EE, ℓ]
  - Missing batch 216 (failed job)
- `consolidated/`: Intermediate processing files
- `archived/`: Final training-ready files (see archived/README.md)
- `bt_testing_params/`, `bt_validation_params/`: Testing/validation sets

### `batch_scripts/`
SLURM job submission scripts for SeaWulf cluster.

**Files**:
- Partition-specific scripts: `partition-name.sbatch`
- Single-file generation: `single_file.sbatch`
- On-the-fly generation: `*_gen.sbatch`

## Data Generation Pipeline

1. **Generate parameters + data vectors** (on-the-fly):
```bash
   mpirun -n 20 python CMBunidv_gen.py -f <file_index>
```
   - Outputs: `input/_{n}.npy` (params) + `output/base_truth/_{n}_cmb.npy` (data)
   - Time: 1-2 hours per batch with CosmoRec
   - Requirements: `covtrainT0.npy` covariance matrix

2. **Consolidate and archive** (after all batches complete):
```bash
   python consolidate_datavectors.py  # Combine files
   python split_and_archive.py        # Create training-ready format
```

## Key Scripts

- `CMBunidv_gen.py`: Main data generation (parameters + CAMB)
- `consolidate_datavectors.py`: Combine batch files into single array
- `split_and_archive.py`: Create final training format (see archived/)

## Dataset Summary

- **Total cosmologies**: 1,028,000 (514 batches × 2000)
- **Missing**: Batch 216 (job failure)
- **Cosmology**: ΛCDM + dummy nodes (mnu=0.06, w0=-1, wa=0)
- **Method**: CAMB with CosmoRec recombination
- **Accuracy**: High-accuracy for Stage-IV surveys