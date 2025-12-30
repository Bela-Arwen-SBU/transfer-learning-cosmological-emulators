import numpy as np
import os

"""
Split and Archive CMB Training Data
====================================
Splits consolidated CMB data vectors into TT, TE, EE and creates
companion cosmological parameter files with matching naming convention.

Input: 
    - output/consolidated/all_cosmologies.npy (shape: N, 4998, 4)
    - input/basetruth_params/*.npy (shape: 2000, 9 each)
    
Output (Vivian's naming convention):
    - CLTT_HIGHACC_LCDM.npy + COSMO_CLTT_HIGHACC_LCDM.dat
    - CLTE_HIGHACC_LCDM.npy + COSMO_CLTE_HIGHACC_LCDM.dat  
    - CLEE_HIGHACC_LCDM.npy + COSMO_CLEE_HIGHACC_LCDM.dat
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
data_file = "output/consolidated/all_cosmologies.npy"
params_dir = "input/basetruth_params"
output_dir = "output/archived"

SKIP_BATCH = 216  # This batch failed and is missing from data

# =============================================================================
# SPLIT DATA VECTORS
# =============================================================================
print("="*70)
print("STEP 1: Split data vectors into TT, TE, EE")
print("="*70)

os.makedirs(output_dir, exist_ok=True)

print(f"\nLoading consolidated data: {data_file}")
data = np.memmap(data_file, dtype='float32', mode='r', 
                 shape=(1028000, 4998, 4))
print(f"Shape: {data.shape}")
print(f"Columns: [TT, TE, EE, ℓ]") 

n_cosmo, n_ell, n_cols = data.shape

# Extract each spectrum
print("\nExtracting spectra...")
TT = data[:, :, 0]
TE = data[:, :, 1]  
EE = data[:, :, 2]

print(f"  TT shape: {TT.shape}")
print(f"  TE shape: {TE.shape}")
print(f"  EE shape: {EE.shape}")

# Save each spectrum
print("\nSaving spectra...")
np.save(os.path.join(output_dir, "CLTT_HIGHACC_LCDM.npy"), TT)
print(f"  Saved: CLTT_HIGHACC_LCDM.npy")

np.save(os.path.join(output_dir, "CLTE_HIGHACC_LCDM.npy"), TE)
print(f"  Saved: CLTE_HIGHACC_LCDM.npy")

np.save(os.path.join(output_dir, "CLEE_HIGHACC_LCDM.npy"), EE)
print(f"  Saved: CLEE_HIGHACC_LCDM.npy")

# =============================================================================
# CONSOLIDATE PARAMETERS
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Consolidate cosmological parameters")
print("="*70)

# Get all parameter files (sorted numerically)
param_files = sorted([f for f in os.listdir(params_dir) if f.endswith('.npy')],
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

print(f"\nFound {len(param_files)} parameter files")

# Filter out batch 216
param_files_filtered = [f for f in param_files if f != f'_{SKIP_BATCH}.npy']
n_files = len(param_files_filtered)
print(f"After skipping batch {SKIP_BATCH}: {n_files} files")

# Load first to get shape
first_params = np.load(os.path.join(params_dir, param_files_filtered[0]))
n_cosmo_per_file, n_params = first_params.shape
print(f"Each file shape: ({n_cosmo_per_file}, {n_params})")

# Check alignment
total_cosmo_params = n_files * n_cosmo_per_file
print(f"\nTotal cosmologies in params: {total_cosmo_params:,}")
print(f"Total cosmologies in data: {n_cosmo:,}")
if total_cosmo_params == n_cosmo:
    print("✓ Params and data are aligned!")
else:
    print("✗ WARNING: Params and data counts don't match!")

# Consolidate parameters using memmap
print(f"\nConsolidating parameters...")
all_params = np.zeros((total_cosmo_params, n_params), dtype=np.float32)

for i, fname in enumerate(param_files_filtered):
    if i % 50 == 0:
        print(f"  File {i+1}/{n_files}")
    
    params = np.load(os.path.join(params_dir, fname))
    start_idx = i * n_cosmo_per_file
    end_idx = start_idx + n_cosmo_per_file
    all_params[start_idx:end_idx] = params

# Save as .dat files (one for each spectrum)
print("\nSaving parameter files...")
np.savetxt(os.path.join(output_dir, "COSMO_CLTT_HIGHACC_LCDM.dat"), 
           all_params, fmt='%.6e')
print(f"  Saved: COSMO_CLTT_HIGHACC_LCDM.dat")

np.savetxt(os.path.join(output_dir, "COSMO_CLTE_HIGHACC_LCDM.dat"), 
           all_params, fmt='%.6e')
print(f"  Saved: COSMO_CLTE_HIGHACC_LCDM.dat")

np.savetxt(os.path.join(output_dir, "COSMO_CLEE_HIGHACC_LCDM.dat"), 
           all_params, fmt='%.6e')
print(f"  Saved: COSMO_CLEE_HIGHACC_LCDM.dat")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ARCHIVAL COMPLETE")
print("="*70)
print(f"\nOutput directory: {output_dir}")
print(f"\nFiles created:")
print(f"  CLTT_HIGHACC_LCDM.npy ({n_cosmo:,} × {n_ell})")
print(f"  CLTE_HIGHACC_LCDM.npy ({n_cosmo:,} × {n_ell})")
print(f"  CLEE_HIGHACC_LCDM.npy ({n_cosmo:,} × {n_ell})")
print(f"  COSMO_CLTT_HIGHACC_LCDM.dat ({total_cosmo_params:,} × {n_params})")
print(f"  COSMO_CLTE_HIGHACC_LCDM.dat ({total_cosmo_params:,} × {n_params})")
print(f"  COSMO_CLEE_HIGHACC_LCDM.dat ({total_cosmo_params:,} × {n_params})")
print(f"\nSkipped batch: {SKIP_BATCH}")
print("="*70)