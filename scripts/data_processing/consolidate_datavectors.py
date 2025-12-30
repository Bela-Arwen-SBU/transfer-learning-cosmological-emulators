import numpy as np
import os

"""
Consolidate CMB Data Vectors (Memory-Mapped Version)
=====================================================
Combines multiple batch files of CMB power spectra into a single consolidated array.
Uses memory mapping to write directly to disk without loading full array into RAM.

Input: Directory containing *_cmb.npy files with shape (n_cosmologies, n_ell, 4)
       where columns are [TT, TE, EE, ℓ]
       
Output: Single .npy file with all cosmologies stacked

Memory usage: Only loads one file at a time (~160 MB)
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
input_dir = "/gpfs/projects/MirandaGroup/bela/cocoa/Cocoa/LCDM/output/base_truth"
output_dir = "/gpfs/projects/MirandaGroup/bela/cocoa/Cocoa/LCDM/output/consolidated"
output_filename = "all_cosmologies.npy"

# =============================================================================
# MAIN
# =============================================================================

os.makedirs(output_dir, exist_ok=True)

# Get all batch files
batch_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_cmb.npy')])
n_files = len(batch_files)
print(f"Found {n_files} files in {input_dir}")

# Load first file to get dimensions
first = np.load(os.path.join(input_dir, batch_files[0]))
n_cosmo_per_file, n_ell, n_cols = first.shape
print(f"Each file shape: ({n_cosmo_per_file}, {n_ell}, {n_cols})")
print(f"Columns: [TT, TE, EE, ℓ]")

# Calculate total size
total_cosmo = n_files * n_cosmo_per_file
print(f"\nTotal cosmologies to consolidate: {total_cosmo:,}")
print(f"Output array shape: ({total_cosmo}, {n_ell}, {n_cols})")
estimated_gb = (total_cosmo * n_ell * n_cols * 4) / 1e9
print(f"Estimated size: {estimated_gb:.2f} GB")

# Create memory-mapped array (writes directly to disk)
output_path = os.path.join(output_dir, output_filename)
print(f"\nCreating memory-mapped file: {output_path}")
all_data = np.memmap(output_path, dtype='float32', mode='w+', 
                     shape=(total_cosmo, n_ell, n_cols))

# Process files one at a time
print(f"\nProcessing {n_files} files (one at a time, writing directly to disk)...")
for i, fname in enumerate(batch_files):
    if i % 50 == 0 or i == n_files - 1:
        print(f"  File {i+1}/{n_files}: {fname}")
    
    # Load batch
    batch = np.load(os.path.join(input_dir, fname))
    
    # Write directly to memory-mapped file
    start_idx = i * n_cosmo_per_file
    end_idx = start_idx + n_cosmo_per_file
    all_data[start_idx:end_idx] = batch
    
    # Flush to disk every 50 files
    if i % 50 == 0:
        all_data.flush()

# Final flush
all_data.flush()

file_size_gb = os.path.getsize(output_path) / 1e9
print(f"\nSaved successfully!")
print(f"Final file size: {file_size_gb:.2f} GB")
print(f"Shape: {all_data.shape}")

print("\nDone!")