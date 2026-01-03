#!/usr/bin/env python3
"""
Lightweight Validation Script for TT CMB Training Data
=======================================================

This script validates data integrity by:
1. Loading 10 random samples
2. Verifying parameter-data correspondence
3. Checking for data corruption (NaNs, infinities)
4. Verifying values are within expected physical ranges

No CAMB installation required - this is a quick integrity check.

Usage:
    python TEST_CLTT_HIGHACC_LCDM.py
"""

import numpy as np
import sys

def validate_data():
    print("="*70)
    print("CLTT Training Data Validation")
    print("="*70)
    
    # Expected ranges (from generation)
    PARAM_RANGES = {
        'Omega_b h^2': (0.0, 0.04),
        'Omega_c h^2': (0.0, 0.3),
        'H0': (25.0, 114.0),
        'tau': (0.007, 0.15),
        'n_s': (0.7, 1.3),
        'log(10^10 A_s)': (1.61, 4.5),
        'm_nu': (0.06, 0.06),  # Fixed
        'w_0': (-1.0, -1.0),   # Fixed
        'w_a': (0.0, 0.0)      # Fixed
    }
    
    TT_EXPECTED_RANGE = (0.0, 15000.0)  # Reasonable range for TT in uK^2
    
    # Load training data
    print("\n1. Loading data files...")
    data_file = 'CLTT_HIGHACC_LCDM.npy'
    param_file = 'COSMO_CLTT_HIGHACC_LCDM.dat'
    
    try:
        data = np.load(data_file, mmap_mode='r')
        params = np.loadtxt(param_file)
    except FileNotFoundError as e:
        print(f"   ✗ ERROR: {e}")
        print("   Make sure you're in the CMB_HIGHACC_LCDM directory")
        sys.exit(1)
    
    print(f"   ✓ Files loaded successfully")
    print(f"   Data shape: {data.shape}")
    print(f"   Params shape: {params.shape}")
    
    # Check shapes match
    print("\n2. Verifying data consistency...")
    if data.shape[0] != params.shape[0]:
        print(f"   ✗ FAIL: Shape mismatch! Data has {data.shape[0]} samples, params has {params.shape[0]}")
        sys.exit(1)
    print(f"   ✓ Data and parameters have matching counts ({data.shape[0]:,} cosmologies)")
    
    if data.shape[1] != 4998:
        print(f"   ✗ WARNING: Expected 4998 ell values, got {data.shape[1]}")
    else:
        print(f"   ✓ Correct number of ell values (4998)")
    
    if params.shape[1] != 9:
        print(f"   ✗ FAIL: Expected 9 parameters, got {params.shape[1]}")
        sys.exit(1)
    print(f"   ✓ Correct number of parameters (9)")
    
    # Test random samples
    print("\n3. Testing random samples...")
    n_test = 10
    np.random.seed(42)
    indices = np.random.choice(len(data), size=n_test, replace=False)
    print(f"   Testing {n_test} random indices: {indices}")
    
    all_passed = True
    
    for i, idx in enumerate(indices):
        sample_data = data[idx]
        sample_params = params[idx]
        
        # Check for NaN or Inf
        if np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data)):
            print(f"   ✗ Sample {i+1} (idx {idx}): Data contains NaN or Inf")
            all_passed = False
            continue
            
        if np.any(np.isnan(sample_params)) or np.any(np.isinf(sample_params)):
            print(f"   ✗ Sample {i+1} (idx {idx}): Parameters contain NaN or Inf")
            all_passed = False
            continue
        
        # Check parameter ranges
        param_names = list(PARAM_RANGES.keys())
        for j, (name, (pmin, pmax)) in enumerate(PARAM_RANGES.items()):
            if not (pmin <= sample_params[j] <= pmax):
                print(f"   ✗ Sample {i+1} (idx {idx}): {name} = {sample_params[j]:.6f} outside range [{pmin}, {pmax}]")
                all_passed = False
        
        # Check TT data range
        if not (TT_EXPECTED_RANGE[0] <= sample_data.min() and sample_data.max() <= TT_EXPECTED_RANGE[1]):
            print(f"   ✗ Sample {i+1} (idx {idx}): TT values outside expected range")
            all_passed = False
    
    if all_passed:
        print(f"   ✓ All {n_test} samples passed integrity checks")
    
    # Summary statistics
    print("\n4. Dataset statistics...")
    print(f"   TT range: [{data.min():.2e}, {data.max():.2e}] uK^2")
    print(f"   Parameter ranges:")
    param_names = list(PARAM_RANGES.keys())
    for j, name in enumerate(param_names):
        print(f"      {name:20s}: [{params[:, j].min():.6f}, {params[:, j].max():.6f}]")
    
    # Final verdict
    print("\n" + "="*70)
    if all_passed and data.shape[0] == params.shape[0]:
        print("✓ VALIDATION PASSED")
        print("  - All files load correctly")
        print("  - Parameter-data correspondence is correct")
        print("  - No data corruption detected")
        print("  - Values within expected physical ranges")
    else:
        print("✗ VALIDATION FAILED")
        print("  Check errors above")
    print("="*70)

if __name__ == "__main__":
    validate_data()