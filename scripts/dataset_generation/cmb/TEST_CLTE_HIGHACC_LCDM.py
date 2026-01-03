#!/usr/bin/env python3
"""
Validation Script for TE CMB Training Data
===========================================
This script validates the TE data by:
1. Loading random samples from the training data
2. Verifying parameter-data correspondence and integrity
3. Re-running CAMB with exact same parameters
4. Computing Delta chi^2 to verify data integrity

Expected result: Delta chi^2 ~ 0 for all samples

Usage:
    python TEST_CLTE_HIGHACC_LCDM.py              # Both lightweight + CAMB (default)
    python TEST_CLTE_HIGHACC_LCDM.py --quick      # Lightweight only
    python TEST_CLTE_HIGHACC_LCDM.py --camb-only  # CAMB validation only
    python TEST_CLTE_HIGHACC_LCDM.py --n-samples 20 --seed 456
"""
import numpy as np
import sys
import argparse
from cobaya.yaml import yaml_load
from cobaya.model import get_model

# CAMB configuration (same as generation)
yaml_string = r"""
stop_at_error: false
likelihood:
  planck_2018_lensing.clik:
    path: /gpfs/projects/MirandaGroup/yijie/cocoa/Cocoa/external_modules/
    clik_file: plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing
params:
  omegabh2:
    prior:
      min: 0.0
      max: 0.4
    ref:
      dist: norm
      loc: 0.02239
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.0
      max: 0.5
    ref:
      dist: norm
      loc: 0.1178
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  H0:
    prior:
      min: 20
      max: 120
    ref:
      dist: norm
      loc: 67.5
      scale: 2
    proposal: 0.001
    latex: H_0
  tau:
    prior:
      min: 0.01
      max: 0.2
    ref:
      dist: norm
      loc: 0.06
      scale: 0.006
    proposal: 0.003
    latex: \tau_\mathrm{reio}
  ns:
    prior:
      min: 0.6
      max: 1.3
    ref:
      dist: norm
      loc: 0.965
      scale: 0.005
    proposal: 0.005
    latex: n_\mathrm{s}
  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.064
      scale: 0.05
    proposal: 3
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  mnu:
    prior:
      min: -10
      max: 5
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: m_{\nu}
  w:
    prior:
      min: -10
      max: 5
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}
  wa:
    prior:
      min: -5 
      max: 5
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_a
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  A_planck:
    value: 1
  thetastar:
    derived: true
    latex: \Theta_\star
  rdrag:
    derived: True
    latex: r_\mathrm{drag}
theory:
  camb:
    stop_at_error: false
    path: /gpfs/projects/MirandaGroup/yijie/cocoa/Cocoa/external_modules/code/CAMB
    extra_args:
      halofit_version: mead2020
      dark_energy_model: ppf
      lmax: 7500
      AccuracyBoost: 1.5
      kmax: 10
      k_per_logint: 130
      lens_margin: 2050
      lens_potential_accuracy: 8
      lens_k_eta_reference: 18000.0
      nonlinear: NonLinear_both
      recombination_model: CosmoRec
      Accuracy.AccurateBB: True
      min_l_logl_sampling: 6000
      DoLateRadTruncation: False
      lSampleBoost: 10
      lAccuracyBoost: 3
"""

def validate_data():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Validate CMB TT training data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation (lightweight checks only, no CAMB)')
    parser.add_argument('--camb-only', action='store_true',
                        help='CAMB validation only (skip lightweight checks)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for sample selection (default: 123)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to test (default: 10)')
    args = parser.parse_args()
    
    # Determine what to run
    run_lightweight = not args.camb_only
    run_camb = not args.quick
    
    print("="*70)
    print("CLTE Training Data Validation")
    print("="*70)
    if args.quick:
        print("Mode: Lightweight checks only")
    elif args.camb_only:
        print("Mode: CAMB validation only")
    else:
        print("Mode: Full validation (lightweight + CAMB)")
    
    # Load training data
    print("\nLoading training data...")
    data_file = 'CLTE_HIGHACC_LCDM.npy'
    param_file = 'COSMO_CLTE_HIGHACC_LCDM.dat'
    
    try:
        data = np.load(data_file, mmap_mode='r')
        params = np.loadtxt(param_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running this script in the CMB_HIGHACC_LCDM directory")
        sys.exit(1)
    
    print(f"  Data shape: {data.shape}")
    print(f"  Params shape: {params.shape}")
    
    # Select random samples
    np.random.seed(args.seed)
    indices = np.random.choice(len(data), size=args.n_samples, replace=False)
    print(f"\nTesting {args.n_samples} random samples (seed={args.seed}): {indices}")
    
    # =========================================================================
    # LIGHTWEIGHT VALIDATION
    # =========================================================================
    if run_lightweight:
        print("\n" + "="*70)
        print("LIGHTWEIGHT INTEGRITY CHECKS")
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
        
        TT_EXPECTED_RANGE = (-150, 150)  # TE can be negative!
        
        # Check shapes match
        print("\n1. Verifying data consistency...")
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
        print(f"\n2. Testing {args.n_samples} random samples...")
        
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
            
            # Check TE data range
            if not (TT_EXPECTED_RANGE[0] <= sample_data.min() and sample_data.max() <= TT_EXPECTED_RANGE[1]):
                print(f"   ✗ Sample {i+1} (idx {idx}): TE values outside expected range")
                all_passed = False
        
        if all_passed:
            print(f"   ✓ All {args.n_samples} samples passed integrity checks")
        
        # Summary statistics
        print("\n3. Dataset statistics...")
        print(f"   TE range: [{data.min():.2e}, {data.max():.2e}] uK^2")
        print(f"   Parameter ranges:")
        param_names = list(PARAM_RANGES.keys())
        for j, name in enumerate(param_names):
            print(f"      {name:20s}: [{params[:, j].min():.6f}, {params[:, j].max():.6f}]")
        
        # Lightweight verdict
        if all_passed and data.shape[0] == params.shape[0]:
            print("\n✓ LIGHTWEIGHT VALIDATION PASSED")
            print("  - All files load correctly")
            print("  - Parameter-data correspondence is correct")
            print("  - No data corruption detected")
            print("  - Values within expected physical ranges")
        else:
            print("\n✗ LIGHTWEIGHT VALIDATION FAILED")
            print("  Check errors above")
            if not run_camb:
                sys.exit(1)
    
    # =========================================================================
    # CAMB VALIDATION
    # =========================================================================
    if run_camb:
        print("\n" + "="*70)
        print("FULL CAMB VALIDATION")
        print("="*70)
        
        camb_ell_min = 2
        camb_ell_max = 5000
        
        # Test each sample
        print("\nValidating samples...")
        print("-"*70)
        
        chi2_values = []
        
        for i, idx in enumerate(indices):
            # Get stored data
            stored_te = data[idx]
            stored_params = params[idx]
            
            # Convert parameters to CAMB input format # AND (NEW) INITIALIZE CAMB MODEL EACH TIME WITHIN LOOP
            # Params are: [Omega_b h^2, Omega_c h^2, H0, tau, n_s, log(10^10 A_s), m_nu, w_0, w_a]
            print("\nInitializing CAMB model...")
            model = get_model(yaml_load(yaml_string))
            # print(f"    Before to_input: {stored_params}")
            input_params = model.parameterization.to_input(stored_params)
            # print(f"    After to_input: {input_params}")
            ### Remove fixed parameters - let YAML handle them
            input_params.pop("As", None)
            # input_params.pop("mnu", None)
            # input_params.pop("w", None) 
            # input_params.pop("wa", None)
            # print(f"\n  Parameters being used:")
            # print(f"    Stored params: {stored_params}")
            # print(f"    Input to CAMB: {input_params}")
            
            # Run CAMB
            try:
                model.logposterior(input_params)  			# Comment out to bypass Prior Check
                theory = list(model.theory.values())[1]
                # theory.calculate(input_params, cached=False)   	# uncomment to BYPASS PRIOR CHECK
                cmb = theory.get_Cl()
                regenerated_te = cmb["te"][camb_ell_min:camb_ell_max]
                
                print(f"\n  Debug for sample {i+1}:")
                print(f"    Stored TE first 5 values: {stored_te[:5]}")
                print(f"    Regenerated TE first 5 values: {regenerated_te[:5]}")
                print(f"    Stored TE shape: {stored_te.shape}")
                print(f"    Regenerated TE shape: {regenerated_te.shape}")
                print(f"    Stored TE range: [{stored_te.min():.2e}, {stored_te.max():.2e}]")
                print(f"    Regenerated TE range: [{regenerated_te.min():.2e}, {regenerated_te.max():.2e}]")
                            
                # Compute chi^2
                diff = stored_te - regenerated_te
                chi2 = np.sum(diff**2)
                chi2_values.append(chi2)
                
                status = "PASS" if chi2 < 1e-6 else "FAIL"
                print(f"Sample {i+1:2d} (index {idx:6d}): Delta chi^2 = {chi2:.2e} ... {status}")
                
            except Exception as e:
                print(f"Sample {i+1:2d} (index {idx:6d}): FAILED - {e}")
                chi2_values.append(np.nan)
        
        # Summary
        print("-"*70)
        print("\nValidation Summary:")
        valid_chi2 = [c for c in chi2_values if not np.isnan(c)]
        
        if len(valid_chi2) > 0:
            print(f"  Successful validations: {len(valid_chi2)}/{args.n_samples}")
            print(f"  Mean Delta chi^2: {np.mean(valid_chi2):.2e}")
            print(f"  Max Delta chi^2: {np.max(valid_chi2):.2e}")
            
            if np.max(valid_chi2) <= 2e-6:
                print("\n✓ VALIDATION PASSED - Data integrity confirmed!")
            else:
                print("\n✗ VALIDATION FAILED - Delta chi^2 too large!")
                sys.exit(1)
        else:
            print("\n✗ VALIDATION FAILED - No successful CAMB runs!")
            sys.exit(1)
    
    print("="*70)

if __name__ == "__main__":
    validate_data()
