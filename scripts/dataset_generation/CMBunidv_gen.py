"""
CMB Dataset Generator with On-The-Fly Parameter Generation

Generates ΛCDM training data for neural network emulators:
- Parameters: Multivariate normal sampling with hard priors
- Datavectors: CMB power spectra via CAMB with CosmoRec
- Dummy nodes: mnu=0.06, w0=-1, wa=0 (constant for transfer learning)

Usage:
    mpirun -n 20 python CMBunidv_gen.py -f <file_index>
    
Outputs:
    ./input/_{n}.npy      - Parameters (2000, 9)
    ./output/_{n}_cmb.npy - CMB spectra (2000, 4998, 4)
    
Requirements:
    - covtrainT0.npy
    - 20 MPI ranks recommended
    - ~1-2 hours per file with CosmoRec
"""

import numpy as np
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import platform
import yaml
from mpi4py import MPI
from scipy.stats import qmc
import copy
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
from schwimmbad import MPIPool

if "-f" in sys.argv:
    idx = sys.argv.index('-f')
n = int(sys.argv[idx+1])

yaml_string=r"""
stop_at_error: false
likelihood:
  planck_2018_lensing.clik:
    path: /gpfs/projects/MirandaGroup/yijie/cocoa/Cocoa/external_modules/
    #clik_file: plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2.clik_lensing
    clik_file: plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing
params:
  
  omegabh2:
    prior:
      min: 0.0
      max: 0.4
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.0
      max: 0.5
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  H0:
    prior:
      min: 20
      max: 120
    ref:
      dist: norm
      loc: 67
      scale: 2
    proposal: 0.001
    latex: H_0
  tau:
    prior:
      min: 0.01
      max: 0.2
    ref:
      dist: norm
      loc: 0.055
      scale: 0.006
    proposal: 0.003
    latex: \tau_\mathrm{reio}
  ns:
    prior:
      min: 0.6
      max: 1.3
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.005
    proposal: 0.005
    latex: n_\mathrm{s}
  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.0448
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



output: ./projects/example/chains/EXAMPLE_EVALUATE0

"""


#===================================================================================================
# Parameter generation function

def generate_parameters(N, cov_file, seed=None):
    """
    Generate cosmological parameters using multivariate normal + hard priors.
    
    Args:
        N: Number of samples
        cov_file: Path to 6x6 covariance matrix (.npy)
        seed: Random seed (default: None)
        
    Returns:
        (N, 9) array: [ωb h², ωc h², H0, τ, ns, logAs, mnu, w0, wa]
                      First 6 vary, last 3 are dummy nodes (constant)
    
    Method:
        1. Sample from N(μ, T*Σ) where T=256 (T=128 for validation and testing sets)
        2. Clip to hard priors (see Table I ranges)
        3. Append dummy nodes: mnu=0.06, w0=-1, wa=0
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Mean values for 6 LCDM parameters: [ωb h², ωc h², H0, τ, ns, logAs]
    mean = np.array([0.02239, 0.1178, 67.5, 0.06, 0.965, 3.064])
    
    # Load 6x6 covariance matrix
    cov = np.load(cov_file, allow_pickle=True)
    
    # Temperature scaling (controls sampling width) 
    T = 256                                       # (T=128 for validation and testing sets)
    
    # Generate samples from multivariate normal
    d = np.random.multivariate_normal(mean, cov*T, N)
    
    # Apply hard priors - scale for clipping
    d_scaled = d.copy()
    d_scaled[:, 0] *= 100  # ωb h² → 100×ωb h²
    d_scaled[:, 1] *= 10   # ωc h² → 10×ωc h²
    d_scaled[:, 3] *= 10   # τ → 10×τ
    
    # Clip to physical ranges
    d_scaled[:, 0] = np.clip(d_scaled[:, 0], 0, 4)        # 100×ωb h²
    d_scaled[:, 1] = np.clip(d_scaled[:, 1], 0, 3)        # 10×ωc h²
    d_scaled[:, 2] = np.clip(d_scaled[:, 2], 25, 114)     # H0 [km/s/Mpc]
    d_scaled[:, 3] = np.clip(d_scaled[:, 3], 0.07, 1.5)   # 10×τ
    d_scaled[:, 4] = np.clip(d_scaled[:, 4], 0.7, 1.3)    # ns
    d_scaled[:, 5] = np.clip(d_scaled[:, 5], 1.61, 4.5)   # log(10^10 As)
    
    # Unscale back to original units
    d[:, 0] = d_scaled[:, 0] / 100
    d[:, 1] = d_scaled[:, 1] / 10
    d[:, 2] = d_scaled[:, 2]
    d[:, 3] = d_scaled[:, 3] / 10
    d[:, 4] = d_scaled[:, 4]
    d[:, 5] = d_scaled[:, 5]
    
    # Add dummy nodes for transfer learning (constant values)
    dummy = np.ones((N, 3))
    dummy[:, 0] = 0.06   # mnu [eV]
    dummy[:, 1] = -1.0   # w0 (ΛCDM)
    dummy[:, 2] = 0.0    # wa (no evolution)
    
    # Final: (N, 9) = [6 varying LCDM params + 3 dummy nodes]
    samples = np.concatenate([d, dummy], axis=1)
    
    return samples


#===================================================================================================
# Main execution

if __name__ == '__main__':
    model = get_model(yaml_load(yaml_string))
    
    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)
    camb_ell_max = 5000
    camb_ell_min = 2
    
    # Configuration
    N_SAMPLES = 2000                      # Samples per file
    COV_FILE = './input/covtrainT0.npy'   # Double-check filepath
    
    # Output paths
    datavectors_file_path = '_'+str(n)
    CMB_DIR = './output/' + datavectors_file_path + '_cmb.npy'        # Double-check filepath
    PARAMS_DIR = './input/' + datavectors_file_path + '.npy'          # Double-check filepath
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    
    print(f'rank {rank} is at barrier')
    
    camb_ell_range = camb_ell_max - camb_ell_min
    camb_num_spectra = 4
    
    if rank == 0:
        print(f'Generating {N_SAMPLES} parameters on the fly for file {n}...')
        
        # Generate parameters (deterministic seed per file)
        seed = n * 12345
        samples = generate_parameters(N_SAMPLES, COV_FILE, seed=seed)
        
        # Save parameters for record keeping
        os.makedirs('./input', exist_ok=True)      # Double-check filepath
        np.save(PARAMS_DIR, samples)
        print(f'Saved parameters to {PARAMS_DIR}')
        print(f'Parameter shape: {samples.shape}')
        print(f'First sample: {samples[0]}')
        
        total_num_dvs = len(samples)
        
        # Distribute parameters across MPI ranks
        param_info = samples[0:total_num_dvs:num_ranks]  # rank 0's subset
        for i in range(1, num_ranks):
            comm.send(samples[i:total_num_dvs:num_ranks], dest=i, tag=1)
    else:
        param_info = comm.recv(source=0, tag=1)
    
    # Each rank computes its assigned datavectors
    num_datavector = len(param_info)
    total_cls = np.zeros(
        (num_datavector, camb_ell_range, camb_num_spectra), dtype="float32"
    )
    
    for i in range(num_datavector):
        input_params = model.parameterization.to_input(param_info[i])
        input_params.pop("As", None)
        
        try:
            model.logposterior(input_params)
            theory = list(model.theory.values())[1]
            cmb = theory.get_Cl()
            
        except Exception as e:
            print(f'rank {rank}, sample {i} failed: {e}')
        else:
            # Extract CMB power spectra [TT, TE, EE, PP]
            total_cls[i, :, 0] = cmb["tt"][camb_ell_min:camb_ell_max]
            total_cls[i, :, 1] = cmb["te"][camb_ell_min:camb_ell_max]
            total_cls[i, :, 2] = cmb["ee"][camb_ell_min:camb_ell_max]
            total_cls[i, :, 3] = cmb["pp"][camb_ell_min:camb_ell_max]
    
    # Gather results at rank 0
    if rank == 0:
        result_cls = np.zeros((total_num_dvs, camb_ell_range, 4), dtype="float32")
        result_cls[0:total_num_dvs:num_ranks] = total_cls
        
        for i in range(1, num_ranks):
            result_cls[i:total_num_dvs:num_ranks, :, 0] = comm.recv(source=i, tag=10)
            result_cls[i:total_num_dvs:num_ranks, :, 1] = comm.recv(source=i, tag=11)
            result_cls[i:total_num_dvs:num_ranks, :, 2] = comm.recv(source=i, tag=12)
            result_cls[i:total_num_dvs:num_ranks, :, 3] = comm.recv(source=i, tag=13)
        
        # Save final datavectors
        os.makedirs('./output', exist_ok=True) 
        np.save(CMB_DIR, result_cls)
        print(f'Saved CMB datavectors to {CMB_DIR}')
        print(f'Datavector shape: {result_cls.shape}')
        
    else:
        # Send results back to rank 0
        comm.send(total_cls[:, :, 0], dest=0, tag=10)
        comm.send(total_cls[:, :, 1], dest=0, tag=11)
        comm.send(total_cls[:, :, 2], dest=0, tag=12)
        comm.send(total_cls[:, :, 3], dest=0, tag=13)
    
    if rank == 0:
        print(f'File {n} complete!')
