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
n= int(sys.argv[idx+1])

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


####add main function.


#===================================================================================================
# datavectors

# def generate_parameters(N, u_bound, l_bound, mode, parameters_file, save=True):
#     D = len(u_bound)
#     if mode=='train':
        
#         N_LHS = int(0.05*N)
#         sampler = qmc.LatinHypercube(d=D)
#         sample = sampler.random(n=N_LHS)
#         sample_scaled = qmc.scale(sample, l_bound, u_bound)

#         N_uni = N-N_LHS
#         data = np.random.uniform(low=l_bound, high=u_bound, size=(N_uni, D))
#         samples = np.concatenate((sample_scaled, data), axis=0)
#     else:
#         samples = np.random.uniform(low=l_bound, high=u_bound, size=(N, D))

#     if save:
#         np.save(parameters_file, samples)
#         print('(Input Parameters) Saved!')
#     return samples



if __name__ == '__main__':
    model = get_model(yaml_load(yaml_string))
    

    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)
    camb_ell_max = 5000
    camb_ell_min = 2
    datavectors_file_path = '_'+str(n)
    parameters_file  = './input/_'+str(n)+'.npy'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')
        
    camb_ell_range = camb_ell_max-camb_ell_min
    camb_num_spectra = 4
    CMB_DIR = './output/' + datavectors_file_path + '_cmb.npy'
    #EXTRA_DIR = datavectors_file_path + '_extra.npy'

    if rank == 0:
        samples = np.load(parameters_file,allow_pickle=True)#[:200]
        total_num_dvs = len(samples)

        param_info = samples[0:total_num_dvs:num_ranks]#reading for 0th rank input
        for i in range(1,num_ranks):#sending other ranks' data
            comm.send(
                samples[i:total_num_dvs:num_ranks], 
                dest = i, 
                tag  = 1
            )
                
    else:
            
        param_info = comm.recv(source = 0, tag = 1)

    num_datavector = len(param_info)
    total_cls = np.zeros(
            (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
        ) 
    #extra_dv = np.zeros(
     #       (num_datavector, 2), dtype = "float32"
      #  ) 
    for i in range(num_datavector):
        #print(param_info[i])
        input_params = model.parameterization.to_input(param_info[i])
        input_params.pop("As", None)

        try:
            model.logposterior(input_params)
            theory = list(model.theory.values())[1]
            cmb = theory.get_Cl()
            #rdrag = theory.get_param("rdrag")
            #thetastar = theory.get_param("thetastar")
                
        except:
            print('fail')
        else:
            total_cls[i,:,0] = cmb["tt"][camb_ell_min:camb_ell_max]
            total_cls[i,:,1] = cmb["te"][camb_ell_min:camb_ell_max]
            total_cls[i,:,2] = cmb["ee"][camb_ell_min:camb_ell_max]
            total_cls[i,:,3] = cmb["pp"][camb_ell_min:camb_ell_max]

            #extra_dv[i,0] = thetastar
            #extra_dv[i,1] = rdrag

    if rank == 0:
        result_cls   = np.zeros((total_num_dvs, camb_ell_range, 4), dtype="float32")
        result_extra = np.zeros((total_num_dvs, 2), dtype="float32") 
        result_cls[0:total_num_dvs:num_ranks] = total_cls ## CMB       
        #result_extra[0:total_num_dvs:num_ranks]   = extra_dv ##0: 100theta^*, 1: r_drag

        for i in range(1,num_ranks):        
            result_cls[i:total_num_dvs:num_ranks,:,0] = comm.recv(source = i, tag = 10)
            result_cls[i:total_num_dvs:num_ranks,:,1] = comm.recv(source = i, tag = 11)
            result_cls[i:total_num_dvs:num_ranks,:,2] = comm.recv(source = i, tag = 12)
            result_cls[i:total_num_dvs:num_ranks,:,3] = comm.recv(source = i, tag = 13)
            #result_extra[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 14)

        np.save(CMB_DIR, result_cls)
        #np.save(EXTRA_DIR, result_extra)
            
    else:    
        comm.send(total_cls[:,:,0], dest = 0, tag = 10)
        comm.send(total_cls[:,:,1], dest = 0, tag = 11)
        comm.send(total_cls[:,:,2], dest = 0, tag = 12)
        comm.send(total_cls[:,:,3], dest = 0, tag = 13)
        #comm.send(extra_dv, dest = 0, tag = 14)
