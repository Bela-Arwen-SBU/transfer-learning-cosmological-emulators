#===================================================================================================
# IMPORTS
#===================================================================================================

# === Python Standard Library ===
import os
import sys
import argparse
from datetime import datetime

# === External Packages ===
import numpy as np
import yaml
import torch
import h5py as h5

# === Local ===
from emulator import ResMLP, ResCNN, ResTRF # emulator architecture

#===================================================================================================
# COMMAND LINE ARGUMENTS
# All arguments for training configuration, transfer learning, and output control.
# Usage: python train_emulator.py --yaml <yaml> --probe <probe> [options]
# Run with --help for full list of arguments.
#===================================================================================================

parser = argparse.ArgumentParser(prog='train_emulator')

# === Standard Training Arguments  ===

parser.add_argument("--yaml", "-y",
    dest="cobaya_yaml",
    help="The training YAML containing the training_args block",
    type=str,
    nargs='?')

parser.add_argument("--probe", "-p",
    dest="probe",
    help="the probe, listed in the yaml, of which to generate data vectors for.",
    type=str,
    nargs='?')

parser.add_argument("--epochs", "-e",
    dest="n_epochs",
    help="(int) number of training epochs. Default=250",
    type=int,
    default=250,
    nargs='?')

parser.add_argument("--batchsize", "-b",
    dest="batch_size",
    help="(int) batch size to use while training. Default=256",
    type=int,
    default=256,
    nargs='?')

parser.add_argument("--learning_rate", "-lr",
    dest="learning_rate",
    help="(float) learning rate to use while training. Default=1e-3",
    type=float,
    default=1e-3,
    nargs='?')

parser.add_argument("--weight_decay", "-wd",
    dest="weight_decay",
    help="(float) Weight decay (adds L2 norm of model weights to loss fcn) to use while training. Default=0",
    type=float,
    default=0.0,
    nargs='?')

parser.add_argument("--save_losses", "-s",
    dest="save_losses",
    help="(bool) Save losses at each epoch to a text file 'losses.txt'. Default=False",
    type=bool,
    default=False,
    nargs='?')

parser.add_argument("--mask", "-m",
    dest="training_masked",
    help="Mask the training data using the mask define in the dataset file. Default=False",
    type=bool,
    default=False,
    nargs='?')

# === Additions by Béla (2/25) ===

parser.add_argument("--ntrain", "-nt",
    dest="n_train",
    help="(int) Number of training samples. Default=None (use all)",
    type=int,
    default=None,
    nargs='?')

parser.add_argument("--save_testing_metrics", "-st",
    dest="save_testing_metrics",
    help="(bool) Save testing metrics to a text file. Default=False",
    type=bool,
    default=False,
    nargs='?')

parser.add_argument("--squeeze_factor", "-sf",
    dest="squeeze_factor",
    help="(float) Divide covariance matrix by this factor (cov = cov / squeeze_factor). Default=1.0 (no squeezing)",
    type=float,
    default=1.0,
    nargs='?')

# === Transfer Learning Arguments (added by Béla (2/26))=== 

parser.add_argument("--transfer_learning", "-tl",
    dest="transfer_learning",
    help="(bool) Enable transfer learning mode. Default=False",
    type=bool,
    default=False,
    nargs='?')

parser.add_argument("--pretrained_model", "-pm",
    dest="pretrained_model",
    help="Path to pretrained model (.pth file) for transfer learning. Corresponding .h5 file must exist at same path.",
    type=str,
    default=None,
    nargs='?')

parser.add_argument("--freeze_strategy", "-fs",
    dest="freeze_strategy",
    help="Layer freezing strategy for transfer learning. See freeze_layers() for full list.",
    type=str,
    default='none',
    choices=[
        'none',
        # ResMLP + ResCNN early (from input side)
        'early_1', 'early_2', 'early_3', 'early_4',
        # ResCNN only early (CNN section)
        'early_5', 'early_6', 'early_7',
        # ResMLP + ResCNN late (from output side)
        'late_1', 'late_2', 'late_3', 'late_4',
        # ResCNN only late (CNN section)
        'late_5', 'late_6', 'late_7',
        # Both
        'input_output',
        # ResBlock-specific
        'resnet_1', 'resnet_2', 'resnet_3',
        'resnet_12', 'resnet_23', 'resnet_123',
        # ResCNN CNN-specific
        'cnn_only', 'cnn_transform', 'cnn_all',
    ],
    nargs='?')

# === Standard Arguments ===
args, unknown     = parser.parse_known_args()
cobaya_yaml       = args.cobaya_yaml
probe             = args.probe
n_epochs          = args.n_epochs
batch_size        = args.batch_size
learning_rate     = args.learning_rate
weight_decay      = args.weight_decay
save_losses       = args.save_losses
training_masked   = args.training_masked
# === Additions by Béla (2/25) ===
n_train              = args.n_train
save_testing_metrics = args.save_testing_metrics
squeeze_factor       = args.squeeze_factor
# === Transfer Learning Arguments (2/26) ===
transfer_learning = args.transfer_learning
pretrained_model  = args.pretrained_model
freeze_strategy   = args.freeze_strategy

#===================================================================================================
# COVARIANCE MATRIX LOADER
# Reads the covariance matrix from the .dataset file specified in the YAML.
# Supports masked and unmasked covariance matrices.
# Args:
#   train_yaml       (str): path to the training YAML
#   masked          (bool): if True, applies the scale cut mask defined in the dataset file
#   squeeze_factor (float): divide covariance by this factor (default=1.0, no squeezing) (added by Bela)
#===================================================================================================

def get_cov(train_yaml, masked=False, squeeze_factor=1.0):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'cov_file'):
            cov_file = split[2]

    full_cov = np.loadtxt(path+'/'+cov_file)
    cov_scenario = full_cov.shape[1]
    size = int(np.max(full_cov[:])+1)

    cov = np.zeros((size,size))

    for line in full_cov:
        i = int(line[0])
        j = int(line[1])

        if(cov_scenario == 3):
            cov_ij = line[2]
        elif(cov_scenario == 4):
            cov_ij = line[2]+line[3]
        elif(cov_scenario == 10):
            cov_ij = line[8]+line[9]

        cov[i,j] = cov_ij
        cov[j,i] = cov_ij

    # Added by Bela: apply squeeze factor to covariance matrix
    cov = cov / squeeze_factor

    cov_inv = np.linalg.inv(cov) # QUESTION: never used, safe to remove?

    if ( masked==False ):
        return cov

    else:
        mask = get_mask(train_yaml).astype(bool)
        cov_masked = cov[mask][:,mask] # QUESTION: never used, safe to remove?
        cov_inv_masked = np.zeros((size,size))
        
        for i in range(size):
            for j in range(i,size):
                if(i!=j):
                    mask_row    = mask[i]
                    mask_column = mask[j]

                    cov_inv_masked[i,j] = cov[i,j] * mask_row * mask_column
                    cov_inv_masked[j,i] = cov_inv_masked[i,j]
                else:
                    cov_inv_masked[i,j] = cov[i,j]

        cov_inv_masked = np.linalg.inv(cov_inv_masked)[mask][:,mask]

        return np.linalg.inv(cov_inv_masked)

#===================================================================================================
# SCALE CUT MASK LOADER
# Reads the binary scale cut mask from the .dataset file specified in the YAML.
# Returns a 1D array of 0s and 1s ordered by data vector index.
#===================================================================================================

def get_mask(train_yaml):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'mask_file'):
            mask_file = split[2]

    mask = np.loadtxt(path+'/'+mask_file)
    idxs = np.argsort(mask[:,0])

    return mask[:,1][idxs]

#===================================================================================================
# FIDUCIAL DATA VECTOR LOADER
# Reads the fiducial data vector from the .dataset file specified in the YAML.
# Used for centering the data vector before diagonalization.
#===================================================================================================

def get_datavector(train_yaml):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'data_file'):
            dv_file = split[2]

    datavector = np.loadtxt(path+'/'+dv_file)
    idxs = np.argsort(datavector[:,0])

    return datavector[:,1][idxs]

#===================================================================================================
# TRANSFER LEARNING: LAYER FREEZING
# Freezes model layers based on the specified freeze strategy.
# Supports ResMLP and ResCNN. ResTRF support is a TODO.
#===================================================================================================

def freeze_layers(model, freeze_strategy, transfer_learning=True):
    """
    Freeze model layers for transfer learning.

    ResMLP structure (model.model[i]):
        [0] Input Linear  (input_dim -> int_dim)
        [1] ResBlock 1
        [2] ResBlock 2
        [3] ResBlock 3
        [4] Output Linear (int_dim -> output_dim)
        [5] Affine norm   <- never freeze

    ResCNN structure (named attributes):
        input_layer   : Linear (input_dim -> int_dim)
        Res1/2/3      : ResBlocks
        Act1/2/3      : Activations after each ResBlock
        CNN_transform : Linear (int_dim -> cnn_dim)
        convs         : ModuleList of Conv1d layers
        cnn_acts      : ModuleList of CNN activations
        out_layer     : Linear (cnn_dim -> output_dim)
        norm          : Affine <- never freeze

    TODO: ResTRF

    Returns:
        frozen_params (int), total_params (int)
    """

    if not transfer_learning:
        return 0, sum(p.numel() for p in model.parameters())

    total_params  = sum(p.numel() for p in model.parameters())
    frozen_params = 0

    def freeze(modules):
        nonlocal frozen_params
        if not isinstance(modules, list):
            modules = [modules]
        for m in modules:
            for param in m.parameters():
                param.requires_grad = False
                frozen_params += param.numel()

    def report(label, trainable_desc):
        pct = 100 * frozen_params / total_params
        print(f'TRANSFER LEARNING: {label} ({frozen_params}/{total_params} = {pct:.1f}% frozen)')
        print(f'TRANSFER LEARNING: Trainable: {trainable_desc}')

    # === ResMLP ===
    if isinstance(model, ResMLP):

        if freeze_strategy == 'none':
            print('TRANSFER LEARNING: No layers frozen - full fine-tuning (ResMLP)')

        elif freeze_strategy == 'early_1':
            freeze(model.model[0])
            report('Early 1 - input', 'ResBlocks 1,2,3 + Output + Affine')

        elif freeze_strategy == 'early_2':
            freeze([model.model[0], model.model[1]])
            report('Early 2 - input + Res1', 'ResBlocks 2,3 + Output + Affine')

        elif freeze_strategy == 'early_3':
            freeze([model.model[0], model.model[1], model.model[2]])
            report('Early 3 - input + Res1+2', 'ResBlock 3 + Output + Affine')

        elif freeze_strategy == 'early_4':
            freeze([model.model[0], model.model[1], model.model[2], model.model[3]])
            report('Early 4 - input + Res1+2+3', 'Output + Affine only')

        elif freeze_strategy == 'late_1':
            freeze(model.model[4])
            report('Late 1 - output', 'Input + ResBlocks 1,2,3 + Affine')

        elif freeze_strategy == 'late_2':
            freeze([model.model[4], model.model[3]])
            report('Late 2 - output + Res3', 'Input + ResBlocks 1,2 + Affine')

        elif freeze_strategy == 'late_3':
            freeze([model.model[4], model.model[3], model.model[2]])
            report('Late 3 - output + Res3+2', 'Input + ResBlock 1 + Affine')

        elif freeze_strategy == 'late_4':
            freeze([model.model[4], model.model[3], model.model[2], model.model[1]])
            report('Late 4 - output + Res3+2+1', 'Input + Affine only')

        elif freeze_strategy == 'input_output':
            freeze([model.model[0], model.model[4]])
            report('Input+Output', 'ResBlocks 1,2,3 + Affine')

        elif freeze_strategy == 'resnet_1':
            freeze(model.model[1])
            report('ResNet 1 only', 'Input + ResBlocks 2,3 + Output + Affine')

        elif freeze_strategy == 'resnet_2':
            freeze(model.model[2])
            report('ResNet 2 only', 'Input + ResBlocks 1,3 + Output + Affine')

        elif freeze_strategy == 'resnet_3':
            freeze(model.model[3])
            report('ResNet 3 only', 'Input + ResBlocks 1,2 + Output + Affine')

        elif freeze_strategy == 'resnet_12':
            freeze([model.model[1], model.model[2]])
            report('ResNet 1+2', 'Input + ResBlock 3 + Output + Affine')

        elif freeze_strategy == 'resnet_23':
            freeze([model.model[2], model.model[3]])
            report('ResNet 2+3', 'Input + ResBlock 1 + Output + Affine')

        elif freeze_strategy == 'resnet_123':
            freeze([model.model[1], model.model[2], model.model[3]])
            report('ResNet 1+2+3', 'Input + Output + Affine')

        elif freeze_strategy in ('early_5', 'early_6', 'early_7',
                                  'late_5', 'late_6', 'late_7',
                                  'cnn_only', 'cnn_transform', 'cnn_all'):
            raise ValueError(f"freeze_strategy='{freeze_strategy}' is only valid for ResCNN, not ResMLP.")

        else:
            raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}")

    # === ResCNN ===
    elif isinstance(model, ResCNN):

        if freeze_strategy == 'none':
            print('TRANSFER LEARNING: No layers frozen - full fine-tuning (ResCNN)')

        elif freeze_strategy == 'early_1':
            freeze(model.input_layer)
            report('Early 1 - input_layer', 'Res1+2+3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'early_2':
            freeze([model.input_layer, model.Res1, model.Act1])
            report('Early 2 - input + Res1', 'Res2+3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'early_3':
            freeze([model.input_layer, model.Res1, model.Act1, model.Res2, model.Act2])
            report('Early 3 - input + Res1+2', 'Res3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'early_4':
            freeze([model.input_layer, model.Res1, model.Act1,
                    model.Res2, model.Act2, model.Res3, model.Act3])
            report('Early 4 - input + Res1+2+3', 'CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'early_5':
            freeze([model.input_layer, model.Res1, model.Act1,
                    model.Res2, model.Act2, model.Res3, model.Act3,
                    model.CNN_transform])
            report('Early 5 - input + Res1+2+3 + CNN_transform', 'convs + out_layer + norm')

        elif freeze_strategy == 'early_6':
            freeze([model.input_layer, model.Res1, model.Act1,
                    model.Res2, model.Act2, model.Res3, model.Act3,
                    model.CNN_transform, model.convs, model.cnn_acts])
            report('Early 6 - input + Res1+2+3 + CNN_transform + convs', 'out_layer + norm only')

        elif freeze_strategy == 'early_7':
            freeze([model.input_layer, model.Res1, model.Act1,
                    model.Res2, model.Act2, model.Res3, model.Act3,
                    model.CNN_transform, model.convs, model.cnn_acts,
                    model.out_layer])
            report('Early 7 - everything except norm', 'norm only (essentially frozen)')

        elif freeze_strategy == 'late_1':
            freeze(model.out_layer)
            report('Late 1 - out_layer', 'input + Res1+2+3 + CNN_transform + convs + norm')

        elif freeze_strategy == 'late_2':
            freeze([model.out_layer, model.convs, model.cnn_acts])
            report('Late 2 - out_layer + convs', 'input + Res1+2+3 + CNN_transform + norm')

        elif freeze_strategy == 'late_3':
            freeze([model.out_layer, model.convs, model.cnn_acts, model.CNN_transform])
            report('Late 3 - out_layer + convs + CNN_transform', 'input + Res1+2+3 + norm')

        elif freeze_strategy == 'late_4':
            freeze([model.out_layer, model.convs, model.cnn_acts,
                    model.CNN_transform, model.Res3, model.Act3])
            report('Late 4 - out_layer + convs + CNN_transform + Res3', 'input + Res1+2 + norm')

        elif freeze_strategy == 'late_5':
            freeze([model.out_layer, model.convs, model.cnn_acts,
                    model.CNN_transform, model.Res3, model.Act3, model.Res2, model.Act2])
            report('Late 5 - out_layer + convs + CNN_transform + Res3+2', 'input + Res1 + norm')

        elif freeze_strategy == 'late_6':
            freeze([model.out_layer, model.convs, model.cnn_acts,
                    model.CNN_transform, model.Res3, model.Act3,
                    model.Res2, model.Act2, model.Res1, model.Act1])
            report('Late 6 - out_layer + convs + CNN_transform + Res3+2+1', 'input + norm only')

        elif freeze_strategy == 'late_7':
            freeze([model.out_layer, model.convs, model.cnn_acts,
                    model.CNN_transform, model.Res3, model.Act3,
                    model.Res2, model.Act2, model.Res1, model.Act1,
                    model.input_layer])
            report('Late 7 - everything except norm', 'norm only (essentially frozen)')

        elif freeze_strategy == 'input_output':
            freeze([model.input_layer, model.out_layer])
            report('Input+Output', 'Res1+2+3 + CNN_transform + convs + norm')

        elif freeze_strategy == 'resnet_1':
            freeze([model.Res1, model.Act1])
            report('ResNet 1 only', 'input + Res2+3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'resnet_2':
            freeze([model.Res2, model.Act2])
            report('ResNet 2 only', 'input + Res1+3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'resnet_3':
            freeze([model.Res3, model.Act3])
            report('ResNet 3 only', 'input + Res1+2 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'resnet_12':
            freeze([model.Res1, model.Act1, model.Res2, model.Act2])
            report('ResNet 1+2', 'input + Res3 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'resnet_23':
            freeze([model.Res2, model.Act2, model.Res3, model.Act3])
            report('ResNet 2+3', 'input + Res1 + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'resnet_123':
            freeze([model.Res1, model.Act1, model.Res2, model.Act2, model.Res3, model.Act3])
            report('ResNet 1+2+3', 'input + CNN_transform + convs + out_layer + norm')

        elif freeze_strategy == 'cnn_only':
            freeze([model.convs, model.cnn_acts])
            report('CNN convs only', 'input + Res1+2+3 + CNN_transform + out_layer + norm')

        elif freeze_strategy == 'cnn_transform':
            freeze(model.CNN_transform)
            report('CNN_transform only', 'input + Res1+2+3 + convs + out_layer + norm')

        elif freeze_strategy == 'cnn_all':
            freeze([model.CNN_transform, model.convs, model.cnn_acts])
            report('CNN_transform + convs (full CNN section)', 'input + Res1+2+3 + out_layer + norm')

        else:
            raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}")

    # === ResTRF (TODO) ===
    else:
        # TODO: Add ResTRF support
        raise NotImplementedError(
            f"freeze_layers() not implemented for architecture: {type(model).__name__}. "
            f"Currently supports ResMLP and ResCNN."
        )

    return frozen_params, total_params

#===================================================================================================
# TRAINING PROGRESS BAR
# Prints a live-updating progress bar to stdout during training.
#===================================================================================================

def progress_bar(train_loss, valid_loss, start_time, epoch, total_epochs, optim):
    '''
    Prints a live-updating progress bar to stdout during training.
    '''

    elapsed_time = int((datetime.now() - start_time).total_seconds())
    lr = optim.param_groups[0]['lr']
    epoch=epoch+1

    width = 20
    factor = int( width * (epoch/total_epochs) )
    bar = '['
    for i in range(width):
        if i < factor:
            bar += '#'
        else:
            bar += ' '
    bar += ']'

    remaining_time = int((elapsed_time / (epoch)) * (total_epochs - (epoch)))

    print('\r' + bar + ' ' +                                \
          f'Epoch {epoch:3d}/{total_epochs:3d} | ' +        \
          f'loss={train_loss:1.3e}({valid_loss:1.3e}) | ' + \
          f'lr={lr:1.2e} | ' +                              \
          f'time elapsed={elapsed_time:7d} s; time remaining={remaining_time:7d} s',end='')
    sys.stdout.flush()

#===================================================================================================
# TRAINING ROUTINE
# Loads data, builds model, runs training loop, evaluates on test set, saves model.
# Args:
#   train_yaml         (str):   path to training YAML
#   probe              (str):   'cosmic_shear', 'galaxy_galaxy_lensing', or 'galaxy_clustering'
#   n_epochs           (int):   number of training epochs (default=250)
#   batch_size         (int):   batch size (default=32)
#   learning_rate      (float): ADAM learning rate (default=1e-3)
#   weight_decay       (float): L2 regularization (default=0)
#   save_losses        (bool):  save losses to losses.txt (default=False)
#   save_testing       (bool):  save test metrics to <model>_metrics.txt (default=False)
#   transfer_learning  (bool):  enable transfer learning mode (default=False)
#   pretrained_model   (str):   path to pretrained .pt file (required if transfer_learning=True)
#   freeze_strategy    (str):   layer freezing strategy (default='none')
#===================================================================================================

def train_emulator(train_yaml, probe,
            n_epochs=250, batch_size=32, learning_rate=1e-3, weight_decay=0, 
            save_losses=False, save_testing=False, squeeze_factor=1.0,
            transfer_learning=False, pretrained_model=None, freeze_strategy='none'):
    '''
    Train a cosmological emulator (ResMLP, ResCNN, or ResTRF).

    Args:
        train_yaml        (str):   path to training YAML file
        probe             (str):   'cosmic_shear', 'galaxy_galaxy_lensing', or 'galaxy_clustering'
        n_epochs          (int):   number of training epochs (default=250)
        batch_size        (int):   batch size (default=32)
        learning_rate     (float): ADAM learning rate (default=1e-3)
        weight_decay      (float): L2 regularization weight decay (default=0)
        save_losses       (bool):  save train/valid losses to losses.txt (default=False)
        save_testing      (bool):  save test metrics to <model>_metrics.txt (default=False)
        squeeze_factor    (float): divide covariance by this factor (default=1.0, no squeezing)
        transfer_learning (bool):  load pretrained weights and fine-tune (default=False)
        pretrained_model  (str):   path to pretrained .pt file (required if transfer_learning=True)
        freeze_strategy   (str):   layer freezing strategy, see freeze_layers() (default='none')
    '''
    print('')
    print('Probe =', probe)

    # === Probe Configuration ===
    # TODO: get indices from cosmolike
    if probe=='galaxy_galaxy_lensing':
        start = 1080
        stop = 1995

    elif probe=='cosmic_shear':
        start = 0
        stop = 1080

    elif probe=='galaxy_clustering':
        start=1995
        stop = 2115

    else:
        raise NotImplementedError
    dv_idxs = np.array(range(start, stop))
    # if we want to train with a mask, get the mask, then we need to fix the sizes
    mask = np.ones(stop-start) 
    
    # === Load Configuration and File Paths ===
    with open(train_yaml,'r') as stream:
        args = yaml.safe_load(stream)

    if args['train_args']['training_data_path'][0] == '/':
        PATH = args['train_args']['training_data_path']
    else:
        PATH = os.environ.get("ROOTDIR") + '/' + args['train_args']['training_data_path']
     
    model_filename = args['train_args'][probe]['extra_args']['file'][0]
    extra_filename = args['train_args'][probe]['extra_args']['extra'][0]

    print('Model will be saved to:',model_filename,'and',extra_filename)
    print('')

    covmat_file      = PATH + args['train_args']['data_covmat_file'] # QUESTION: never used, safe to remove?

    train_datavectors_file = PATH + args['train_args']['train_datavectors_file']
    train_parameters_file  = PATH + args['train_args']['train_parameters_file']

    valid_datavectors_file = PATH + args['train_args']['valid_datavectors_file']
    valid_parameters_file  = PATH + args['train_args']['valid_parameters_file']

    test_datavectors_file = PATH + args['train_args']['test_datavectors_file']
    test_parameters_file  = PATH + args['train_args']['test_parameters_file']

    print('Loading training data from:')
    print(train_datavectors_file)
    print(train_parameters_file)
    print('Loading validation data from:')
    print(valid_datavectors_file)
    print(valid_parameters_file)
    print('Loading testing data from:')
    print(test_datavectors_file)
    print(test_parameters_file)
    print('')
    
    # === Build Model ===
    # get the parameters
    sampled_params = args['train_args'][probe]['extra_args']['ord'][0]
    sampling_dim = len(sampled_params)

    print('Parameters are:')
    print(sampled_params)
    print('')

    # get device
    device = args['train_args'][probe]['extra_args']['device']

    # get model
    model_info = args['train_args'][probe]['extra_args']['extrapar'][0]

    # Reordered by Béla: MLP -> CNN -> TRF for consistency with imports and freeze_layers()
    if( 'MLP' == model_info['MLA'] ):
        model = ResMLP(sampling_dim,
            int(np.sum(mask)),  # output_dim from mask, not model_info['OUTPUT_DIM']
            model_info['INT_DIM_RES'])
    elif( 'CNN' == model_info['MLA'] ):
        model = ResCNN(sampling_dim,
            int(np.sum(mask)),
            model_info['INT_DIM_RES'],
            model_info['CNN_DIM'],
            model_info['KERNEL_DIM'],
            model_info['N_CNN'])
    elif( 'TRF' == model_info['MLA'] ):
        model = ResTRF(sampling_dim,
            int(np.sum(mask)),  # output_dim from mask, not model_info['OUTPUT_DIM']
            model_info['INT_DIM_RES'],
            model_info['INT_DIM_TRF'],
            model_info['NC_TRF'])
    else:
        raise NotImplementedError
    
    # Note: output_dim equals stop-start until masking is fully implemented
    print('Model output shape (# unmasked elements) =', int(np.sum(mask)))
    print('')

    # === Transfer Learning Setup (added by Béla) ===
    # Loads pretrained weights and preprocessing from source model.
    # The pretrained model must have the same architecture as the target model.
    if transfer_learning:
        if pretrained_model is None:
            raise ValueError("transfer_learning=True but no --pretrained_model specified!")

        print(f'\nTRANSFER LEARNING: Loading pretrained model from {pretrained_model}')
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        model.load_state_dict(pretrained_state)

        # Load pretrained preprocessing. Reuses source normalization for consistency
        pretrained_h5 = pretrained_model.replace('.pt', '.h5')
        with h5.File(pretrained_h5, 'r') as hf:
            pretrained_samples_mean = torch.tensor(hf['sample_mean'][:], dtype=torch.float32)
            pretrained_samples_std  = torch.tensor(hf['sample_std'][:],  dtype=torch.float32)

        print(f'TRANSFER LEARNING: Loaded pretrained preprocessing parameters')
        print(f'TRANSFER LEARNING: Freeze strategy: {freeze_strategy}')

        frozen_params, total_params = freeze_layers(model, freeze_strategy, transfer_learning=True)
    else:
        pretrained_samples_mean = None
        pretrained_samples_std  = None
        freeze_layers(model, 'none', transfer_learning=False)

    # === Load Data ===
    print('Processing the data. May take some time...')

    # read the header from the train_parameters to get column ordering
    f_train = open(train_parameters_file)
    train_params = np.array(f_train.readline().split(' ')[1:])
    train_params[-1] = train_params[-1][:-1] # because the last param has a \n

    f_valid = open(valid_parameters_file)
    valid_params = np.array(f_valid.readline().split(' ')[1:])
    valid_params[-1] = valid_params[-1][:-1] # because the last param has a \n

    f_test = open(test_parameters_file)
    test_params = np.array(f_test.readline().split(' ')[1:])
    test_params[-1] = test_params[-1][:-1] # because the last param has a \n

    train_idxs = []
    valid_idxs = []
    test_idxs  = []
    
    for p in sampled_params:
        train_idxs.append(np.where(train_params==p)[0][0])
        valid_idxs.append(np.where(valid_params==p)[0][0])
        test_idxs.append(np.where(test_params==p)[0][0])

    # load the data of the given train_prefix and valid_prefix. Leave on cpu to save vram!
    # overwrite with my own manually so I can combine files. Add this functionality later!
    #print(f_train)
    #x_train = torch.as_tensor(f_train)

    x_train = torch.as_tensor(np.loadtxt(train_parameters_file)[:,train_idxs],dtype=torch.float32)
    y_train = torch.as_tensor(np.load(train_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    # Added by Béla: subsample training data if n_train is specified
    if n_train is not None:
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
        print(f'Using n_train={n_train} samples')

    x_valid = torch.as_tensor(np.loadtxt(valid_parameters_file)[:,valid_idxs],dtype=torch.float32)
    y_valid = torch.as_tensor(np.load(valid_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    x_test = torch.as_tensor(np.loadtxt(test_parameters_file)[:,test_idxs],dtype=torch.float32)
    y_test = torch.as_tensor(np.load(test_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    # convert data
    covmat = torch.as_tensor(get_cov(train_yaml, masked=training_masked, squeeze_factor=squeeze_factor)[start:stop, start:stop],dtype=torch.float32)
    #dv_fid = torch.as_tensor(torch.mean(y_train[dv_idxs],axis=0),dtype=torch.float64)
    dv_fid = torch.as_tensor(get_datavector(train_yaml)[dv_idxs],dtype=torch.float32)

    # === Preprocess Data ===
    # normalize the input parameters
    # Added by Béla: use pretrained normalization in transfer learning mode for consistency
    if transfer_learning:
        samples_mean = pretrained_samples_mean
        samples_std  = pretrained_samples_std
        print('TRANSFER LEARNING: Using pretrained normalization parameters')
    else:
        samples_mean = torch.Tensor(x_train.mean(axis=0, keepdims=True))
        samples_std  = torch.Tensor(x_train.std(axis=0, keepdims=True))


    x_train = torch.div( (x_train - samples_mean), 5.0*samples_std)
    x_valid = torch.div( (x_valid - samples_mean), 5.0*samples_std)
    x_test  = torch.div( (x_test  - samples_mean), 5.0*samples_std)

    # samples_cov  = torch.Tensor(torch.cov(torch.t(x_train)))
    # s_evals, s_evecs = torch.linalg.eigh(samples_cov)
    # x_train = torch.div((x_train - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))
    # x_valid = torch.div((x_valid - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))
    # x_test  = torch.div((x_test  - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))

    # x_scale = torch.max(x_train) - torch.min(x_train)
    # x_train = x_train/x_scale
    # x_valid = x_valid/x_scale
    # x_test  = x_test /x_scale

    # diagonalize the training datavectors
    dv_evals, dv_evecs = torch.linalg.eigh(covmat)
    #inv_covmat = torch.diag(1/dv_evals).type(torch.float32).to(device)

    y_train = torch.div( (y_train - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_valid = torch.div( (y_valid - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_test  = torch.div( (y_test  - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))

    y_scale = 1.0#torch.max(y_train) - torch.min(y_train)
    #print(y_scale)

    # convert to float32
    x_train = torch.as_tensor(x_train,dtype=torch.float32)
    y_train = torch.as_tensor(y_train,dtype=torch.float32)
    x_valid = torch.as_tensor(x_valid,dtype=torch.float32)
    y_valid = torch.as_tensor(y_valid,dtype=torch.float32)
    x_test  = torch.as_tensor(x_test, dtype=torch.float32)
    y_test  = torch.as_tensor(y_test, dtype=torch.float32)

    # I'm going to try adding a new transform that just rescales everything to be between 0 and 1.
    #rescale = torch.max(y_train) - torch.min(y_train)
    #rescale = rescale.to(device)
    #print(rescale.get_device())

    output_dim = int(np.sum(mask))

    # === Setup Optimizer and Data Loaders ===
    # Added by Béla: optimize trainable parameters only in transfer learning mode
    if transfer_learning:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        print(f'TRANSFER LEARNING: Optimizer using {len(trainable_params)} trainable parameter tensors')
    else:
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=10, factor=0.5)

    model.to(device)

    generator = torch.Generator(device=device) # QUESTION: generator is unused, safe to remove?
    trainset    = torch.utils.data.TensorDataset(x_train, y_train)
    validset    = torch.utils.data.TensorDataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)

    # === Training Loop ===
    print('\rBegin training...', end='')
    train_start_time = datetime.now()

    losses_train = []
    losses_valid = []
    loss = 100.

    tmp_evecs = dv_evecs.to(device)
    tmp_evals = dv_evals.to(device)
    tmp_dvfid = dv_fid.to(device)
    tmp_invcovmat = torch.linalg.inv(covmat).to(device) # QUESTION: all of these are only used in commented-out code, safe to remove?

    for e in range(n_epochs):
        model.train()

        # === Training Loss ===
        losses = []
        for i, data in enumerate(trainloader):    
            X       = data[0].to(device)
            Y_batch = data[1].to(device)
            Y_pred  = model(X)*y_scale

            # PCA part
            diff = Y_batch - Y_pred
            chi2 = torch.bmm(diff.view(batch_size,1,output_dim), diff.view(batch_size, output_dim, 1))
            # i give up, lets just change basis back.
            #Y_pred  = torch.mul(Y_pred,  torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
            #Y_batch = torch.mul(Y_batch, torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
            #chi2 = torch.diag((Y_pred - Y_batch) @ tmp_invcovmat @ torch.t(Y_pred - Y_batch))

            # loss = torch.mean(chi2)                      # ordinary chi2
            # loss = torch.mean((1+2*chi2)**(1/2))-1       # hyperbola
            loss = torch.mean(chi2**(1/2))                 # sqrt(chi2)

            losses.append(loss.cpu().detach().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

        losses_train.append(np.mean(losses))

        # === Validation Loss ===
        losses=[]
        with torch.no_grad():
            model.eval()
            losses = []
            for i, data in enumerate(validloader):  
                X_v       = data[0].to(device)
                Y_v_batch = data[1].to(device)
                Y_v_pred = model(X_v)*y_scale

                diff_v = Y_v_batch - Y_v_pred
                #chi2_v = torch.diag(diff_v @ torch.t(diff_v)) # QUESTION: safe to remove?
                chi2_v = torch.bmm(diff_v.view(batch_size,1,output_dim), diff_v.view(batch_size, output_dim, 1))

                # Y_v_pred  = torch.mul(Y_v_pred,  torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
                # Y_v_batch = torch.mul(Y_v_batch, torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
                # chi2_v = torch.diag((Y_v_pred - Y_v_batch) @ tmp_invcovmat @ torch.t(Y_v_pred - Y_v_batch)) # QUESTION: safe to remove?

                # loss_vali = torch.mean(chi2_v)                      # ordinary chi2
                # loss_vali = torch.mean((1+2*chi2_v)**(1/2))-1       # hyperbola
                loss_vali = torch.mean(chi2_v**(1/2))                 # sqrt(chi2)

                losses.append(float(loss_vali.cpu().detach().numpy()))

            losses_valid.append(np.mean(losses))

            scheduler.step(losses_valid[e])
            optim.zero_grad()

        progress_bar(losses_train[-1],losses_valid[-1],train_start_time, e, n_epochs, optim)
    
    # Added by Béla (saves unique filename for losses file)
    if ( save_losses ):
        losses_filename = model_filename.replace('.pt', '_losses.txt')
        np.savetxt(losses_filename, np.array([losses_train,losses_valid],dtype=np.float32))

    # === Save Model ===
    torch.save(model.state_dict(), model_filename)

    with h5.File(extra_filename, 'w') as f:
        f['sample_mean']  = samples_mean
        #f['s_evecs']       = s_evecs # QUESTION: safe to remove?
        #f['s_evals']       = s_evals # QUESTION: safe to remove?
        f['sample_std']   = 5.0*samples_std
        f['dv_fid']        = dv_fid
        f['dv_evals']      = torch.sqrt(dv_evals)
        f['dv_evecs']      = dv_evecs 
        #f['rescaling']     = rescale # QUESTION: safe to remove?
        f['train_params']  = sampled_params
        #f['model_info']    = model_info # QUESTION: safe to remove?


    # === Test Model ===
    print('')
    print('Testing the model...')
    print('')

    # Reminder:
    # C = UDU^{-1}
    # dv_norm = D^{-1/2} U^{-1} dv
    # where D is diagonal and U is orthogonal, so
    # dv_norm.T = dv.T U D^{-1/2}
    # 
    # chi2 = dv.T @ C^{-1} @ dv
    #      = dv.T @ UD^{-1}U^{-1} @ dv
    #      = dv.T U @ D^{-1/2} D^{-1/2} @ U^{-1} dv
    #      = (dv.T U D^{-1/2}) @ (D^{-1/2} U^{-1} dv)
    #      = dv_norm.T @ dv_norm
    # and dv_norm is just the model output!

    with torch.no_grad():
        model.eval()

        Y_t = model(x_test.to(device))*y_scale
        diff = y_test.to(device) - Y_t
        delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
        #Y_pred  = torch.mul(Y_t,  torch.sqrt(dv_evals).to(device)) @ torch.t(dv_evecs.to(device)) + dv_fid.to(device)
        #Y_batch = torch.mul(y_test, torch.sqrt(dv_evals)) @ torch.t(dv_evecs) + dv_fid
        #delta_chi2 = torch.diag((Y_pred.to('cpu') - Y_batch) @ torch.linalg.inv(covmat) @ torch.t(Y_pred.to('cpu') - Y_batch))

        # want to do a manual check of chi2 to see if they are the same? # THEY ARE! so why is my NN performing so bad on real data?
        # y_test_data_basis = torch.mul(y_test, torch.sqrt(dv_evals))
        # y_test_data_basis = y_test_data_basis @ torch.t(dv_evecs) + dv_fid

        # Y_t_data_basis = torch.mul(Y_t.to('cpu'), torch.sqrt(dv_evals)) 
        # Y_t_data_basis = Y_t_data_basis @ torch.t(dv_evecs) + dv_fid
        # diff_data_basis = Y_t_data_basis - y_test_data_basis
        # delta_chi2_data_basis = torch.diag(diff_data_basis @ torch.linalg.inv(covmat) @ torch.t(diff_data_basis))
        # print(delta_chi2.to('cpu') - delta_chi2_data_basis)
        # print(torch.mean(delta_chi2))
        # print(torch.mean(delta_chi2_data_basis))  # QUESTION: safe to remove?

        chi2_g_1  = 0
        chi2_g_p2 = 0

        for c in delta_chi2:
            if( c>0.2 ):
                chi2_g_p2 += 1
            if( c>=1 ):
                chi2_g_1 += 1

        print('Testing results.')
        print('Mean   Delta Chi2 = {:1.3e}'.format(torch.mean(delta_chi2).cpu().detach().numpy()))
        print('Median Delta Chi2 = {:1.3e}'.format(torch.median(delta_chi2).cpu().detach().numpy()))
        print('N points with Chi2 > 1  :', chi2_g_1)
        print('N points with Chi2 > 0.2:', chi2_g_p2)

        # Done :)
        print('\nDone!')

        # === Save Testing Metrics (added by Béla) ===
        if save_testing:
            metrics_filename = model_filename.replace('.pt', '_metrics.txt')
            with open(metrics_filename, 'w') as mf:
                mf.write(f'mean_chi2 {torch.mean(delta_chi2).cpu().detach().numpy():.6e}\n')
                mf.write(f'median_chi2 {torch.median(delta_chi2).cpu().detach().numpy():.6e}\n')
                mf.write(f'n_outliers_0p2 {chi2_g_p2}\n')
                mf.write(f'n_outliers_1 {chi2_g_1}\n')
                mf.write(f'n_test {len(delta_chi2)}\n')
            print(f'Saved testing metrics to: {metrics_filename}')

        # === Debugging: Best Point Analysis (Yijie) ===
        # Finds the best-performing point on test, valid, and train sets for comparison against Cocoa
        if not save_testing:
            Y_t = model(x_test.to(device))
            diff = y_test.to(device) - Y_t
            delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
            delta_chi2_testing = torch.min(delta_chi2)
            idx_testing = torch.where(delta_chi2==delta_chi2_testing)
            print(delta_chi2_testing,idx_testing)

            Y_v = model(x_valid.to(device))
            diff = y_valid.to(device) - Y_v
            delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
            delta_chi2_valid = torch.min(delta_chi2)
            idx_valid = torch.where(delta_chi2==delta_chi2_valid)
            print(delta_chi2_valid,idx_valid)

            Y_t = model(x_train.to(device))
            diff = y_train.to(device) - Y_t
            delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
            delta_chi2_train = torch.min(delta_chi2)
            idx_train = torch.where(delta_chi2==delta_chi2_train)
            print(delta_chi2_train,idx_train)

    return

# QUESTION: n_train and training_masked are accessed as globals inside train_emulator() rather than
# passed as parameters like the other args. Is it ok to refactor to pass explicitly?
if __name__ == "__main__":
    train_emulator(cobaya_yaml, probe, 
        n_epochs, batch_size, learning_rate, weight_decay, 
        save_losses, save_testing_metrics, squeeze_factor,
        transfer_learning, pretrained_model, freeze_strategy)