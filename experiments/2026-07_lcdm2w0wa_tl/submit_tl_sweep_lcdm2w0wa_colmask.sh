#!/bin/bash
# Usage: sbatch submit_tl_sweep_lcdm2w0wa_colmask.sh <freeze_strategy>
#   e.g. sbatch submit_tl_sweep_lcdm2w0wa_colmask.sh early_1
# Column-masked runs: early_1 (old embedding locked, rest trains) | late_4
# (only the 2 new input columns + affine train) | none (full FT, old cols locked)
#SBATCH --job-name=tl_l2w_cm
#SBATCH --output=/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_lcdm2w0wa_tl_colmask_T500_run1_clip90_intdim362/sweep_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h200x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G

STRATEGY="$1"
case "${STRATEGY}" in
    none|early_1|late_4) ;;
    *) echo "ERROR: pass one of: none early_1 late_4"; exit 1 ;;
esac

module purge
module load miniconda/3

source ~/.bashrc
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
conda activate cocoa_bela

cd /lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa
source stop_cocoa.sh
conda deactivate
conda activate cocoa_bela
source start_cocoa.sh

export FREEZE_STRATEGY="${STRATEGY}"
./projects/roman_real/scripts/run_tl_sweep_lcdm2w0wa_colmask_taka_T500_run1_lr3_bs32.sh
