#!/bin/bash
#SBATCH --job-name=tl_strategy_sweep_taka500_to_mead500
#SBATCH --output=/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500/sweep_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h200x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
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
./projects/roman_real/scripts/run_tl_strategy_sweep_taka500_to_mead500.sh
