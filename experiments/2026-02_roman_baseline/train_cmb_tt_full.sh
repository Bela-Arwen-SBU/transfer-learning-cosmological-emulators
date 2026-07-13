#!/bin/bash
#SBATCH --job-name=cmb_train_tt
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=train_tt_full_%j.log
#SBATCH --error=train_tt_full_%j.err

# Load environment
cd /gpfs/projects/MirandaGroup/bela/cocoa/Cocoa
eval "$(conda shell.bash hook)"
conda activate cocoa_bela
source start_cocoa.sh
cd LCDM

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Delete old normalization file to regenerate with fix
rm -f output/CMB_HIGHACC_LCDM/emul_CLTT_HIGHACC_LCDM_extra.npy

# Run full 700-epoch training
python train_cmb_tt_emulator.py --mode tt --batch 512 --epoch 700

echo "Job finished at: $(date)"
