#!/bin/bash
#SBATCH --job-name=cmb_single
#SBATCH --output=LCDM/output/logs/cmb_single_%A.txt
#SBATCH --time=48:00:00
#SBATCH -p hbm-long-96core
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=8

BATCH_NUM=$1

source ~/.bashrc
source /gpfs/software/Anaconda/etc/profile.d/conda.sh
conda activate cocoa_bela
cd /gpfs/projects/MirandaGroup/bela/cocoa/Cocoa
source start_cocoa.sh
cd LCDM

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "Processing batch ${BATCH_NUM} at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} python ./CMBunidvbela.py -f ${BATCH_NUM}
echo "Completed batch ${BATCH_NUM} at $(date)"
