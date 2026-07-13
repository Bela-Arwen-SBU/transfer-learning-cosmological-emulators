#!/bin/bash
#SBATCH --job-name=roman_cs_dv_T250_test
#SBATCH --output=roman_cs_dv_T250_test_%j.txt
#SBATCH --time=4:00:00
#SBATCH -p short-96core
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1

echo "Available CPUs: $SLURM_JOB_CPUS_PER_NODE"
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of tasks is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

source ~/.bashrc
source /gpfs/software/Anaconda/etc/profile.d/conda.sh
conda init bash
cd /gpfs/projects/MirandaGroup/bela/cocoa/Cocoa
source stop_cocoa.sh
conda deactivate
conda activate cocoa_bela
source start_cocoa.sh

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_NUM_THREADS=1

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./projects/roman_real/scripts/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_takahashi_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 250 \
        --maxcorr 0.15 \
        --datavsfile roman_real_lcdm_taka_test_T250 \
        --paramfile roman_real_lcdm_taka_test_T250 \
        --failfile roman_real_lcdm_taka_failed_test_T250 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

echo "Done at $(date)"
