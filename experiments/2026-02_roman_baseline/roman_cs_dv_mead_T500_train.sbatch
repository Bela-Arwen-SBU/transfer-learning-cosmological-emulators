#!/bin/bash
#SBATCH --job-name=roman_cs_dv_mead_T500_train
#SBATCH --output=roman_cs_dv_mead_T500_train_%j.txt
#SBATCH --time=48:00:00
#SBATCH -p hbm-long-96core
#SBATCH --nodes=2
#SBATCH --ntasks=192
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
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_mead2020_cs_CNN.yaml \
        --nparams 100000 \
        --unif 0 \
        --temp 500 \
        --maxcorr 0.15 \
        --datavsfile mead_T500/roman_real_lcdm_mead_train_T500 \
        --paramfile mead_T500/roman_real_lcdm_mead_train_T500 \
        --failfile mead_T500/roman_real_lcdm_mead_failed_train_T500 \
        --freqchk 2000 \
        --loadchk 1 \
        --append 1

echo "Done at $(date)"
