#!/bin/bash
#SBATCH --job-name=roman_cs_dv_validtest
#SBATCH --output=roman_cs_dv_validtest_%j.txt
#SBATCH --time=12:00:00
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

# --- taka T200 valid ---
echo "Starting taka T200 valid at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_takahashi_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 100 \
        --maxcorr 0.15 \
        --datavsfile taka_T200/roman_real_lcdm_taka_valid_T100 \
        --paramfile taka_T200/roman_real_lcdm_taka_valid_T100 \
        --failfile taka_T200/roman_real_lcdm_taka_failed_valid_T100 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

# --- taka T200 test ---
echo "Starting taka T200 test at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_takahashi_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 100 \
        --maxcorr 0.15 \
        --datavsfile taka_T200/roman_real_lcdm_taka_test_T100 \
        --paramfile taka_T200/roman_real_lcdm_taka_test_T100 \
        --failfile taka_T200/roman_real_lcdm_taka_failed_test_T100 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

# --- taka T500 valid ---
echo "Starting taka T500 valid at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_takahashi_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 250 \
        --maxcorr 0.15 \
        --datavsfile taka_T500/roman_real_lcdm_taka_valid_T250 \
        --paramfile taka_T500/roman_real_lcdm_taka_valid_T250 \
        --failfile taka_T500/roman_real_lcdm_taka_failed_valid_T250 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

# --- taka T500 test ---
echo "Starting taka T500 test at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_takahashi_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 250 \
        --maxcorr 0.15 \
        --datavsfile taka_T500/roman_real_lcdm_taka_test_T250 \
        --paramfile taka_T500/roman_real_lcdm_taka_test_T250 \
        --failfile taka_T500/roman_real_lcdm_taka_failed_test_T250 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

# --- mead T500 valid ---
echo "Starting mead T500 valid at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_mead2020_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 250 \
        --maxcorr 0.15 \
        --datavsfile mead_T500/roman_real_lcdm_mead_valid_T250 \
        --paramfile mead_T500/roman_real_lcdm_mead_valid_T250 \
        --failfile mead_T500/roman_real_lcdm_mead_failed_valid_T250 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

# --- mead T500 test ---
echo "Starting mead T500 test at $(date)"
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self \
    --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
    python ./external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
        --root projects/roman_real \
        --fileroot emulators/nla_cosmic_shear \
        --yaml lcdm_mead2020_cs_CNN.yaml \
        --nparams 10000 \
        --unif 0 \
        --temp 250 \
        --maxcorr 0.15 \
        --datavsfile mead_T500/roman_real_lcdm_mead_test_T250 \
        --paramfile mead_T500/roman_real_lcdm_mead_test_T250 \
        --failfile mead_T500/roman_real_lcdm_mead_failed_test_T250 \
        --freqchk 2000 \
        --loadchk 0 \
        --append 0

echo "Done at $(date)"
