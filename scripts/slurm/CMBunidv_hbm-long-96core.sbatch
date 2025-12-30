#!/bin/bash
#SBATCH --job-name=cmblhcdv
#SBATCH --output=cmblhcdv_%A_%a.txt
#SBATCH --time=48:00:00
#SBATCH -p hbm-long-96core
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=8


echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Slurm array task ID is $SLURM_ARRAY_TASK_ID     # Slurm Array ID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

sleep_time=$(( 5 + (SLURM_ARRAY_TASK_ID % 20) * 3))
sleep $sleep_time

source ~/.bashrc
source /gpfs/software/Anaconda/etc/profile.d/conda.sh # Load conda from system Anaconda
conda init bash
cd /gpfs/projects/MirandaGroup/bela/cocoa/Cocoa
source stop_cocoa.sh
conda deactivate
conda activate cocoa_bela
source start_cocoa.sh
cd LCDM

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=8
fi
declare -i init=$((SLURM_ARRAY_TASK_ID*5))
declare -i temp=$((SLURM_ARRAY_TASK_ID+1))
declare -i end=$((temp*5))

echo "Processing files $init to $((end-1))"

for ((j=$init; j<$end; j++)); do
    echo "Starting file $j at $(date)"
    mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} python ./CMBunidvbela.py -f $j
    echo "Completed file $j at $(date)"
done
