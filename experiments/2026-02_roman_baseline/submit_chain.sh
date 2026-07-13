#!/bin/bash
# Submit 9 sequential append jobs for Roman CS Taka 100k each (900k total)
# Run from: /gpfs/projects/MirandaGroup/bela/cocoa/Cocoa

SBATCH_SCRIPT="projects/roman_real/scripts/roman_cs_dv_T200_train.sbatch"

JID=$(sbatch $SBATCH_SCRIPT | awk '{print $NF}')
echo "Submitted job 1: $JID"

for i in $(seq 2 9); do
  JID=$(sbatch --dependency=afterok:$JID $SBATCH_SCRIPT | awk '{print $NF}')
  echo "Submitted job $i: $JID"
done

echo "All 9 jobs submitted. Final job ID: $JID"
echo "Run 'sq' to check queue."
