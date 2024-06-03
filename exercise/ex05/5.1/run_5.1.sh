#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise5.1.py"
LIST=("Identity" "BatchNorm" "GroupNorm")
for i in "${LIST[@]}"
do
    python exercise5.1.py --norm_layer $i
done