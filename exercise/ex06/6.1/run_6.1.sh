#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise6.1.py"
python exercise6.1.py --epochs 50 
# python plot6.1.py
