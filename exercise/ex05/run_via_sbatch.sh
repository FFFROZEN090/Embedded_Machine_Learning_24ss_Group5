#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise05_template.py"

# For the first task, use 50 epochs and ResNet, save the results in file "output01.txt"
python exercise05_template.py --epochs 50 --model ResNet --output_file output01.txt

# For the second task, use VGG and ConvNet, save the results in file "output01.txt"
python exercise05_template.py --epochs 50 --model VGG11 --output_file output01.txt
python exercise05_template.py --epochs 50 --model ConvNet --output_file output01.txt




