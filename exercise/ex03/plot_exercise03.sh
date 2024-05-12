#!/bin/bash
# # ex3.1
# python plot.py --prefix1 "cpu" --prefix2 "gpu"

# # ex3.2
# python plot.py --prefix1 "MLP" --prefix2 "CNN"
# python plot.py --prefix1 "MLP" --prefix2 "CNN" --use_epochs

# ex3.3
OPTIMIZERS_LIST=("SGD" "Adam" "RMSprop" "Adagrad")
# for optimizer in "${OPTIMIZERS_LIST[@]}"
# do
#     python plot_lrs.py --prefix $optimizer
# done
LRS_LIST=(0.1 0.0010 0.0010 0.0077)
python plot_optimizers.py --prefixs "${OPTIMIZERS_LIST[@]}" --lrs "${LRS_LIST[@]}"