# Run the exercise03 program as required pipeline

# Define Arguments
BATCH_SIZE=64
TEST_BATCH_SIZE=1000
EPOCHS=30
LEARNING_RATE=0.01
NO_CUDA=0
SEED=1
LOG_INTERVAL=10
# Optimizer Choices: "SGD", "Adam", "RMSprop", "Adadelta"
OPTIMIZER="SGD"
# Model Choices: "CNN", "MLP"
MODEL="MLP"
# Data Choices: "MNIST", "CIFAR10"
DATA="MNIST"


# Define a learning rate list from 0.1 to 0.01 with logspace
LEARNING_RATE_LIST=(0.1 0.0599 0.0359 0.0215 0.0129 0.0077 0.0046 0.0028 0.0017 0.0010)

# Task 1: Train a model with the MLP architecture on the MNIST dataset. 30 Epochs. One Run on CPU
python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --no-cuda --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA

# Task 2: Train a model with the MLP architecture on the MNIST dataset. 30 Epochs. One Run on GPU
python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA

# Task 3: Switch Data to CIFAR10. Train a model with the MLP architecture on the CIFAR10 dataset. 30 Epochs. One Run on GPU
DATA="CIFAR10"
python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA

# Task 4: Switch Model to CNN. Train a model with the CNN architecture on the CIFAR10 dataset. 30 Epochs. One Run on GPU
MODEL="CNN"
python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA

# Define a learning rate scheduler


# Task 5: Run with CIFAR10 dataset, CNN model, 30 epochs, with SGD optimizer, on GPU and tranverse the learning rate list
for lr in "${LEARNING_RATE_LIST[@]}"
do
    python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $lr --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA
done

# Task 6: Run with CIFAR10 dataset, CNN model, 30 epochs, with Adam optimizer, on GPU and tranverse the learning rate list
OPTIMIZER="Adam"
for lr in "${LEARNING_RATE_LIST[@]}"
do
    python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $lr --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA
done

# Task 7: Run with CIFAR10 dataset, CNN model, 30 epochs, with RMSprop optimizer, on GPU and tranverse the learning rate list
OPTIMIZER="RMSprop"
for lr in "${LEARNING_RATE_LIST[@]}"
do
    python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $lr --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA
done

# Task 8: Run with CIFAR10 dataset, CNN model, 30 epochs, with Adagrad optimizer, on GPU and tranverse the learning rate list
OPTIMIZER="Adagrad"
for lr in "${LEARNING_RATE_LIST[@]}"
do
    python3 exercise03.py --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE --epochs $EPOCHS --lr $lr --seed $SEED --log-interval $LOG_INTERVAL --optimizer $OPTIMIZER --model $MODEL --data $DATA
done

