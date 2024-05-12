import matplotlib.pyplot as plt
import re
import argparse

def plot_lr(prefix):
    filename = file_name = "logs/datalog_" + prefix + ".txt"
    # Read the content of the file
    with open(filename, 'r') as file:
        text = file.read()
    lrs = re.findall(r"lr=(0.\d+)", text)
    lrs = [float(lr) for lr in lrs]
    # Regex to extract all test info blocks
    test_info_blocks = re.findall(r"Test Info: \[(.*?)\]", text, re.DOTALL)
    # Dictionary to hold data by learning rate
    test_data_by_lr = {}

    # Process each block
    for i, block in enumerate(test_info_blocks):
        lr = lrs[i]
        epochs = re.findall(r"'epoch': (\d+)", block)
        accuracies = re.findall(r"'accuracy': (\d+\.\d+)", block)

        # Convert strings to appropriate types
        epochs = [int(epoch) for epoch in epochs]
        accuracies = [float(acc) for acc in accuracies]

        # Store in dictionary
        test_data_by_lr[lr] = {
            'epochs': epochs,
            'accuracies': accuracies
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for lr, info in test_data_by_lr.items():
        ax.plot(info['epochs'], info['accuracies'], label=f'LR={lr}', marker='o')

    ax.set_title(f'Test Accuracy vs Epochs for {prefix}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

    plt.show()

    # Save the plot
    fig.savefig(f'figures/{prefix}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot test accuracy vs epochs for different learning rates')
    parser.add_argument('--prefix', type=str, help='Prefix of the log file')
    args = parser.parse_args()

    plot_lr(args.prefix)
    