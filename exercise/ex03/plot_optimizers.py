import matplotlib.pyplot as plt
import re
import argparse

def plot_lr(prefixs, lrs):
    data = {}
    for i, prefix in enumerate(prefixs):
        filename = f"logs/datalog_{prefix}.txt"
        # Read the content of the file
        with open(filename, 'r') as file:
            text = file.read()
        # Split the text into blocks divided by lines of hyphens
        blocks = re.split(r'-{38,}', text)
        # Filter blocks that contain the specific learning rate
        lr_pattern = rf"lr={lrs[i]},"
        filtered_block = [block for block in blocks if re.search(lr_pattern, block)][0]
        if prefix == 'Adam':
            print([block for block in blocks if re.search(lr_pattern, block)])
        test_info_block = re.search(r"Test Info: \[(.*?)\]", filtered_block, re.DOTALL).group(1)
        accuracies = [float(match) for match in re.findall(r"'accuracy': (\d+\.\d+)", test_info_block)]
        epochs = [int(match) for match in re.findall(r"'epoch': (\d+)", test_info_block)]

        data[prefix] = {
            'epochs': epochs,
            'accuracies': accuracies,
            'lr': lrs[i]
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for optimizer, info in data.items():
        label = f"{optimizer} (LR={info['lr']})"
        ax.plot(info['epochs'], info['accuracies'], label=label, marker='o')

    ax.set_title('Test Accuracy for Different Optimizers')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

    plt.show()

    # Save the plot
    fig.savefig(f'figures/final.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot test accuracy vs epochs for different optimizers')
    parser.add_argument('--prefixs', nargs='+', help='Prefixes of the log files')
    parser.add_argument('--lrs', nargs='+', type=float, help='Learning rates to plot')
    args = parser.parse_args()

    plot_lr(args.prefixs, args.lrs)
    