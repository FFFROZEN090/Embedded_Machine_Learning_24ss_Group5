import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    pruning_results = {}
    bops_list = []
    current_pruning_rate = None
    last_train_loss = None
    last_epoch = None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            if 'Starting training with pruning rate:' in line:
                match = re.search(r'Starting training with pruning rate: ([0-9.]+)', line)
                if match:
                    if current_pruning_rate is not None and last_train_loss is not None:
                        # Save the last train loss if present
                        pruning_results[current_pruning_rate]['train_losses'].append(last_train_loss)
                    
                    current_pruning_rate = float(match.group(1))
                    pruning_results[current_pruning_rate] = {'epochs': [], 'train_losses': [], 'test_losses': [], 'test_accuracies': []}
                    print(f"Starting training with pruning rate: {current_pruning_rate}")  # Debug statement
            
            elif 'Train Epoch' in line and current_pruning_rate is not None:
                epoch_match = re.search(r'Train Epoch: (\d+)', line)
                loss_match = re.search(r'Loss: ([0-9.]+)', line)
                if epoch_match and loss_match:
                    epoch = int(epoch_match.group(1))
                    loss = float(loss_match.group(1))
                    if last_epoch != epoch:  # New epoch, save last train loss
                        if last_train_loss is not None:
                            pruning_results[current_pruning_rate]['train_losses'].append(last_train_loss)
                        last_epoch = epoch
                        pruning_results[current_pruning_rate]['epochs'].append(epoch)
                    last_train_loss = loss

            elif 'Test Epoch' in line and current_pruning_rate is not None:
                if last_train_loss is not None:
                    pruning_results[current_pruning_rate]['train_losses'].append(last_train_loss)
                    last_train_loss = None
                loss_match = re.search(r'Average loss: ([0-9.]+)', line)
                accuracy_match = re.search(r'Accuracy: [0-9]+/[0-9]+ \(([0-9.]+)%\)', line)
                if loss_match and accuracy_match:
                    loss = float(loss_match.group(1))
                    accuracy = float(accuracy_match.group(1))
                    pruning_results[current_pruning_rate]['test_losses'].append(loss)
                    pruning_results[current_pruning_rate]['test_accuracies'].append(accuracy)

            elif 'Pruning rate:' in line and 'BOPs:' in line:
                match = re.search(r'Pruning rate: ([0-9.]+), BOPs: ([0-9]+)', line)
                if match:
                    bops = int(match.group(2))
                    pruning_results[current_pruning_rate]['bops'] = bops
                    bops_list.append((current_pruning_rate, bops))
                    print(f"Completed training for pruning rate: {current_pruning_rate}, BOPs: {bops}")  # Debug statement

    print(f"Pruning results: {pruning_results}")  # Debug statement
    print(f"BOPs list: {bops_list}")  # Debug statement

    # Check if 0.8 pruning rate is present
    if 0.8 not in pruning_results:
        print("Warning: Pruning rate 0.8 not found in the log file.")
    else:
        print("Pruning rate 0.8 found in the log file.")

    return pruning_results, bops_list

def plot_pruning_results(pruning_results):
    plt.figure()
    for pruning_rate, results in pruning_results.items():
        epochs = results['epochs']
        test_accuracies = results['test_accuracies']
        bops = results.get('bops', 'N/A')
        if epochs and test_accuracies:
            print(f"Plotting pruning rate: {pruning_rate}")  # Debug statement
            label = f'Pruning rate {pruning_rate} (BOPs: {bops})'
            plt.plot(epochs, test_accuracies, marker='o', markersize=2, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison Across Pruning Rates (Unstructured Pruning)')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    plt.clf()

def main():
    log_file = 'training_log.txt'
    pruning_results, bops_list = parse_log(log_file)
    plot_pruning_results(pruning_results)

if (__name__) == '__main__':
    main()
