import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.dates as mdates

def parse_training_data(filename):
    """ Parses the training data file for model performance metrics. """
    with open(filename, 'r') as file:
        content = file.read()

    blocks = content.split("####################")[1:-1]  # Split blocks of data
    data = []
    for block in blocks:
        # Skip this block if it is empty or '\n'
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        model_info = lines[10].split()
        model = model_info[1]
        epoch_data = [line for line in lines if line.startswith('Train Epoch:')]
        test_data = [line for line in lines if line.startswith('Current time:')]
        parameter_info = [line for line in lines if line.startswith('Number of parameters:')][0]

        start_time = float([line for line in lines if line.startswith('Starting training at:')][0].split(': ')[1])
        end_time = float([line for line in lines if line.startswith('Finished training at:')][0].split(': ')[1])
        duration = end_time - start_time

        num_params = int(parameter_info.split(': ')[1].split()[0])
        macs = int(lines[-2].split(': ')[1])
        
        # Calculate the time per epoch based on the total duration and number of epochs
        epoch_times = [start_time + i * (duration / len(epoch_data)) for i in range(len(epoch_data))]

        for i, line in enumerate(epoch_data):
            epoch, loss = int(line.split(',')[0].split()[-1]), float(line.split(':')[-1])
            data.append({
                'Model': model,
                'Epoch': epoch,
                'Training Loss': loss,
                'Num Params': num_params,
                'MACs': macs,
                'Epoch Time': epoch_times[i]  # Assign calculated epoch time
            })

        for line in test_data:
            epoch = int(line.split(';')[1].split(',')[0].split()[-1])
            loss = float(line.split('loss:')[1].split(',')[0])
            accuracy_string = line.split('Accuracy:')[1].strip()
            accuracy = float(accuracy_string.split('/')[0]) / float(accuracy_string.split('/')[1].split()[0]) * 100
            data.append({
                'Model': model,
                'Epoch': epoch,
                'Testing Loss': loss,
                'Accuracy': accuracy,
                'Num Params': num_params,
                'MACs': macs
            })

    return pd.DataFrame(data)


def plot_loss_over_time(data):
    """ Plots training loss over calculated epoch times for each model. """
    # Ensure the 'Epoch Time' is in the correct datetime format
    data['Epoch Time'] = pd.to_datetime(data['Epoch Time'], unit='s')
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x='Epoch Time', y='Training Loss', hue='Model', style='Model', markers=True, dashes=False)
    plt.title('Training Loss Over Time for Different Models')
    plt.xlabel('Time')
    plt.ylabel('Training Loss')
    
    # Improve the x-axis with date formatting
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.xticks(rotation=45)
    
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    plt.savefig('loss_over_time.png')



def plot_loss_over_epoch(data):
    """ Plots training loss over training duration for each model. """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x='Epoch', y='Training Loss', hue='Model', style='Model', markers=True, dashes=False)
    plt.title('Training Loss Over Epochs for Different Models')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    plt.savefig('loss_over_epoch.png')

def plot_accuracy_vs_epoch(data):
    """ Plots accuracy vs. epochs for each model. """
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Epoch', y='Accuracy', hue='Model', marker='o', linestyle='-')
    plt.title('Accuracy vs. Epochs for Different Models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    plt.savefig('accuracy_vs_epoch.png')

def plot_parameters_and_macs(data):
    """ Plots number of parameters and MACs for each model. """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of Parameters', color=color)
    ax1.bar(data['Model'].unique(), data.groupby('Model')['Num Params'].mean(), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('MACs', color=color)
    ax2.plot(data['Model'].unique(), data.groupby('Model')['MACs'].mean(), color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Number of Parameters and MACs per Model')
    plt.show()
    plt.savefig('parameters_and_macs.png')

def plot_time_vs_epoch(data):
    """ Plots training time vs. epochs for each model. """
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Epoch', y='Time', hue='Model', marker='o')
    plt.title('Training Time vs. Epochs for Different Models')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    plt.savefig('time_vs_epoch.png')

def plot_loss_over_relative_time(data):
    """ Plots training loss over relative epoch times for each model. """
    # Normalize 'Epoch Time' relative to the start time of each model
    data['Relative Epoch Time (Hours)'] = data.groupby('Model')['Epoch Time'].transform(lambda x: (x - x.min()) / 3600)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x='Relative Epoch Time (Hours)', y='Training Loss', hue='Model', style='Model', markers=True, dashes=False)
    plt.title('Training Loss Over Relative Time for Each Model')
    plt.xlabel('Relative Epoch Time (Hours)')
    plt.ylabel('Training Loss')

    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    plt.savefig('loss_over_relative_time.png')

if __name__ == "__main__":
    data = parse_training_data('output.txt')
    plot_loss_over_relative_time(data)
    plot_loss_over_epoch(data)
    plot_accuracy_vs_epoch(data)
    plot_parameters_and_macs(data)