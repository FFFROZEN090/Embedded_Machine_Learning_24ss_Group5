import matplotlib.pyplot as plt
import re
import argparse

def plot_test_accuracy(prefix1, prefix2, use_epochs=False):
    filename1 = file_name = "logs/datalog_" + prefix1 + ".txt"
    filename2 = file_name = "logs/datalog_" + prefix2 + ".txt"
    # Read the content of the file
    with open(filename1, 'r') as file:
        text1 = file.read()
    
    with open(filename2, 'r') as file:
        text2 = file.read()

    # Define patterns to capture entire lines containing "Train Info" or "Test Info"
    train_pattern = r'^Train Info:.*$'
    test_pattern = r'^Test Info:.*$'

    # Extract epochs, times, and accuracies using regex
    accuracy_pattern = r"'accuracy': ([0-9]+\.[0-9]+)"
    epoch_pattern = r"'epoch': ([0-9]+)"
    time_pattern = r"'time': ([0-9]+\.[0-9]+)"
    start_time_pattern = r"Start Time: ([0-9]+\.[0-9]+)"

    # Process each line in the data
    for line in text1.splitlines():
        # Check for training information lines
        if re.match(train_pattern, line):
            train_line = line
        # Check for testing information lines
        if re.match(test_pattern, line):
            test_line = line
    
    
    start_time1 = float(re.search(start_time_pattern, text1).group(1))
    accuracies1 = [float(match) for match in re.findall(accuracy_pattern, test_line)]
    epochs1 = [int(match) for match in re.findall(epoch_pattern, test_line)]
    times1 = [float(match) - start_time1 for match in re.findall(time_pattern, test_line)]

    for line in text2.splitlines():
        # Check for training information lines
        if re.match(train_pattern, line):
            train_line = line
        # Check for testing information lines
        if re.match(test_pattern, line):
            test_line = line

    start_time2 = float(re.search(start_time_pattern, text2).group(1))
    accuracies2 = [float(match) for match in re.findall(accuracy_pattern, test_line)]
    epochs2 = [int(match) for match in re.findall(epoch_pattern, test_line)]
    times2 = [float(match) - start_time2 for match in re.findall(time_pattern, test_line)]
    
    # Plot accuracy over time consumed
    fig, ax = plt.subplots()
    if use_epochs:
        ax.plot(epochs1, accuracies1, marker='o')
        ax.plot(epochs2, accuracies2, marker='x')
        ax.legend([prefix1, prefix2])
        ax.set_xlabel('Epochs')
    else:
        ax.plot(times1, accuracies1, marker='o')
        ax.plot(times2, accuracies2, marker='x')
        ax.legend([prefix1, prefix2])
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    plt.show()

    # Save the plot to a file
    plt.savefig("figures/" + prefix1 + "_" + prefix2 + ".png")
    if use_epochs:
        plt.savefig("figures/" + prefix1 + "_" + prefix2 + "_epochs.png")

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Plot test accuracy')
    parser.add_argument('--prefix1', type=str, help='Prefix of the first log file')
    parser.add_argument('--prefix2', type=str, help='Prefix of the second log file')
    parser.add_argument('--use_epochs', action='store_true', help='Use epochs instead of time')

    # Parse the arguments
    args = parser.parse_args()
    prefix1 = args.prefix1
    prefix2 = args.prefix2
    use_epochs = args.use_epochs

    plot_test_accuracy(prefix1, prefix2, use_epochs)

if __name__ == '__main__':
    main()