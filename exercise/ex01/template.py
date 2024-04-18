
# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.show()

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    result = torch.sin(2 * np.pi * x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model()
plt.savefig('Initial_data.png')
plt.clf()


# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data)
    error = 0.5 * torch.mean((y_pred - y_data) ** 2)
    return error

model_degree = 3

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model)
plt.savefig('Initial_fit.png')
plt.clf()

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size

# Create a new plot for an overfitted Polynomial of 11-th degree.
model_degree = 11
model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print("Overfitted model with degree 11")
print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model)
plt.savefig('Overfitted_fit_11.png')
plt.clf()




# Define Root Mean Squared Error function
def rmse(model, x_data, y_data):
    y_pred = model(x_data)
    rmse = torch.sqrt(torch.mean((y_pred - y_data) ** 2))
    return rmse

# Define a function that plot Polynomial from degree 0 to 11 in a single plot with rms error, the plot should have 12 subplots with 6 rows and 2 columns
def plot_polynomials(x_train, y_train, x_test, y_test):
    fig, axs = plt.subplots(6, 2, figsize=(10, 20))
    axs = axs.ravel()
    for i in range(12):
        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=i)
        train_err = rmse(model, x_train, y_train)
        test_err = rmse(model, x_test, y_test)
        # plot ground truth
        axs[i].plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
        axs[i].plot(x_train, y_train, 'ob', label='Train data')
        axs[i].plot(x_test, y_test, 'xr', label='Test data')
        axs[i].plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')
        axs[i].set_title(f"Degree: {model.degree()}")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].legend()
        axs[i].text(0.5, 0.5, f"Train RMSE: {train_err:.2f}\nTest RMSE: {test_err:.2f}", horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

    plt.tight_layout()

plot_polynomials(x_train, y_train, x_test, y_test)
plt.savefig('RMS_plot.png')
plt.clf()

# Vary with the size of the data

# Starting with 10-th degree polynomial
""" 
Change the sample size of the data until the RMSE difference between the train and test data is less than 0.0001.
Parameters:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    model_degree: int
    noise_amplitude: float
    n_samples: int
    step_size: int
    
Returns:
    steps: int
    train_err: float
    test_err: float
    n_samples: int
"""
def vary_data_size(x_train, y_train, x_test, y_test, model_degree, noise_amplitude, n_samples, step_size):
    steps = 0
    train_err = []
    test_err = []
    while True:
        steps += 1
        x_train = torch.linspace(0, 1, n_samples)
        y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))
        x_test = torch.linspace(0, 1, n_samples)
        y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
        train_err.append(rmse(model, x_train, y_train))
        test_err.append(rmse(model, x_test, y_test))
        if abs(train_err[-1] - test_err[-1]) < 0.0001:
            break
        n_samples += step_size

    return steps, train_err, test_err, n_samples
   

steps, train_err, test_err, n_samples = vary_data_size(x_train, y_train, x_test, y_test, 10, 0.15, 11, 1)

# Plot the Train RMSE and Test RMSE and also shows the difference between them
def plot_rmse_diff(train_err, test_err, steps):
    plt.plot(range(steps), train_err, label='Train RMSE')
    plt.plot(range(steps), test_err, label='Test RMSE')
    plt.fill_between(range(steps), train_err, test_err, color='gray', alpha=0.5, label='Difference')
    plt.xlabel("Steps")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

plot_rmse_diff(train_err, test_err, steps)
plt.savefig('RMSE_diff.png')
plt.clf()


