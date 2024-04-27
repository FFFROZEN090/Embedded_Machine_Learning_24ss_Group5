from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

class Linear():
    def __init__(self, in_features: int, out_features: int, batch_size: int, lr=0.001):
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.weight = torch.randn(in_features, out_features) * np.sqrt(1. / in_features)
        self.bias = torch.randn(out_features) * np.sqrt(1. / in_features)
        self.grad_weight = torch.zeros(in_features, out_features)
        self.grad_bias = torch.zeros(out_features)
        self.input = torch.zeros(batch_size, in_features)
        self.test = test

    def forward(self, input):
        self.input = input
        output = input @ self.weight + self.bias
        return output

    def backward(self, grad_output):
        m = self.input.shape[1]
        self.grad_weight = 1/m * (self.input.t() @ grad_output)
        self.grad_bias = 1/m * (torch.sum(grad_output, dim=0))
        grad_input = grad_output @ self.weight.t()
        return grad_input

    def update(self):
        self.weight = self.weight - self.lr * self.grad_weight
        self.bias = self.bias - self.lr * self.grad_bias

class Sigmoid():
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = torch.zeros(batch_size)

    def forward(self, input):
        self.input = input
        output = 1 / (1 + torch.exp(-input))
        return output

    def backward(self, grad_output):
        sigmoid_forward = 1 / (1 + torch.exp(-self.input))
        grad_input = grad_output * sigmoid_forward * (1 - sigmoid_forward)
        return grad_input

def Softmax(input):
    output = torch.exp(input) / torch.sum(torch.exp(input), dim=1).view(-1, 1)
    return output

def compute_loss(target, prediction):
    return -torch.sum(target * torch.log(prediction))/prediction.shape[0]

def compute_gradient(target, prediction):
    return (prediction - target)

class MLP():
    def __init__(self, batch_size, lr):
        super(MLP, self).__init__()
        self.linear0 = Linear(28*28, 512, batch_size, lr)
        self.sigmoid0 = Sigmoid(512, batch_size)
        self.linear1 = Linear(512, 128, batch_size, lr)
        self.sigmoid1 = Sigmoid(128, batch_size)
        self.linear2 = Linear(128, 10, batch_size, lr)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        x = Softmax(x)
        return x

    def backward(self, x):
        x = self.linear2.backward(x)
        x = self.sigmoid1.backward(x)
        x = self.linear1.backward(x)
        x = self.sigmoid0.backward(x)
        x = self.linear0.backward(x)

    def update(self):
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()

def train(args, model, train_loader, epoch, train_losses):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model.forward(data)
        loss = compute_loss(target, output)
        gradient = compute_gradient(target, output)
        model.backward(gradient)
        model.update()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / data.shape[0]))
    average_train_loss = loss.item() / data.shape[0]
    train_losses.append(average_train_loss)


def test(args, model, test_loader, epoch, test_losses, test_accuracies, accuracies):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model.forward(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
       
        target = F.one_hot(target)
        loss = compute_loss(target, output)
        test_loss += loss

    test_loss /= len(test_loader.dataset)

    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_losses.append(test_loss)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(test_accuracy)
    accuracies.append(test_accuracy)

def plot_metrics(train_losses, test_losses, test_accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title("Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('losses.png') 
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title("Accuracy over Epochs(lr = 0.1)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.savefig('accuracy.png')
    plt.show()

def plot_accuracies(accuracies):
    plt.figure(figsize=(8, 6))
    for lr, acc in accuracies.items():
        plt.plot(acc, label=f'LR={lr}')
    plt.legend()
    plt.title("Accuracy over Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.savefig('accuracies.png')
    plt.show()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_losses = []
    test_losses = []
    test_accuracies = []

    learning_rates = [1., 0.5, 0.25, 0.01, 0.001]
    accuracies = {lr: [] for lr in learning_rates}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform,
                       target_transform=torchvision.transforms.Compose([
                                 lambda x:torch.LongTensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x, 10),
                                 lambda x:x.squeeze()]))

    dataset_test = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size = args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size = args.batch_size)

    for lr in learning_rates:
        with torch.no_grad():
            model = MLP(args.batch_size, lr)
            for epoch in range(1, args.epochs + 1):
                train(args, model, train_loader, epoch, train_losses)
                test(args, model, test_loader, epoch, test_losses, test_accuracies, accuracies[lr])

    #plot_metrics(train_losses, test_losses, test_accuracies) 
    plot_accuracies(accuracies) 
if __name__ == '__main__':
    main()
    