from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# Import Time
import time

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Take 28x28 images as input and output 512 features
        self.linear0 = nn.Linear(28*28, 512)
        # ReLU activation function
        self.sigmoid0 = nn.Sigmoid()
        # Output layer: Take 512 features as input and output 128 classes
        self.linear1 = nn.Linear(512, 128)
        # ReLU activation function
        self.sigmoid1 = nn.Sigmoid()
        # Output layer: Take 128 features as input and output 10 classes
        self.linear2 = nn.Linear(128, 10)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear0(x)
        x = self.sigmoid0(x)
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

class MLP_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        # Take 32x32x3 images as input and output 512 features
        self.linear0 = nn.Linear(32*32*3, 512)
        # ReLU activation function
        self.sigmoid0 = nn.Sigmoid()
        # Output layer: Take 512 features as input and output 128 classes
        self.linear1 = nn.Linear(512, 128)
        # ReLU activation function
        self.sigmoid1 = nn.Sigmoid()
        # Output layer: Take 128 features as input and output 10 classes
        self.linear2 = nn.Linear(128, 10)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear0(x)
        x = self.sigmoid0(x)
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

""" 
Define CNN as Follow Structure:
Layer 1: Convolution: Input channels: 3, Output channels: 32, Kernel size: 3, Stride: 1
Layer 2: Convolution: Input channels: 32, Output channels: 64, Kernel size: 3, Stride: 2
Layer 3: Convolution: Input channels: 64, Output channels: 128, Kernel size: 3, Stride: 1
Layer 4: Flatten layer
Layer 5: Linear: Input size: 18432, Output channels: 128
Layer 6: Linear: Input size: 128, Output channels: 10 Linear(128, 10)

Activation function after computational layers: ReLU
Output activation function: LogSoftmax

"""
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, 1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 64, 3, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(18432, 128)
        self.relu3 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear0(x)
        x = self.relu3(x)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=1)
        return x



def train(args, model, device, train_loader, optimizer, epoch, train_info):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    average_train_loss = loss.item() / data.shape[0]

    # Set Train info
    train_info_instance = {'epoch': epoch, 'time': time.time(), 'loss': average_train_loss}
    train_info.append(train_info_instance)

def test(model, device, test_loader, epoch, test_info):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Set Test info
    test_info_instance = {'epoch': epoch, 'time': time.time(), 'loss': test_loss, 'accuracy': 100. * correct / len(test_loader.dataset)}
    test_info.append(test_info_instance)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # Custom arguments
    # Logfile prefix
    parser.add_argument('--prefix', type=str, default='0', metavar='P',
                        help='Logfile prefix to write the results to')
    # Optimizer Choice
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                        help='Optimizer to use (default: SGD)')
    
    # Dataset Choice Between MNIST and CIFAR10
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                        help='Dataset to use (default: MNIST)')
    
    # Model Choice Between MLP and CNN
    parser.add_argument('--model', type=str, default='MLP', metavar='M',
                        help='Model to use (default: MLP)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    train_info = []
    test_info = []

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Get Dataset
    if args.dataset == 'MNIST':
        dataset_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset_test = datasets.MNIST('../data', train=False,
                        transform=transform)
    elif args.dataset == 'CIFAR10':
        # Change the transform for CIFAR10
        transform=transforms.Compose([
            transforms.ToTensor()
            ])
        dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
        dataset_test = datasets.CIFAR10('../data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    # Choose Model
    if args.model == 'MLP' and args.dataset == 'MNIST':
        model = MLP().to(device)
    elif args.model == 'MLP' and args.dataset == 'CIFAR10':
        model = MLP_CIFAR10().to(device)
    elif args.model == 'CNN':
        model = CNN().to(device)

    # Choose Optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Start Timer
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_info)
        test(model, device, test_loader, epoch, test_info)

    # End Timer
    end = time.time()

    # Save infos to txt file

    # Define file name
    file_name = "logs/datalog_" + args.prefix + ".txt"

    # Check and Open file with append mode
    with open(file_name, 'a') as file:
        # Write a plit line
        file.write('--------------------------------------\n')
        # Write arguments
        file.write('Arguments: ' + str(args) + '\n')
        # Write the start and end time
        file.write('Start Time: ' + str(start) + '\n')
        file.write('End Time: ' + str(end) + '\n')
        # Write the train info
        file.write('Train Info: ' + str(train_info) + '\n')
        # Write the test info
        file.write('Test Info: ' + str(test_info) + '\n')
        # Write a plit line
        file.write('--------------------------------------\n')
        # Write a new line
        file.write('\n\n')




if __name__ == '__main__':
    main()
