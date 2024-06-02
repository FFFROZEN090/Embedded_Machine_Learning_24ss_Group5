from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn
from typing import Any, Callable, List, Optional, Type, Union
import os

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity,
    ) -> None:
        super().__init__()
        # TODO: Implement the basic residual block!
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the basic residual block!
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity):
        super().__init__()
        # Initial normalization layer
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.block1_1 = BasicBlock(32, 32, 1, self.norm_layer)
        self.block1_2 = BasicBlock(32, 32, 1, self.norm_layer)
        self.block1_3 = BasicBlock(32, 32, 1, self.norm_layer)
        self.block2_1 = BasicBlock(32, 64, 2, self.norm_layer)
        self.block2_2 = BasicBlock(64, 64, 1, self.norm_layer)
        self.block2_3 = BasicBlock(64, 64, 1, self.norm_layer)
        self.block3_1 = BasicBlock(64, 128, 2, self.norm_layer)
        self.block3_2 = BasicBlock(128, 128, 1, self.norm_layer)
        self.block3_3 = BasicBlock(128, 128, 1, self.norm_layer)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = F.relu(x)
        x = torch.sum(x, [2,3])
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class ConvNet(nn.Module):
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
    
class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
        layers = [        
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Flatten(1),
        nn.Linear(512 ,4096),
        nn.Dropout(dropout_p),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.Dropout(dropout_p),
        nn.ReLU(),
        nn.Linear(4096, 10)
        ]

        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output
    



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            cur_time = time.time()
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                cur_time,
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/data.shape[0] ))
            
            # Return the training information
            return cur_time, epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()/data.shape[0]
            

def test(model, device, test_loader, epoch, file_name='output.txt'):
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
    cur_time = time.time()

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        cur_time,
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Return the test information
    return cur_time, epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    
"""
Output all arguments of the main function to a text file.
The format of the output should be:
--------------------
arg1: val1
arg2: val2
...
--------------------
Input: 
    args: argparse.ArgumentParser
    file_name: str
Output:
    None
"""
def output_args(args, file_name):
    # Check if the file exists, if exists, open it with append mode
    # Otherwise, open it with write mode
    if os.path.exists(file_name):
        f = open(file_name, 'a')
    else:
        f = open(file_name, 'w')
    # Write all arguments to the file
    f.write('--------------------\n')
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')
    f.write('--------------------\n')
    f.close()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_file', type=str, default='output.txt', metavar='S',
                        help='output file name')
    # Choose different model
    parser.add_argument('--model', type=str, default='ResNet', metavar='S',
                        help='model name')
    
    args = parser.parse_args()

    # Check if the output file exists, if exists, use append mode
    # Otherwise, use write mode
    if os.path.exists(args.output_file):
        f = open(args.output_file, 'a')
    else:
        f = open(args.output_file, 'w')

    # Mark the start of the output
    f.write('####################\n')

    # Close the file
    f.close()

    # Output all arguments of the main function to a text file
    output_args(args, args.output_file)


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print(f'Curr trafos: ', transform)


    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True,
                    transform=transform)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True,
                    transform=transform)
    
    # Use dataloader to load the dataset
    dataset_train = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    dataset_test = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    # Specify the ResNet model
    if args.model == 'ResNet':
        norm_layer = nn.Identity
        model = ResNet(norm_layer=norm_layer)
    elif args.model == 'VGG11':
        model = VGG11()
    elif args.model == 'ConvNet':
        model = ConvNet()
    else:
        exit('Model not found')
    model = model.to(device)

    if args.L2_reg is not None:
        L2_reg = args.L2_reg
    else:
        L2_reg = 0.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=L2_reg)

    start_time = time.time()
    print(f'Starting training at: {start_time:.4f}')
    train_info = []
    for epoch in range(1, args.epochs + 1):
        train_info.append(train(args, model, device, dataset_train, optimizer, epoch))
    end_time = time.time()
    print(f'Finished training at: {end_time:.4f}') 

    # Output the training information to a file
    f = open(args.output_file, 'a')
    f.write('--------------------\n')
    f.write('TRAINING INFORMATION\n')
    f.write(f'Starting training at: {start_time:.4f}\n')
    f.write(f'Finished training at: {end_time:.4f}\n')
    for info in train_info:
        f.write(f'Current time: {info[0]:.4f}; Train Epoch: {info[1]} [{info[2]}/{info[3]} ({info[4]:.0f}%)]\tLoss: {info[5]}\n')
    f.write('--------------------\n')

    test_info = test(model, device, dataset_test, args.epochs)
    f.write('--------------------\n')
    f.write('TEST INFORMATION\n')
    f.write(f'Current time: {test_info[0]:.4f}; Test Epoch: {test_info[1]}, Test set: Average loss: {test_info[2]}, Accuracy: {test_info[3]}/{test_info[4]} ({test_info[5]:.4f}%)\n')
    f.write('--------------------\n')

    # Calculate the number of parameters in the model and MAC operations
    num_params = sum(p.numel() for p in model.parameters())
    mac = 0
    for info in train_info:
        mac += info[3] * num_params
    mac += test_info[4] * num_params
    f.write('--------------------\n')
    f.write('MODEL INFORMATION\n')
    f.write(f'Number of parameters: {num_params}\n')
    f.write(f'Number of MAC operations: {mac}\n')
    f.write('--------------------\n')

    # Mark the end of the output
    f = open(args.output_file, 'a')
    f.write('####################\n')

if __name__ == '__main__':
    main()
