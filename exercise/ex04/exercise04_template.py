from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn
import matplotlib.pyplot as plt

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
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss

def test(model, device, test_loader, epoch):
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

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def plot_loss(train_loss_list, test_loss_list, add_info):
    epochs = range(1, len(train_loss_list) + 1)
    plt.plot(epochs, train_loss_list, marker='o', label='train loss')
    plt.plot(epochs, test_loss_list, marker='o', label='test loss')
    plt.legend()

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig('loss_{}.png'.format(add_info))
    plt.clf()


def plot_acc(test_acc_list, info):
    epochs = range(1, len(test_acc_list) + 1)
    plt.plot(epochs, test_acc_list, marker='o', label='test accuracy')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('acc_{}.png'.format(info))
    plt.clf()

def plot_dropout(dropout_test_acc):
    for dropout, acc_list in dropout_test_acc.items():
      epochs = range(1, len(acc_list) + 1)
      plt.plot(epochs, acc_list, marker='o', label=dropout)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('acc_different_dropout.png')
    plt.clf()

def plot_trans_acc(trans_test):
    for trans, acc_list in trans_test.items():
      epochs = range(1, len(acc_list) + 1)
      plt.plot(epochs, acc_list, marker='o', label=trans)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('acc_different_transformer.png')
    plt.clf()

def plot_trans_time(trans_test,time_list):
    for trans, acc_list in trans_test.items():
      time = time_list[trans]
      plt.plot(time, acc_list, marker='o', label=trans)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("accuracy")
    plt.savefig('time_different_transformer.png')
    plt.clf()

def plot_histogram(weights):
    num = len(weights)
    fig, axs = plt.subplots(num, 1, figsize=(10, num * 5), sharex=True)
    if num == 1:
        axs = [axs]
    for i, (key, ws) in enumerate(weights.items()):
        ws = ws.cpu().detach().numpy()
        ws = ws.flatten()
        axs[i].hist(ws, bins=20, density=True, alpha=0.75, color='g')
        axs[i].set_title(f'L2 value: {key}')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(True)
    axs[-1].set_xlabel('Weight values')
    fig.suptitle('Histogram of weights of last conv layer')
    plt.tight_layout()
    plt.savefig('weight_histogram.png')
    plt.clf()


def plot_l2(test_acc):
    for value, acc_list in test_acc.items():
        epochs = range(1, len(acc_list) + 1)
        plt.plot(epochs, acc_list, marker='o', label=value)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('l2_different_accuracy.png')
    plt.clf()

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
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = [transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms)

    dataset_train = datasets.SVHN('../data', split='train', download=True,
                       transform=train_transforms)
    dataset_test = datasets.SVHN('../data', split='test', download=True,
                       transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f'Starting training at: {time.time():.4f}')

    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    test_time_list = []
    start = time.time()

    #train the model with 30 epochs and plot the train loss, test loss and test accuracy
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer,epoch)
        test_loss, acc = test(model, device, test_loader, epoch)
        cur_time = time.time()
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(acc)
        test_time_list.append(int(cur_time - start))
    plot_loss(train_loss_list, test_loss_list, 1)
    plot_acc(test_acc_list,2)

    #different dropout
    dropout_test_acc = {}
    for dropout in [0.0,0.7,0.8,0.9]:
        model = VGG11(dropout_p=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        test_acc_list=[]
        for epoch in range(1, args.epochs + 1):
          _ = train(args, model, device, train_loader, optimizer,epoch)
          _, acc = test(model, device, test_loader, epoch)
          test_acc_list.append(acc)
        dropout_test_acc[ str(dropout)] = test_acc_list
    plot_dropout(dropout_test_acc)

    #L2 regularization
    l2_test_acc = {}
    weight_conv = {}
    for l2_value in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = VGG11(dropout_p=args.dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=l2_value)
        test_acc_list=[]
        for epoch in range(1, args.epochs + 1):
          _ = train(args, model, device, train_loader, optimizer,epoch)
          _, acc = test(model, device, test_loader, epoch)
          test_acc_list.append(acc)
        l2_test_acc[ str(l2_value)] = test_acc_list
        weight_conv[str(l2_value)] = model.layers[18].weight
    plot_histogram(weight_conv)
    plot_l2(l2_test_acc)
    
    # data augmentation--different transformer
    trans_list=['original','horizontal_flip','perspective','random_crop']
    transforms_list = [
        [transforms.ToTensor()],
        [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()],
        [transforms.RandomPerspective(distortion_scale=0.5, p=0.5),transforms.ToTensor()],
        [transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)), transforms.ToTensor()]
    ]
    trans_acc={}
    trans_time={}
    for i, trans in enumerate(transforms_list):
        test_transforms = transforms.Compose(trans)
        train_transforms = [trans]
        train_transforms = transforms.Compose(train_transforms)

        dataset_train = datasets.SVHN('../data', split='train', download=False,
                           transform=train_transforms)
        dataset_test = datasets.SVHN('../data', split='test', download=False,
                           transform=test_transforms)
        train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        model = VGG11(dropout_p=args.dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        test_list=[]
        time_list=[]
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            _ = train(args, model, device, train_loader, optimizer,epoch)
            test_loss, acc = test(model, device, test_loader, epoch)
            cur_time = time.time()
            test_list.append(acc)
            time_list.append(int(cur_time - start))
        trans_acc[trans_list[i]] = test_list
        trans_time[trans_list[i]] = time_list
    plot_trans_acc(trans_acc)
    plot_trans_time(trans_acc,trans_time)





    if (args.L2_reg is not None):
        f_name = f'trained_VGG11_L2-{args.L2_reg}.pt'
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')


if __name__ == '__main__':
    main()
