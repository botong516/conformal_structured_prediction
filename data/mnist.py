
import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST(root='./', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    return trainloader, testloader

def train_model(trainloader):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1):
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs + torch.randn(inputs.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch: %d, loss: %.3f" % (epoch, running_loss / 50000))
    return net

def evaluate_model(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images + torch.randn(images.shape)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("MNIST accuracy: %d %%" % (100 * correct / total))

def create_datafile(testloader, net):
    directory = "../collected/mnist"
    datafile = os.path.join(directory, "mnist.csv")
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = open(datafile, 'w')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images + torch.randn(images.shape)
            outputs = F.softmax(net(images), dim=1)
            for preds, label in zip(outputs.data, labels):
                f.write(','.join([str(label.numpy())] + [str(p) for p in preds.numpy()]) + '\n')
    f.close()
    print(f"Saved MNIST data to {datafile}.")

if __name__ == '__main__':
    trainloader, testloader = load_data()
    net = train_model(trainloader)
    evaluate_model(testloader, net)
    create_datafile(testloader, net)
