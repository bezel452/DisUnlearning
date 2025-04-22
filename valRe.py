import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    batch_size = 256
    transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root=file_path, train=True, download=True, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=file_path, train=False, download=True, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def test(testloader, model):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        losses = []
        for i, (feature, label) in enumerate(testloader):
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, label)
            losses.append(loss.detach())
            total += label.size(0)
            correct += (predicted == label).sum()
        print(f"Validating ----- Accuracy on the TEST set: {(100 * correct / total)}%")
    return torch.stack(losses).mean()

if __name__ == '__main__':
    file_path = '/home/data/datasets/cifar10'
    train_loader, test_loader = load_data(file_path)
    model = resnet18(num_classes = 10).to(device = device)
    model.load_state_dict(torch.load('ValidateR18.pth'))
    test(test_loader, model)
