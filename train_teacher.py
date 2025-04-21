import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from models import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path, batch_size):
    transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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

    
def create_model():
    model = ResNet.ResNet34(classes_num=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    return model, criterion, optimizer

def train(trainloader, testloader, epochs, model, optimizer, criterion):
    print('----------------------Training----------------------')
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[30, 60, 90],  # 学习率衰减节点
        gamma=0.5
    )
    for epoch in range(epochs):
        for i, (feature, label) in enumerate(trainloader):
            model.train()
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            print(f"Train Epoch:[{epoch + 1} | {epochs}] | Loss: {loss.data.item()} | Acc: {(predicted == label).sum() / len(predicted)}")
        acc = test(testloader, model)
        scheduler.step()
    torch.save(model.state_dict(), 'TEACHERpretrain_R34.pth')

def test(testloader, model):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (feature, label) in enumerate(testloader):
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
        print(f"Validating ----- Accuracy on the TEST set: {(100 * correct / total)}%")

    return correct / total

if __name__ == '__main__':
    file_path = '/home/data/datasets/cifar10'
    model, criterion, optimizer = create_model()
    trainloader, testloader = load_data(file_path, 256)
    train(trainloader, testloader, 100, model, optimizer, criterion)
