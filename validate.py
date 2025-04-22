import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from models import ResNet
from torchvision.models import resnet18, resnet34

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

def test(model, testloader, Name):
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
        print(f"Validating ----- Model: {Name} Accuracy on the TEST set: {(100 * correct / total)}%")

if __name__ == '__main__':
    file_path = '/home/data/datasets/cifar10'
    _, testloader = load_data(file_path, batch_size=256)
    t1 = ResNet.ResNet34(classes_num=10).to(device)
    t2 = resnet34(num_classes = 10).to(device = device)
    s1 = ResNet.ResNet18(classes_num=10).to(device=device)
    s2 = resnet18(num_classes = 10).to(device = device)
    t1.load_state_dict(torch.load('TEACHERpretrain_R34.pth'))
    t2.load_state_dict(torch.load('TEACHERpretrain_TV_R34.pth'))
    s1.load_state_dict(torch.load('STUDENT_Res18.pth'))
    s2.load_state_dict(torch.load('STUDENT_R18VAL.pth'))

    test(t1, testloader, 'Teacher1')
    test(s1, testloader, "Student1")
    test(t2, testloader, 'Teacher2')
    test(s2, testloader, "Student2")
