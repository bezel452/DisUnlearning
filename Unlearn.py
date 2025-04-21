import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from unlearning import unlearn
from models.ResNet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
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

    testset = torchvision.datasets.CIFAR10(root=file_path, train=False, download=True, transform=transforms_test)
    return trainset, testset

if __name__ == '__main__':
    torch.manual_seed(100)
    file_path = '/home/data/datasets/cifar10'
    batch_size = 256
    train_ds, valid_ds = load_data(file_path)
    
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    classes_to_forget = [0, 2]

    retain_samples, forget_valid_dl, retain_valid_dl = unlearn.data_prepare(classes, train_ds, valid_ds, classes_to_forget, batch_size)
    model = ResNet18(classes_num=10).to(device)
    model.load_state_dict(torch.load("STUDENT_Res18.pth"))

    print("Performance Before Unlearning on Forget Class")
    history = unlearn.validate(model, forget_valid_dl)
    print(f"Accuracy: {history['Acc'] * 100}%")
    print(f"Loss: {history['Loss']}")

    print("Performance Before Unlearning on Retain Class")
    history = unlearn.validate(model, retain_valid_dl)
    print(f"Accuracy: {history['Acc'] * 100}%")
    print(f"Loss: {history['Loss']}")

    noisy_loader, other_samples = unlearn.train_noise(model, retain_samples, classes_to_forget, batch_size)
    unlearn.impair(model, train_ds, noisy_loader)
    unlearn.repair(model, train_ds, other_samples, batch_size)

    print("Performance After Unlearning on Forget Class")
    history = unlearn.validate(model, forget_valid_dl)
    print(f"Accuracy: {history['Acc'] * 100}%")
    print(f"Loss: {history['Loss']}")

    print("Performance After Unlearning on Retain Class")
    history = unlearn.validate(model, retain_valid_dl)
    print(f"Accuracy: {history['Acc'] * 100}%")
    print(f"Loss: {history['Loss']}")

    torch.save(model.state_dict(), "STUDENT_Res18_forget.pth")