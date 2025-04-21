import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(100)

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

def train(train_loader):
    model = resnet18(num_classes = 10).to(device = device)
    epochs = 40
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader
