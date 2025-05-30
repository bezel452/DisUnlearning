import torch 
import torch.nn as nn
from torch.nn import functional as F

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += identity
        return F.relu(x, inplace=True)
    
class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += identity
        return F.relu(x, inplace=True)

class ResNet18(nn.Module):
    def __init__(self, classes_num):
        super(ResNet18, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(256, classes_num)
        )
    
    def forward(self, x):
        x = self.prepare(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, classes_num):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        #    nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

'''
model = ResNet34(10)
x = torch.randn(1, 3, 32, 32)
print(model.prepare(x).shape)
'''