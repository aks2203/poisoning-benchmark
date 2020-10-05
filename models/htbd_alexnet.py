import torch
import torch.nn as nn


class HTBDAlexNet(nn.Module):
    def __init__(self, num_classes=10, feature_size=4096):
        super(HTBDAlexNet, self).__init__()
        self.feature_size = feature_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(feature_size, num_classes)

    def forward(self, x, penu=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if penu:
            return x
        out = self.linear(x)
        return out
