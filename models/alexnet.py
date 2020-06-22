import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, feature_size=64):
        super(AlexNet, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.lrn1 = nn.LocalResponseNorm(9, k=1.0, alpha=0.001, beta=0.75)
        self.conv2 = nn.Conv2d(64, feature_size, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(9, k=1.0, alpha=0.001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.linear1 = nn.Linear(feature_size * 8 * 8, 384)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(384, 192)
        self.relu4 = nn.ReLU()
        self.linear = nn.Linear(192, num_classes)

    def forward(self, x):
        feats = self.maxpool2(
            self.lrn2(
                self.relu2(
                    self.conv2(self.lrn1(self.maxpool1(self.relu1(self.conv1(x)))))
                )
            )
        )
        feats_vec = feats.view(feats.size(0), self.feature_size * 8 * 8)
        out = self.linear(self.relu4(self.linear2(self.relu3(self.linear1(feats_vec)))))
        return out

    def penultimate(self, x):
        feats = self.maxpool2(
            self.lrn2(
                self.relu2(
                    self.conv2(self.lrn1(self.maxpool1(self.relu1(self.conv1(x)))))
                )
            )
        )
        feats_vec = feats.view(feats.size(0), self.feature_size * 8 * 8)
        out = self.relu4(self.linear2(self.relu3(self.linear1(feats_vec))))
        return out
