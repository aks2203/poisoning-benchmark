"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        in_planes,
        out_planes,
        expansion,
        stride,
        train_dp,
        test_dp,
        droplayer=0,
        bdp=0,
    ):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.train_dp = train_dp
        self.test_dp = test_dp

        self.droplayer = droplayer
        self.bdp = bdp

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if self.stride == 1 and action == 1:
            # if stride is not 1, then there is no skip connection. so we keep this layer unchanged
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            if self.test_dp > 0 or (self.training and self.train_dp > 0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)

            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = (
                    torch.bernoulli(
                        self.bdp
                        * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(
                            out.device
                        )
                    )
                    / self.bdp
                )
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(
            in_planes=32,
            train_dp=train_dp,
            test_dp=test_dp,
            droplayer=droplayer,
            bdp=bdp,
        )
        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.test_dp = test_dp
        self.bdp = bdp

    def _make_layers(self, in_planes, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        layers = []

        # get the total number of blocks
        nblks = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            nblks += num_blocks

        dl_step = droplayer / nblks

        blkidx = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                dl = dl_step * blkidx
                blkidx += 1

                layers.append(
                    Block(
                        in_planes,
                        out_planes,
                        expansion,
                        stride,
                        train_dp=train_dp,
                        test_dp=test_dp,
                        droplayer=dl,
                        bdp=bdp,
                    )
                )
                in_planes = out_planes
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.layers:
            layer.test_dp = dp

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        if out.shape[-1] == 2:
            out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, penu=False):
        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if "linear" in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()
