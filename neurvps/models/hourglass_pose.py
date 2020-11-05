"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Yichao Zhou (VanishingNet)
(c) Yichao Zhou (LCNN)
(c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HourglassNet", "hg"]


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, resample=None):
        """
        Initialize the convolutional layer.

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            planes: (todo): write your description
            stride: (int): write your description
            resample: (int): write your description
        """
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.resample = resample
        self.stride = stride

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.resample is not None:
            residual = self.resample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        """
        Initialize the block.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            num_blocks: (int): write your description
            planes: (todo): write your description
            depth: (float): write your description
        """
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        """
        Generate residuals of residuals.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            num_blocks: (int): write your description
            planes: (str): write your description
        """
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        """
        Generate a list of blocks.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            num_blocks: (int): write your description
            planes: (str): write your description
            depth: (int): write your description
        """
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        """
        Perform a forward computation.

        Args:
            self: (todo): write your description
            n: (todo): write your description
            x: (todo): write your description
        """
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        """
        Forward forward forward forward forward forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    def __init__(self, planes, block, head, depth, num_stacks, num_blocks):
        """
        Initialize the layers.

        Args:
            self: (todo): write your description
            planes: (todo): write your description
            block: (todo): write your description
            head: (todo): write your description
            depth: (float): write your description
            num_stacks: (int): write your description
            num_blocks: (int): write your description
        """
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, planes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(planes, ch, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        """
        Make residual residuals.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            planes: (str): write your description
            blocks: (todo): write your description
            stride: (int): write your description
        """
        resample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            resample = nn.Conv2d(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=stride
            )
        layers = [block(self.inplanes, planes, stride, resample)]
        self.inplanes = planes * block.expansion
        for i in range(blocks - 1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        """
        Make a convolution.

        Args:
            self: (todo): write your description
            inplanes: (bool): write your description
            outplanes: (todo): write your description
        """
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out[::-1]


def hg(**kwargs):
    """
    Create a hg2d model.

    Args:
    """
    model = HourglassNet(
        planes=kwargs["planes"],
        block=Bottleneck2D,
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1)),
        depth=kwargs["depth"],
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
    )
    return model


def main():
    """
    Main function.

    Args:
    """
    hg(depth=2, num_stacks=1, num_blocks=1)


if __name__ == "__main__":
    main()
