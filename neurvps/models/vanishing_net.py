import sys
import math
import random
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as LA
import matplotlib.pyplot as plt
import torch.nn.functional as F

from neurvps.utils import plot_image_grid
from neurvps.config import C, M
from neurvps.models.conic import ConicConv


class VanishingNet(nn.Module):
    def __init__(self, backbone, output_stride=4, upsample_scale=1):
        super().__init__()
        self.backbone = backbone
        self.anet = ApolloniusNet(output_stride, upsample_scale)
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_dict):
        x = self.backbone(input_dict["image"])[0]
        N, _, H, W = x.shape
        test = input_dict.get("test", False)
        if test:
            c = len(input_dict["vpts"])
        else:
            c = M.smp_rnd + C.io.num_vpts * len(M.multires) * (M.smp_pos + M.smp_neg)
        x = x[:, None].repeat(1, c, 1, 1, 1).reshape(N * c, _, H, W)

        if test:
            vpts = [to_pixel(v) for v in input_dict["vpts"]]
            vpts = torch.tensor(vpts, device=x.device)
            return self.anet(x, vpts).sigmoid()

        vpts_gt = input_dict["vpts"].cpu().numpy()
        vpts, y = [], []
        for n in range(N):

            def add_sample(p):
                vpts.append(to_pixel(p))
                y.append(to_label(p, vpts_gt[n]))

            for vgt in vpts_gt[n]:
                for st, ed in zip([0] + M.multires[:-1], M.multires):
                    # positive samples
                    for _ in range(M.smp_pos):
                        add_sample(sample_sphere(vgt, st, ed))
                    # negative samples
                    for _ in range(M.smp_neg):
                        add_sample(sample_sphere(vgt, ed, ed * M.smp_multiplier))
            # random samples
            for _ in range(M.smp_rnd):
                add_sample(sample_sphere(np.array([0, 0, 1]), 0, math.pi / 2))

        y = torch.tensor(y, device=x.device, dtype=torch.float)
        vpts = torch.tensor(vpts, device=x.device)

        x = self.anet(x, vpts)
        L = self.loss(x, y)
        maskn = (y == 0).float()
        maskp = (y == 1).float()
        losses = {}
        for i in range(len(M.multires)):
            assert maskn[:, i].sum().item() != 0
            assert maskp[:, i].sum().item() != 0
            losses[f"lneg{i}"] = (L[:, i] * maskn[:, i]).sum() / maskn[:, i].sum()
            losses[f"lpos{i}"] = (L[:, i] * maskp[:, i]).sum() / maskp[:, i].sum()

        return {
            "losses": [losses],
            "preds": {"vpts": vpts, "scores": x.sigmoid(), "ys": y},
        }


class ApolloniusNet(nn.Module):
    def __init__(self, output_stride, upsample_scale):
        super().__init__()
        self.fc0 = nn.Conv2d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        if M.conic_6x:
            self.bn00 = nn.BatchNorm2d(32)
            self.conv00 = ConicConv(32, 32)
            self.bn0 = nn.BatchNorm2d(32)
            self.conv0 = ConicConv(32, 32)

        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = ConicConv(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = ConicConv(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = ConicConv(128, 256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = ConicConv(256, 256)

        self.fc1 = nn.Linear(16384, M.fc_channel)
        self.fc2 = nn.Linear(M.fc_channel, M.fc_channel)
        self.fc3 = nn.Linear(M.fc_channel, len(M.multires))

        self.upsample_scale = upsample_scale
        self.stride = output_stride / upsample_scale

    def forward(self, input, vpts):
        # for now we did not do interpolation
        if self.upsample_scale != 1:
            input = F.interpolate(input, scale_factor=self.upsample_scale)
        x = self.fc0(input)

        if M.conic_6x:
            x = self.bn00(x)
            x = self.relu(x)
            x = self.conv00(x, vpts / self.stride - 0.5)
            x = self.bn0(x)
            x = self.relu(x)
            x = self.conv0(x, vpts / self.stride - 0.5)

        # 128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x, vpts / self.stride - 0.5)
        x = self.pool(x)
        # 64
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        # 32
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        # 16
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4(x, vpts / self.stride / 8 - 0.5)
        x = self.pool(x)
        # 8
        x = x.view(x.shape[0], -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


def sample_sphere(v, theta0, theta1):
    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def to_label(w, vpts):
    degree = np.min(np.arccos(np.abs(vpts @ w).clip(max=1)))
    return [int(degree < res + 1e-6) for res in M.multires]


def to_pixel(w):
    x = w[0] / w[2] * C.io.focal_length * 256 + 256
    y = -w[1] / w[2] * C.io.focal_length * 256 + 256
    return y, x
