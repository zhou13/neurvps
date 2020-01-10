#!/usr/bin/env python3
"""Compute vanishing points using corase-to-fine method on the evaluation dataset.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --dump <output-dir>           Optionally, save the vanishing points to npz format.
                                 The coordinate of VPs is in the camera space, see
                                 `to_label` and `to_pixel` in neurvps/models/vanishing_net.py
                                 for more details.
   --noimshow                    Do not show result
"""

import os
import sys
import math
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess

import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from tqdm import tqdm
from docopt import docopt

import neurvps
import neurvps.models.vanishing_net as vn
from neurvps.config import C, M
from neurvps.datasets import Tmm17Dataset, ScanNetDataset, WireframeDataset


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    C.model.im2col_step = 32  # override im2col_step for evaluation
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    if M.backbone == "stacked_hourglass":
        model = neurvps.models.hg(
            planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
        )
    else:
        raise NotImplementedError

    checkpoint = torch.load(args["<checkpoint>"])
    model = neurvps.models.VanishingNet(
        model, C.model.output_stride, C.model.upsample_scale
    )
    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if C.io.dataset.upper() == "WIREFRAME":
        Dataset = WireframeDataset
    elif C.io.dataset.upper() == "TMM17":
        Dataset = Tmm17Dataset
    elif C.io.dataset.upper() == "SCANNET":
        Dataset = ScanNetDataset
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        Dataset(C.io.datadir, split="valid"),
        batch_size=1,
        shuffle=False,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )

    if args["--dump"] is not None:
        os.makedirs(args["--dump"], exist_ok=True)

    err = []
    n = C.io.num_vpts
    for batch_idx, (image, target) in enumerate(tqdm(loader)):
        image = image.to(device)
        input_dict = {"image": image, "test": True}
        vpts_gt = target["vpts"][0]
        vpts_gt *= (vpts_gt[:, 2:3] > 0).float() * 2 - 1
        vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
        input_dict["vpts"] = vpts
        with torch.no_grad():
            score = model(input_dict)[:, -1].cpu().numpy()
        index = np.argsort(-score)
        candidate = [index[0]]
        for i in index[1:]:
            if len(candidate) == n:
                break
            dst = np.min(np.arccos(np.abs(vpts[candidate] @ vpts[i])))
            if dst < np.pi / n:
                continue
            candidate.append(i)
        vpts_pd = vpts[candidate]

        for res in range(1, len(M.multires)):
            vpts = [sample_sphere(vpts_pd[vp], M.multires[-res], 64) for vp in range(n)]
            input_dict["vpts"] = np.vstack(vpts)
            with torch.no_grad():
                score = model(input_dict)[:, -res - 1].cpu().numpy().reshape(n, -1)
            for i, s in enumerate(score):
                vpts_pd[i] = vpts[i][np.argmax(s)]
        for vp in vpts_gt.numpy():
            err.append(min(np.arccos(np.abs(vpts_pd @ vp).clip(max=1))) / np.pi * 180)

        if args["--dump"]:
            np.savez(
                osp.join(args["--dump"], f"{batch_idx:06d}.npz"),
                vpts_pd=vpts_pd,
                vpts_gt=vpts_gt,
            )

    err = np.sort(np.array(err))
    np.savez(args["--output"], err=err)
    y = (1 + np.arange(len(err))) / len(loader) / n

    if not args["--noimshow"]:
        plt.plot(err, y, label="Conic")
        print(" | ".join([f"{AA(err, y, th):.3f}" for th in [0.5, 1, 2, 5, 10, 20]]))
        plt.legend()
        plt.show()


def sample_sphere(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


if __name__ == "__main__":
    main()
