#!/usr/bin/env python3
"""Training and Evaluate the Neural Network
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
    yaml-config                      Path to the yaml hyper-parameter file

Options:
   -h --help                         Show this screen.
   -d --devices <devices>            Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>      Folder name [default: default-identifier]
"""

import os
import sys
import glob
import shlex
import pprint
import random
import shutil
import signal
import os.path as osp
import datetime
import platform
import threading
import subprocess

import yaml
import numpy as np
import torch
from docopt import docopt

import neurvps
import neurvps.models.vanishing_net as vn
from neurvps.config import C, M
from neurvps.datasets import Tmm17Dataset, ScanNetDataset, WireframeDataset


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    # FIXME: not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    num_gpus = args["--devices"].count(",") + 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    batch_size = M.batch_size * num_gpus
    datadir = C.io.datadir
    kwargs = {
        "batch_size": batch_size,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    if C.io.dataset.upper() == "WIREFRAME":
        Dataset = WireframeDataset
    elif C.io.dataset.upper() == "TMM17":
        Dataset = Tmm17Dataset
    elif C.io.dataset.upper() == "SCANNET":
        Dataset = ScanNetDataset
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="train"), shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="valid"), shuffle=False, **kwargs
    )
    epoch_size = len(train_loader)

    if resume_from:
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth.tar"))

    # 2. model
    if M.backbone == "stacked_hourglass":
        model = neurvps.models.hg(
            planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
        )
    else:
        raise NotImplementedError
    model = neurvps.models.VanishingNet(
        model, C.model.output_stride, C.model.upsample_scale
    )

    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr * num_gpus,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=C.optim.lr * num_gpus,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = resume_from or get_outdir(args["--identifier"])
    print("outdir:", outdir)

    try:
        trainer = neurvps.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=batch_size,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
