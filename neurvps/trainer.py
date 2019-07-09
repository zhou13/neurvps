import os
import atexit
import random
import shutil
import signal
import os.path as osp
import threading
import subprocess
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

import neurvps.utils as utils
from neurvps.config import C, M


class Trainer(object):
    def __init__(
        self, device, model, optimizer, train_loader, val_loader, batch_size, out
    ):
        self.device = device

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        # self.run_tensorboard(board_out)
        # time.sleep(1)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def run_tensorboard(self, board_out):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen(
            ["tensorboard", f"--logdir={board_out}", f"--port={C.io.tensorboard_port}"]
        )

        def killme():
            os.kill(p.pid, signal.SIGTERM)

        atexit.register(killme)

    def _loss(self, result):
        losses = result["losses"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * self.batch_size:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        c = M.smp_rnd + C.io.num_vpts * len(M.multires) * (M.smp_pos + M.smp_neg)
        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                image = image.to(self.device)
                input_dict = {"image": image, "vpts": target["vpts"], "eval": True}
                result = self.model(input_dict)
                total_loss += self._loss(result)
                # permute output to be (batch x (nneg + npos) x 2)
                preds = result["preds"]
                vpts = preds["vpts"].reshape(-1, c, 2).cpu().numpy()
                scores = preds["scores"].reshape(-1, c, len(M.multires)).cpu().numpy()
                ys = preds["ys"].reshape(-1, c, len(M.multires)).cpu().numpy()
                for i in range(self.batch_size):
                    index = batch_idx * self.batch_size + i
                    np.savez(
                        f"{npz}/{index:06}.npz",
                        **{k: v[i].cpu().numpy() for k, v in preds.items()},
                    )
                    if index >= 8:
                        continue
                    self.plot(index, image[i], vpts[i], scores[i], ys[i], f"{viz}/{index:06}")

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self.mean_loss = total_loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth.tar"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth.tar"),
            osp.join(npz, "checkpoint.pth.tar"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth.tar"),
                osp.join(self.out, "checkpoint_best.pth.tar"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        time = timer()
        for batch_idx, (image, target) in enumerate(self.train_loader):
            self.optim.zero_grad()
            self.metrics[...] = 0

            image = image.to(self.device)
            input_dict = {"image": image, "vpts": target["vpts"], "eval": False}
            result = self.model(input_dict)

            loss = self._loss(result)
            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self._write_metrics(1, loss.item(), "training", do_print=False)

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            num_images = self.batch_size * self.iteration
            if (
                num_images % C.io.validation_interval == 0
                or num_images == C.io.validation_debug
            ):
                self.validate()
                time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )
        return total_loss

    def plot(self, index, image, vpts, scores, ys, prefix):
        for idx, (vp, score, y) in enumerate(zip(vpts, scores, ys)):
            plt.imshow(image[0].cpu().numpy())
            color = (random.random(), random.random(), random.random())
            plt.scatter(vp[1], vp[0])
            plt.text(
                vp[1] - 20,
                vp[0] - 10,
                " ".join(map("{:.3f}".format, score))
                + "\n"
                + " ".join(map("{:.3f}".format, y)),
                bbox=dict(facecolor=color),
                fontsize=12,
            )
            for xy in np.linspace(0, 512, 10):
                plt.plot(
                    [vp[1], xy, vp[1], xy, vp[1], 0, vp[1], 511],
                    [vp[0], 0, vp[0], 511, vp[0], xy, vp[0], xy],
                    color=color,
                )
            plt.savefig(f"{prefix}_vpts_{idx}.jpg"), plt.close()

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()

    def move(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            for name in obj:
                if isinstance(obj[name], torch.Tensor):
                    obj[name] = obj[name].to(self.device)
            return obj
        assert False


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(sm, fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def _launch_tensorboard(board_out, port, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])

    def kill():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(kill)
