import math
import random
import os.path as osp
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class benchmark(object):
    def __init__(self, msg, enable=True, fmt="%0.3g"):
        """
        Initialize the logger.

        Args:
            self: (todo): write your description
            msg: (str): write your description
            enable: (bool): write your description
            fmt: (str): write your description
        """
        self.msg = msg
        self.fmt = fmt
        self.enable = enable

    def __enter__(self):
        """
        Starts the timer.

        Args:
            self: (todo): write your description
        """
        if self.enable:
            self.start = timer()
        return self

    def __exit__(self, *args):
        """
        Exit exit.

        Args:
            self: (todo): write your description
        """
        if self.enable:
            t = timer() - self.start
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
            self.time = t


def plot_image_grid(im, title):
    """
    Plot a matplot as an image.

    Args:
        im: (array): write your description
        title: (str): write your description
    """
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(im[i])
        plt.colorbar()
    plt.title(title)


def quiver(x, y, ax):
    """
    Quiver a scatter.

    Args:
        x: (todo): write your description
        y: (todo): write your description
        ax: (todo): write your description
    """
    ax.set_xlim(0, x.shape[1])
    ax.set_ylim(x.shape[0], 0)
    ax.quiver(
        x,
        y,
        units="xy",
        angles="xy",
        scale_units="xy",
        scale=1,
        minlength=0.01,
        width=0.1,
        color="b",
    )


def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def argsort2d(arr):
    """
    Return indices of np.

    Args:
        arr: (array): write your description
    """
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]


def __parallel_handle(f, q_in, q_out):
    """
    Parse a function f from a function f.

    Args:
        f: (str): write your description
        q_in: (dict): write your description
        q_out: (bool): write your description
    """
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    """
    Parmap function.

    Args:
        f: (todo): write your description
        X: (todo): write your description
        nprocs: (todo): write your description
        multiprocessing: (todo): write your description
        cpu_count: (int): write your description
        progress_bar: (todo): write your description
        x: (todo): write your description
        x: (todo): write your description
    """
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(X)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]
