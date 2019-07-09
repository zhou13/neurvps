#!/usr/bin/env python
"""Preprocess the SU3 dataset for NeurVPS
Usage:
    dataset/su3.py <dir>
    dataset/su3.py (-h | --help )

Arguments:
    <dir>               Target directory

Options:
   -h --help            Show this screen.
"""
import os
import sys
import json
from glob import glob

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from docopt import docopt
from skimage import io

try:
    sys.path.append(".")
    sys.path.append("..")
    from neurvps.utils import parmap
except Exception:
    raise


def handle(iname):
    prefix = iname.replace(".png", "")
    with open(f"{prefix}_camera.json") as f:
        js = json.load(f)
        RT = np.array(js["modelview_matrix"])

    vpts = []
    # plt.imshow(io.imread(iname))
    for axis in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]:
        vp = RT @ axis
        vp = np.array([vp[0], vp[1], -vp[2]])
        vp /= LA.norm(vp)
        vpts.append(vp)
        # plt.scatter(
        #     vpt[0] / vpt[2] * 2.1875 * 256 + 256,
        #     -vpt[1] / vpt[2] * 2.1875 * 256 + 256
        # )
    # plt.show()
    np.savez_compressed(f"{prefix}_label.npz", vpts=np.array(vpts))


def main():
    args = docopt(__doc__)
    filelist = sorted(glob(args["<dir>"] + "/*/????_0.png"))
    parmap(handle, filelist)


if __name__ == "__main__":
    main()
