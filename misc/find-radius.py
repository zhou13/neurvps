#!/usr/bin/env python3

import math

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def gold_spiral_sampling_patch(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.arange(num_pts) + 0.66
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


arc = math.pi / 2
for level in range(7):
    result = 0
    pts = gold_spiral_sampling_patch(np.array([0, 0, 1]), arc, 64)
    for p in gold_spiral_sampling_patch(np.array([0, 0, 1]), arc, 200000):
        theta = pts @ p
        result = max(result, min(np.arccos(np.abs(theta))))

    print("level:", level)
    print("  acc:", result)
    print("  rad:", result * 1.2)
    print("  deg:", result / 2 / np.pi * 360)
    arc = result * 1.2


ax = plt.figure(figsize=(15, 12)).add_subplot(111, projection="3d")
cb = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2])
ax.scatter(0, 0, 0.5, c="red", marker="^")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.colorbar(cb)
plt.show()
