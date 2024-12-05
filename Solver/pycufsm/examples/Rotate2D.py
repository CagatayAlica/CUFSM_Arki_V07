"""
Lyle Scott, III  // lyle@ls3.io
Multiple ways to rotate a 2D point around the origin / a point.
Timer benchmark results @ https://gist.github.com/LyleScott/d17e9d314fbe6fc29767d8c5c029c362
"""
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt


def rotate_via_numpy(radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    Corners = np.array([[2.5, 2.5, 1.25, 0., 0., 0., 0., 0., 1.25, 2.5, 2.5],
                        [1.773, 0., 0., 0., 2.25, 4.5, 6.75, 9., 9., 9., 7.227]])
    num_rows, num_cols = Corners.shape
    dummyseros = np.zeros(num_cols)
    print(dummyseros)
    print(Corners)
    zerosCsection = np.vstack([Corners, dummyseros])
    print(zerosCsection)
    print(Corners.shape[0])
    print(Corners.shape[1])
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])

    RotatedCorners = np.matmul(j, zerosCsection)
    print(RotatedCorners.T)
    RotatedCorners = np.delete(RotatedCorners,2,0)
    print(RotatedCorners.T)
    return Corners, RotatedCorners


def _main():
    theta = math.radians(90)
    print(rotate_via_numpy(theta)[0])
    print(rotate_via_numpy(theta)[1])

    plt.scatter(rotate_via_numpy(theta)[0][0, :], rotate_via_numpy(theta)[0][1, :])
    plt.scatter(rotate_via_numpy(theta)[1][0, :], rotate_via_numpy(theta)[1][1, :])
    plt.axis('equal')
    plt.grid(color='b', linestyle='-', linewidth=0.2)
    plt.show()


if __name__ == '__main__':
    _main()
