import math

import numpy as np


# Section input
def hatSection(A, B, C, D, t):
    nodes = np.array([[0, 0, D, 1, 1, 1, 1, 0],
                      [1, 0, 0, 1, 1, 1, 1, 0],
                      [2, C / 2, 0, 1, 1, 1, 1, 0],
                      [3, C, 0, 1, 1, 1, 1, 0],
                      [4, C, (1.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [5, C, (2.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [6, C, (3.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [7, C, (4.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [8, C + (1.0 / 4.0) * B, A, 1, 1, 1, 1, 0],
                      [9, C + (2.0 / 4.0) * B, A, 1, 1, 1, 1, 0],
                      [10, C + (3.0 / 4.0) * B, A, 1, 1, 1, 1, 0],
                      [11, C + (4.0 / 4.0) * B, A, 1, 1, 1, 1, 0],
                      [12, C + B, (3.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [13, C + B, (2.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [14, C + B, (1.0 / 4.0) * A, 1, 1, 1, 1, 0],
                      [15, C + B, 0, 1, 1, 1, 1, 0],
                      [16, C + B + C / 2, 0, 1, 1, 1, 1, 0],
                      [17, C + B + C, 0, 1, 1, 1, 1, 0],
                      [18, C + B + C, D, 1, 1, 1, 1, 0]
                      ])
    thickness = t
    elements = np.array([[0, 0, 1, thickness, 0],
                         [1, 1, 2, thickness, 0],
                         [2, 2, 3, thickness, 0],
                         [3, 3, 4, thickness, 0],
                         [4, 4, 5, thickness, 0],
                         [5, 5, 6, thickness, 0],
                         [6, 6, 7, thickness, 0],
                         [7, 7, 8, thickness, 0],
                         [8, 8, 9, thickness, 0],
                         [9, 9, 10, thickness, 0],
                         [10, 10, 11, thickness, 0],
                         [11, 11, 12, thickness, 0],
                         [12, 12, 13, thickness, 0],
                         [13, 13, 14, thickness, 0],
                         [14, 14, 15, thickness, 0],
                         [15, 15, 16, thickness, 0],
                         [16, 16, 17, thickness, 0],
                         [17, 17, 18, thickness, 0]
                         ])
    descp = "Section : A:" + A + " B:" + B + " C:" + C + " D:" + D + " t:" + t
    return nodes, elements, thickness, descp


def rotateCoordinates(XY, angle):
    # Shape of the input matrix
    num_rows, num_cols = XY.shape
    # Creating a dummy zeros matrix
    dummyzeros = np.zeros(num_cols)
    # Adding dummy zeros
    SectionWithZeros = np.vstack([XY, dummyzeros])
    # Cosine and sines
    c, s = np.cos(math.radians(angle)), np.sin(math.radians(angle))
    j = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])
    # Multiply with transformation matrix
    RotatedCoordinates = np.matmul(j, SectionWithZeros)
    # Remove the dummy zeros
    RotatedCoordinates = np.delete(RotatedCoordinates, 2, 0)
    return XY, RotatedCoordinates


def ceeSection(A, B, C, t, angle):
    # C section input matrix is column1 is X column2 is Y coordinates.
    Csection = np.array([[B, C],
                         [B, 0],
                         [B / 2, 0],
                         [0, 0],
                         [0, (1.0 / 4.0) * A],
                         [0, (2.0 / 4.0) * A],
                         [0, (3.0 / 4.0) * A],
                         [0, (4.0 / 4.0) * A],
                         [B / 2, A],
                         [B, A],
                         [B, A - C]
                         ])
    # Function call for rotation
    Csection, RotatedCsection = rotateCoordinates(Csection.T, angle)
    # Create id numbers for each row
    numbers = np.arange(RotatedCsection.shape[1])
    # print(RotatedCsection.T)
    # Adding id numbers to the coordinates matrix
    CsectionWithNumbers = np.vstack([numbers, RotatedCsection])

    # print(CsectionWithNumbers.T)
    # Creating ones
    ones = np.ones((4, RotatedCsection.shape[1]))
    # Adding one numbers to the coordinates matrix
    CsectionWithNumbers = np.vstack([CsectionWithNumbers, ones])
    # Creating zeros
    zeros = np.zeros((RotatedCsection.shape[1]))
    # Adding zeros to the coordinates matrix
    CsectionWithNumbers = np.vstack([CsectionWithNumbers, zeros])
    # print(CsectionWithNumbers.T)

    nodes = CsectionWithNumbers.T
    # If angle is 90, shift section as flange width in Y dir.
    # If angle is 270, shift section as web height in X dir.
    if angle == 90:
        for i in nodes:
            i[2] = i[2] + B
    elif angle == 270:
        for i in nodes:
            i[1] = i[1] + A

    thickness = t
    elements = np.array([[0, 0, 1, thickness, 0],
                         [1, 1, 2, thickness, 0],
                         [2, 2, 3, thickness, 0],
                         [3, 3, 4, thickness, 0],
                         [4, 4, 5, thickness, 0],
                         [5, 5, 6, thickness, 0],
                         [6, 6, 7, thickness, 0],
                         [7, 7, 8, thickness, 0],
                         [8, 8, 9, thickness, 0],
                         [9, 9, 10, thickness, 0]
                         ])
    descp = "Section : A:" + str(A) + " B:" + str(B) + " C:" + str(C) + " t:" + str(t)
    return nodes, elements, thickness, descp


def lengthRange(refLen, unit):
    if unit == "imperial":
        lengths_data = np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36,
            38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
            132, 144, 156, 168, 180, 204, 228, 252, 276, 300])
        lengths_data = np.sort(np.append(lengths_data, refLen))
    else:
        lengths_data = np.array([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000,
            2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000])
    return lengths_data


def grossProp(x, y, t, r):
    # Area of cross section
    da = np.zeros([len(x)])
    ba = np.zeros([len(x)])
    for i in range(1, len(da)):
        da[i] = math.sqrt(math.pow(x[i - 1] - x[i], 2) + math.pow(y[i - 1] - y[i], 2)) * t
        ba[i] = math.sqrt(math.pow(x[i - 1] - x[i], 2) + math.pow(y[i - 1] - y[i], 2))
    Ar = np.sum(da)
    Lt = np.sum(ba)
    # Total rj.tetaj/90
    Trj = 4 * 4 * r * (1.0 / 4.0)
    delta = 0.43 * Trj / Lt
    # First moment of area and coordinate for gravity centre
    sx0 = np.zeros([len(x)])
    sy0 = np.zeros([len(x)])
    for i in range(1, len(sx0)):
        sx0[i] = (y[i] + y[i - 1]) * da[i] / 2
    zgy = np.sum(sx0) / Ar
    for i in range(1, len(sy0)):
        sy0[i] = (x[i] + x[i - 1]) * da[i] / 2
    zgx = np.sum(sy0) / Ar

    # Second moment of area
    Ix0 = np.zeros([len(x)])
    Iy0 = np.zeros([len(x)])
    for i in range(1, len(Ix0)):
        Ix0[i] = (math.pow(y[i], 2) + math.pow(y[i - 1], 2) + y[i] * y[i - 1]) * da[i] / 3
    for i in range(1, len(Iy0)):
        Iy0[i] = (math.pow(x[i], 2) + math.pow(x[i - 1], 2) + x[i] * x[i - 1]) * da[i] / 3
    Ix = np.sum(Ix0) - Ar * math.pow(zgy, 2)
    Iy = np.sum(Iy0) - Ar * math.pow(zgx, 2)

    # Product moment of area
    Ixy0 = np.zeros([len(x)])
    for i in range(1, len(Ixy0)):
        Ixy0[i] = (2 * x[i - 1] * y[i - 1] + 2 * x[i] * y[i] + x[i - 1] * y[i] + x[i] * y[i - 1]) * da[i] / 6
    Ixy = np.sum(Ixy0) - (np.sum(sx0) * np.sum(sy0)) / Ar

    # Principle axis
    alfa = 0.5 * math.atan(2 * Ixy / (Iy - Ix))
    Iksi = 0.5 * (Ix + Iy + math.sqrt(math.pow(Iy - Ix, 2) + 4 * math.pow(Ixy, 2)))
    Ieta = 0.5 * (Ix + Iy - math.sqrt(math.pow(Iy - Ix, 2) + 4 * math.pow(Ixy, 2)))

    # Sectoral coordinates
    w = np.zeros([len(x)])
    w0 = np.zeros([len(x)])
    Iw = np.zeros([len(x)])
    w0[0] = 0
    for i in range(1, len(w0)):
        w0[i] = x[i - 1] * y[i] - x[i] * y[i - 1]
        w[i] = w[i - 1] + w0[i]
        Iw[i] = (w[i - 1] + w[i]) * da[i] / 2
    wmean = np.sum(Iw) / Ar

    # Sectorial constants
    Ixw0 = np.zeros([len(x)])
    Iyw0 = np.zeros([len(x)])
    Iww0 = np.zeros([len(x)])
    for i in range(1, len(Ixw0)):
        Ixw0[i] = (2 * x[i - 1] * w[i - 1] + 2 * x[i] * w[i] + x[i - 1] * w[i] + x[i] * w[i - 1]) * da[i] / 6
        Iyw0[i] = (2 * y[i - 1] * w[i - 1] + 2 * y[i] * w[i] + y[i - 1] * w[i] + y[i] * w[i - 1]) * da[i] / 6
        Iww0[i] = (math.pow(w[i], 2) + math.pow(w[i - 1], 2) + w[i] * w[i - 1]) * da[i] / 3
    Ixw = np.sum(Ixw0) - np.sum(sy0) * np.sum(Iw) / Ar
    Iyw = np.sum(Iyw0) - np.sum(sx0) * np.sum(Iw) / Ar
    Iww = np.sum(Iww0) - math.pow(np.sum(Iw), 2) / Ar

    # Shear centre
    xsc = (Iyw * Iy - Ixw * Ixy) / (Ix * Iy - math.pow(Ixy, 2))
    ysc = (-Ixw * Ix + Iyw * Ixy) / (Ix * Iy - math.pow(Ixy, 2))

    # Warping constant
    Cw = Iww + ysc * Ixw - xsc * Iyw

    # Torsion constant
    It0 = np.zeros([len(x)])
    for i in range(1, len(It0)):
        It0[i] = da[i] * math.pow(t, 2) / 3
    It = np.sum(It0)

    # Distance between centroid and shear centre
    xo = abs(xsc) + zgx
    # Distances from the boundaries
    zgb = zgy
    zgt = max(y) - zgb
    zgl = zgx
    zgr = max(x) - zgl
    # Calculated data
    propData = np.zeros([14])
    propData[0] = Ar * (1 - delta)
    propData[1] = max(zgb, zgt)
    propData[2] = max(zgl, zgr)
    propData[3] = Ix * (1 - 2 * delta)
    propData[4] = Ix * (1 - 2 * delta) / max(zgb, zgt)
    propData[5] = Iy * (1 - 2 * delta)
    propData[6] = Iy * (1 - 2 * delta) / max(zgl, zgr)
    propData[7] = Ixy
    propData[8] = np.sum(Iw)
    propData[9] = xsc
    propData[10] = ysc
    propData[11] = Cw * (1 - 4 * delta)
    propData[12] = It * (1 - 2 * delta)
    propData[13] = xo

    # Data dictionary
    prop = {
        "Ar ": str(round(Ar, 3)) + " in2",
        "zgx ": str(round(zgx, 3)) + " in",
        "zgy ": str(round(zgy, 3)) + " in",
        "Ix ": str(round(Ix, 3)) + " in4",
        "Wx ": str(round(Ix * (1 - 2 * delta) / max(zgb, zgt), 3)) + " in3",
        "Iy ": str(round(Iy, 3)) + " in4",
        "Wy ": str(round(Iy * (1 - 2 * delta) / max(zgl, zgr), 3)) + " in3",
        "Ixy ": str(round(Ixy, 3)) + " in4",
        "Iw ": str(round(np.sum(Iw), 5)) + " in3",
        "xsc ": str(round(xsc, 3)) + " in",
        "ysc ": str(round(ysc, 3)) + " in",
        "Cw ": str(round(Cw, 5)) + " in6",
        "It ": str(round(It, 5)) + " in4",
        "xo ": str(round(xo, 3)) + " in"
    }
    return prop, propData
