import matplotlib.pyplot as plt
import math
import numpy as np


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

class C_Section:
    def __init__(self, A: float, B: float, C: float, t: float, R: float, angle: float):
        self.A = A
        self.B = B
        self.C = C
        self.t = t
        self.R = R
        self.angle = angle
        self.aa = None
        self.bb = None
        self.cc = None
        self.tcore = None
        self.a = None
        self.b = None
        self.c = None
        self.r = None
        self.centerline()
    def centerline(self):
        self.r = self.R + self.t / 2.0
        # Centerline dimensions
        self.aa = self.A - self.t
        self.bb = self.B - self.t
        self.cc = self.C - self.t / 2.0
        self.tcore = self.t - 0.04
        # Flat portions
        self.a = self.aa - 2 * self.r
        self.b = self.bb - 2 * self.r
        self.c = self.cc - self.r
    def tranform(self, alfa, origin):
        # Transformation matrix
        p = origin
        c, s = np.cos(math.radians(alfa)), np.sin(math.radians(alfa))
        j = np.array([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])
        RotatedCorners = np.matmul(j, p)
        return RotatedCorners.T
    def coordinates(self):
        # Bottom right
        origin1 = np.array([self.bb - self.r, self.r, 0.0])
        radius = np.array([0, self.r, 0.0])
        start_ang = 90
        p11 = self.tranform(start_ang + 10, radius)
        p12 = self.tranform(start_ang + 20, radius)
        p13 = self.tranform(start_ang + 30, radius)
        p14 = self.tranform(start_ang + 40, radius)
        p15 = self.tranform(start_ang + 50, radius)
        p16 = self.tranform(start_ang + 60, radius)
        p17 = self.tranform(start_ang + 70, radius)
        p18 = self.tranform(start_ang + 80, radius)
        # Bottom left
        origin2 = np.array([self.r, self.r, 0.0])
        start_ang = 180
        p21 = self.tranform(start_ang + 10, radius)
        p22 = self.tranform(start_ang + 20, radius)
        p23 = self.tranform(start_ang + 30, radius)
        p24 = self.tranform(start_ang + 40, radius)
        p25 = self.tranform(start_ang + 50, radius)
        p26 = self.tranform(start_ang + 60, radius)
        p27 = self.tranform(start_ang + 70, radius)
        p28 = self.tranform(start_ang + 80, radius)
        # Top left
        origin3 = np.array([self.r, self.aa - self.r, 0.0])
        start_ang = 270
        p31 = self.tranform(start_ang + 10, radius)
        p32 = self.tranform(start_ang + 20, radius)
        p33 = self.tranform(start_ang + 30, radius)
        p34 = self.tranform(start_ang + 40, radius)
        p35 = self.tranform(start_ang + 50, radius)
        p36 = self.tranform(start_ang + 60, radius)
        p37 = self.tranform(start_ang + 70, radius)
        p38 = self.tranform(start_ang + 80, radius)
        # Top right
        origin4 = np.array([self.bb - self.r, self.aa - self.r, 0.0])
        start_ang = 0
        p41 = self.tranform(start_ang + 10, radius)
        p42 = self.tranform(start_ang + 20, radius)
        p43 = self.tranform(start_ang + 30, radius)
        p44 = self.tranform(start_ang + 40, radius)
        p45 = self.tranform(start_ang + 50, radius)
        p46 = self.tranform(start_ang + 60, radius)
        p47 = self.tranform(start_ang + 70, radius)
        p48 = self.tranform(start_ang + 80, radius)

        self.Csection = np.array([[self.bb, self.cc],
                             [self.bb, self.r],
                             [origin1[0] + p11[0], origin1[1] + p11[1]],
                             [origin1[0] + p12[0], origin1[1] + p12[1]],
                             [origin1[0] + p13[0], origin1[1] + p13[1]],
                             [origin1[0] + p14[0], origin1[1] + p14[1]],
                             [origin1[0] + p15[0], origin1[1] + p15[1]],
                             [origin1[0] + p16[0], origin1[1] + p16[1]],
                             [origin1[0] + p17[0], origin1[1] + p17[1]],
                             [origin1[0] + p18[0], origin1[1] + p18[1]],
                             [self.bb - self.r, 0],
                             [self.r + self.b / 2.0, 0],
                             [self.r, 0],
                             [origin2[0] + p21[0], origin2[1] + p21[1]],
                             [origin2[0] + p22[0], origin2[1] + p22[1]],
                             [origin2[0] + p23[0], origin2[1] + p23[1]],
                             [origin2[0] + p24[0], origin2[1] + p24[1]],
                             [origin2[0] + p25[0], origin2[1] + p25[1]],
                             [origin2[0] + p26[0], origin2[1] + p26[1]],
                             [origin2[0] + p27[0], origin2[1] + p27[1]],
                             [origin2[0] + p28[0], origin2[1] + p28[1]],
                             [0, self.r],
                             [0, self.r + self.a * (1.0 / 4.0)],
                             [0, self.r + self.a * (2.0 / 4.0)],
                             [0, self.r + self.a * (3.0 / 4.0)],
                             [0, self.r + self.a],
                             [origin3[0] + p31[0], origin3[1] + p31[1]],
                             [origin3[0] + p32[0], origin3[1] + p32[1]],
                             [origin3[0] + p33[0], origin3[1] + p33[1]],
                             [origin3[0] + p34[0], origin3[1] + p34[1]],
                             [origin3[0] + p35[0], origin3[1] + p35[1]],
                             [origin3[0] + p36[0], origin3[1] + p36[1]],
                             [origin3[0] + p37[0], origin3[1] + p37[1]],
                             [origin3[0] + p38[0], origin3[1] + p38[1]],
                             [self.r, self.aa],
                             [self.r + self.b / 2.0, self.aa],
                             [self.bb - self.r, self.aa],
                             [origin4[0] + p41[0], origin4[1] + p41[1]],
                             [origin4[0] + p42[0], origin4[1] + p42[1]],
                             [origin4[0] + p43[0], origin4[1] + p43[1]],
                             [origin4[0] + p44[0], origin4[1] + p44[1]],
                             [origin4[0] + p45[0], origin4[1] + p45[1]],
                             [origin4[0] + p46[0], origin4[1] + p46[1]],
                             [origin4[0] + p47[0], origin4[1] + p47[1]],
                             [origin4[0] + p48[0], origin4[1] + p48[1]],
                             [self.bb, self.aa - self.r],
                             [self.bb, self.aa - self.cc]])
            # =================

        # Function call for rotation
        Csection, RotatedCsection = rotateCoordinates(self.Csection.T, self.angle)
        # Create id numbers for each row
        numbers = np.arange(RotatedCsection.shape[1], dtype=int)
        # Adding id numbers to the coordinates matrix
        CsectionWithNumbers = np.vstack([numbers, RotatedCsection])

        # Creating ones
        ones = np.ones((4, RotatedCsection.shape[1]))
        # Adding one numbers to the coordinates matrix
        CsectionWithNumbers = np.vstack([CsectionWithNumbers, ones])
        # Creating zeros
        zeros = np.zeros((RotatedCsection.shape[1]))
        # Adding zeros to the coordinates matrix
        CsectionWithNumbers = np.vstack([CsectionWithNumbers, zeros])

        # ===================================
        # Final nodes for Opensees
        # ===================================
        nodes = CsectionWithNumbers.T
        # If angle is 90, shift section as flange width in Y dir.
        # If angle is 270, shift section as web height in X dir.
        if self.angle == 90:
            for i in nodes:
                i[2] = i[2] + self.B
        elif self.angle == 270:
            for i in nodes:
                i[1] = i[1] + self.A

        # Shape of the node matrix
        num_cols, num_rows = Csection.shape
        elements = np.empty([num_rows - 1, 5])
        for i in range(num_rows - 1):
            elements[i, 0] = i
            elements[i, 1] = i
            elements[i, 2] = i + 1
            elements[i, 3] = self.t
            elements[i, 4] = 0

        descp = (f'Section :C {A:.3f} x {B:.3f} x {C:.3f} - {t:.3f}\n'
                 f'   A:{A:.3f} in, Web height\n'
                 f'   B:{B:.3f} in, Flange width\n'
                 f'   C:{C:.3f} in, Lip length\n'
                 f'   R:{R:.3f} in, Inner radius\n'
                 f'   t:{t:.3f} in, Thickness')
       # return nodes, elements, thickness, descp