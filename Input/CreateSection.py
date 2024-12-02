import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Literal


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
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


def plotter(nodes, thk, descp, case, fy):
    # Drawing the cross section shape
    # IDs of the nodes.
    id_nodes = nodes[:, 0]
    # X values of the nodes.
    x_nodes = nodes[:, 1]
    # Y values of the nodes.
    y_nodes = nodes[:, 2]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    minimas = []
    fig.suptitle(f'Signature Curve\n{case}\nfy: {fy:.2f} ksi')
    # Thickness
    thk = thk
    for i in range(len(x_nodes)):
        ax2.axes.annotate(f'{id_nodes[i] + 1:.0f}', xy=(x_nodes[i] * 1.03, y_nodes[i]), xycoords='data', fontsize=8)
    ax2.plot(x_nodes, y_nodes, linewidth=thk * 30, color='green', marker="o", markersize=2)
    ax2.axis('equal')
    ax2.axes.set_xlabel('length [in]')
    ax2.axes.set_ylabel('length [in]')
    ax2.axes.set_title(f'Cross Section {descp}')
    # Show the plot
    plt.show()


# ======================================================================================================================
# CREATE A SECTION AND CREATE A LIST FOR NODE AND ELEMENTS
# ======================================================================================================================
class C_Section:
    def __init__(self, A: float, B: float, C: float, t: float, R: float, angle: Literal[0, 90, 270]):
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
        self.nodes = np.array([[]])
        self.elements = None
        self.descp_rep = None
        self.descp_Plot = None
        self.centerline()
        self.coordinates()
        self.ang_shape = None
        self.orientation(self.angle)

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
        # Final nodes and elements for Opensees
        # ===================================
        self.nodes = CsectionWithNumbers.T
        # If angle is 90, shift section as flange width in Y dir.
        # If angle is 270, shift section as web height in X dir.
        if self.angle == 90:
            for i in self.nodes:
                i[2] = i[2] + self.B
        elif self.angle == 270:
            for i in self.nodes:
                i[1] = i[1] + self.A

        # Shape of the node matrix
        num_cols, num_rows = Csection.shape
        self.elements = np.empty([num_rows - 1, 5])
        for i in range(num_rows - 1):
            self.elements[i, 0] = i
            self.elements[i, 1] = i
            self.elements[i, 2] = i + 1
            self.elements[i, 3] = self.t
            self.elements[i, 4] = 0

        self.descp_Plot = f'Section :C {self.A:.3f} x {self.B:.3f} x {self.C:.3f} - {self.t:.3f}'
        self.descp_rep = (f'Section :C {self.A:.3f} x {self.B:.3f} x {self.C:.3f} - {self.t:.3f}\n'
                          f'   A:{self.A:.3f} in, Web height\n'
                          f'   B:{self.B:.3f} in, Flange width\n'
                          f'   C:{self.C:.3f} in, Lip length\n'
                          f'   R:{self.R:.3f} in, Inner radius\n'
                          f'   t:{self.t:.3f} in, Thickness')

    def orientation(self, angle):
        if angle == 0:
            self.ang_shape = (f'      ┌-┐\n'
                              f'        |\n'
                              f'      └-┘\n')
        elif angle == 270:
            self.ang_shape = (f'   ┌   ┐\n'
                              f'   └---┘\n')
        else:
            self.ang_shape = (f'  ┌---┐\n'
                              f'  └   ┘\n')


class C_Section_SharpCorner:
    def __init__(self, A: float, B: float, C: float, t: float, R: float, angle: Literal[0, 90, 270]):
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
        self.nodes = np.array([[]])
        self.elements = None
        self.descp_rep = None
        self.descp_Plot = None
        self.centerline()
        self.coordinates()
        self.ang_shape = None
        self.orientation(self.angle)

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
                                  [self.bb, 0.0],
                                  [self.bb / 2.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, self.aa * (1.0 / 4.0)],
                                  [0.0, self.aa * (2.0 / 4.0)],
                                  [0.0, self.aa * (3.0 / 4.0)],
                                  [0.0, self.aa],
                                  [self.bb / 2.0, self.aa],
                                  [self.bb, self.aa],
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
        # Final nodes and elements for Opensees
        # ===================================
        self.nodes = CsectionWithNumbers.T
        # If angle is 90, shift section as flange width in Y dir.
        # If angle is 270, shift section as web height in X dir.
        if self.angle == 90:
            for i in self.nodes:
                i[2] = i[2] + self.B
        elif self.angle == 270:
            for i in self.nodes:
                i[1] = i[1] + self.A

        # Shape of the node matrix
        num_cols, num_rows = Csection.shape
        self.elements = np.empty([num_rows - 1, 5])
        for i in range(num_rows - 1):
            self.elements[i, 0] = i
            self.elements[i, 1] = i
            self.elements[i, 2] = i + 1
            self.elements[i, 3] = self.t
            self.elements[i, 4] = 0

        self.descp_Plot = f'Section :C {self.A:.3f} x {self.B:.3f} x {self.C:.3f} - {self.t:.3f}'
        self.descp_rep = (f'Section :C {self.A:.3f} x {self.B:.3f} x {self.C:.3f} - {self.t:.3f}\n'
                          f'   A:{self.A:.3f} in, Web height\n'
                          f'   B:{self.B:.3f} in, Flange width\n'
                          f'   C:{self.C:.3f} in, Lip length\n'
                          f'   R:{self.R:.3f} in, Inner radius\n'
                          f'   t:{self.t:.3f} in, Thickness')

    def orientation(self, angle):
        if angle == 0:
            self.ang_shape = (f'      ┌-┐\n'
                              f'        |\n'
                              f'      └-┘\n')
        elif angle == 270:
            self.ang_shape = (f'   ┌   ┐\n'
                              f'   └---┘\n')
        else:
            self.ang_shape = (f'  ┌---┐\n'
                              f'  └   ┘\n')


class U_Section:
    def __init__(self, A: float, B: float, t: float, R: float, angle: Literal[0, 90, 270]):
        self.Usection = None
        self.A = A
        self.B = B
        self.t = t
        self.R = R
        self.angle = angle
        self.aa = None
        self.bb = None
        self.tcore = None
        self.a = None
        self.b = None
        self.r = None
        self.nodes = np.array([[]])
        self.elements = None
        self.descp_rep = None
        self.descp_Plot = None
        self.centerline()
        self.coordinates()
        self.ang_shape = None
        self.orientation(self.angle)

    def centerline(self):
        self.r = self.R + self.t / 2.0
        # Centerline dimensions
        self.aa = self.A - self.t
        self.bb = self.B - self.t / 2.0
        self.tcore = self.t - 0.04
        # Flat portions
        self.a = self.aa - 2 * self.r
        self.b = self.bb - self.r

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
        radius = np.array([0, self.r, 0.0])

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
        start_ang = 0

        self.Usection = np.array([[self.bb, 0],
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
                                  [self.bb, self.aa]])
        # =================

        # Function call for rotation
        Usection, RotatedCsection = rotateCoordinates(self.Usection.T, self.angle)
        # Create id numbers for each row
        numbers = np.arange(RotatedCsection.shape[1], dtype=int)
        # Adding id numbers to the coordinates matrix
        UsectionWithNumbers = np.vstack([numbers, RotatedCsection])

        # Creating ones
        ones = np.ones((4, RotatedCsection.shape[1]))
        # Adding one numbers to the coordinates matrix
        UsectionWithNumbers = np.vstack([UsectionWithNumbers, ones])
        # Creating zeros
        zeros = np.zeros((RotatedCsection.shape[1]))
        # Adding zeros to the coordinates matrix
        UsectionWithNumbers = np.vstack([UsectionWithNumbers, zeros])

        # ===================================
        # Final nodes and elements for Opensees
        # ===================================
        self.nodes = UsectionWithNumbers.T
        # If angle is 90, shift section as flange width in Y dir.
        # If angle is 270, shift section as web height in X dir.
        if self.angle == 90:
            for i in self.nodes:
                i[2] = i[2] + self.B
        elif self.angle == 270:
            for i in self.nodes:
                i[1] = i[1] + self.A

        # Shape of the node matrix
        num_cols, num_rows = Usection.shape
        self.elements = np.empty([num_rows - 1, 5])
        for i in range(num_rows - 1):
            self.elements[i, 0] = i
            self.elements[i, 1] = i
            self.elements[i, 2] = i + 1
            self.elements[i, 3] = self.t
            self.elements[i, 4] = 0

        self.descp_Plot = f'Section :U {self.A:.3f} x {self.B:.3f} - {self.t:.3f}'
        self.descp_rep = (f'Section :U {self.A:.3f} x {self.B:.3f} - {self.t:.3f}\n'
                          f'   A:{self.A:.3f} in, Web height\n'
                          f'   B:{self.B:.3f} in, Flange width\n'
                          f'   R:{self.R:.3f} in, Inner radius\n'
                          f'   t:{self.t:.3f} in, Thickness')

    def orientation(self, angle):
        if angle == 0:
            self.ang_shape = (f'   -┐\n'
                              f'    |\n'
                              f'   -┘\n')
        elif angle == 270:
            self.ang_shape = (f'       \n'
                              f'  └---┘\n')
        else:
            self.ang_shape = (f'  ┌---┐\n'
                              f'       \n')


# ======================================================================================================================
# CALCULATION GROSS SECTION PROPERTIES
# ======================================================================================================================
class GrossProps:
    def __init__(self, x, y, t, r):
        self.Wy = None
        self.Wx = None
        self.propDict = None
        self.cy = None
        self.cx = None
        self.zgr = None
        self.zgl = None
        self.zgt = None
        self.zgb = None
        self.xo = None
        self.It = None
        self.Cw = None
        self.ysc = None
        self.xsc = None
        self.Ixy = None
        self.Iy = None
        self.Ix = None
        self.zgx = None
        self.zgy = None
        self.Ar = None
        self.x = x
        self.y = y
        self.t = t
        self.r = r
        self.calculate()

    def calculate(self):
        x = self.x
        y = self.y
        t = self.t
        r = self.r
        # Area of cross section
        da = np.zeros([len(x)])
        ba = np.zeros([len(x)])
        for i in range(1, len(da)):
            da[i] = math.sqrt(math.pow(x[i - 1] - x[i], 2) + math.pow(y[i - 1] - y[i], 2)) * t
            ba[i] = math.sqrt(math.pow(x[i - 1] - x[i], 2) + math.pow(y[i - 1] - y[i], 2))
        self.Ar = np.sum(da)
        Lt = np.sum(ba)
        # Total rj.tetaj/90
        Trj = 4 * 4 * r * (1.0 / 4.0)
        delta = 0.43 * Trj / Lt
        # First moment of area and coordinate for gravity centre
        sx0 = np.zeros([len(x)])
        sy0 = np.zeros([len(x)])
        for i in range(1, len(sx0)):
            sx0[i] = (y[i] + y[i - 1]) * da[i] / 2
        self.zgy = np.sum(sx0) / self.Ar
        for i in range(1, len(sy0)):
            sy0[i] = (x[i] + x[i - 1]) * da[i] / 2
        self.zgx = np.sum(sy0) / self.Ar

        # Second moment of area
        Ix0 = np.zeros([len(x)])
        Iy0 = np.zeros([len(x)])
        for i in range(1, len(Ix0)):
            Ix0[i] = (math.pow(y[i], 2) + math.pow(y[i - 1], 2) + y[i] * y[i - 1]) * da[i] / 3
        for i in range(1, len(Iy0)):
            Iy0[i] = (math.pow(x[i], 2) + math.pow(x[i - 1], 2) + x[i] * x[i - 1]) * da[i] / 3
        self.Ix = np.sum(Ix0) - self.Ar * math.pow(self.zgy, 2)
        self.Iy = np.sum(Iy0) - self.Ar * math.pow(self.zgx, 2)

        # Product moment of area
        Ixy0 = np.zeros([len(x)])
        for i in range(1, len(Ixy0)):
            Ixy0[i] = (2 * x[i - 1] * y[i - 1] + 2 * x[i] * y[i] + x[i - 1] * y[i] + x[i] * y[i - 1]) * da[i] / 6
        self.Ixy = np.sum(Ixy0) - (np.sum(sx0) * np.sum(sy0)) / self.Ar

        # Principle axis
        alfa = 0.5 * math.atan(2 * self.Ixy / (self.Iy - self.Ix))
        Iksi = 0.5 * (self.Ix + self.Iy + math.sqrt(math.pow(self.Iy - self.Ix, 2) + 4 * math.pow(self.Ixy, 2)))
        Ieta = 0.5 * (self.Ix + self.Iy - math.sqrt(math.pow(self.Iy - self.Ix, 2) + 4 * math.pow(self.Ixy, 2)))

        # Sectoral coordinates
        w = np.zeros([len(x)])
        w0 = np.zeros([len(x)])
        Iw = np.zeros([len(x)])
        w0[0] = 0
        for i in range(1, len(w0)):
            w0[i] = x[i - 1] * y[i] - x[i] * y[i - 1]
            w[i] = w[i - 1] + w0[i]
            Iw[i] = (w[i - 1] + w[i]) * da[i] / 2
        wmean = np.sum(Iw) / self.Ar

        # Sectorial constants
        Ixw0 = np.zeros([len(x)])
        Iyw0 = np.zeros([len(x)])
        Iww0 = np.zeros([len(x)])
        for i in range(1, len(Ixw0)):
            Ixw0[i] = (2 * x[i - 1] * w[i - 1] + 2 * x[i] * w[i] + x[i - 1] * w[i] + x[i] * w[i - 1]) * da[i] / 6
            Iyw0[i] = (2 * y[i - 1] * w[i - 1] + 2 * y[i] * w[i] + y[i - 1] * w[i] + y[i] * w[i - 1]) * da[i] / 6
            Iww0[i] = (math.pow(w[i], 2) + math.pow(w[i - 1], 2) + w[i] * w[i - 1]) * da[i] / 3
        Ixw = np.sum(Ixw0) - np.sum(sy0) * np.sum(Iw) / self.Ar
        Iyw = np.sum(Iyw0) - np.sum(sx0) * np.sum(Iw) / self.Ar
        Iww = np.sum(Iww0) - math.pow(np.sum(Iw), 2) / self.Ar

        # Shear centre
        self.xsc = (Iyw * self.Iy - Ixw * self.Ixy) / (self.Ix * self.Iy - math.pow(self.Ixy, 2))
        self.ysc = (-Ixw * self.Ix + Iyw * self.Ixy) / (self.Ix * self.Iy - math.pow(self.Ixy, 2))

        # Warping constant
        self.Cw = Iww + self.ysc * Ixw - self.xsc * Iyw

        # Torsion constant
        It0 = np.zeros([len(x)])
        for i in range(1, len(It0)):
            It0[i] = da[i] * math.pow(t, 2) / 3
        self.It = np.sum(It0)

        # Distance between centroid and shear centre
        self.xo = abs(self.xsc) + self.zgx
        # Distances from the boundaries
        self.zgb = self.zgy
        self.zgt = max(y) - self.zgb
        self.zgl = self.zgx
        self.zgr = max(x) - self.zgl
        self.cx = max(self.zgl, self.zgr)
        self.cy = max(self.zgb, self.zgt)
        # Section modulus
        self.Wx = self.Ix / max(self.zgb, self.zgt)
        self.Wy = self.Iy / max(self.zgl, self.zgr)

        # Data dictionary
        self.propDict = {
            "Ar ": str(round(self.Ar, 3)) + " in2",
            "zgx ": str(round(self.zgx, 3)) + " in",
            "zgy ": str(round(self.zgy, 3)) + " in",
            "Ix ": str(round(self.Ix, 3)) + " in4",
            "Wx ": str(round(self.Ix / max(self.zgb, self.zgt), 3)) + " in3",
            "Iy ": str(round(self.Iy, 3)) + " in4",
            "Wy ": str(round(self.Iy / max(self.zgl, self.zgr), 3)) + " in3",
            "Ixy ": str(round(self.Ixy, 3)) + " in4",
            "Iw ": str(round(np.sum(self.Cw), 5)) + " in3",
            "xsc ": str(round(self.xsc, 3)) + " in",
            "ysc ": str(round(self.ysc, 3)) + " in",
            "Cw ": str(round(self.Cw, 5)) + " in6",
            "It ": str(round(self.It, 5)) + " in4",
            "xo ": str(round(self.xo, 3)) + " in"
        }
