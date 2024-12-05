from typing import Literal
import numpy as np
import Input.CreateSection as sec


def input_parameters(A, B, C, t, R, fy, Lx, Ly, Lt, Kx, Ky, Kt):
    return {'A': A,
            'B': B,
            'C': C,
            't': t,
            'R': R,
            'fy': fy,
            'Lx': Lx,
            'Ly': Ly,
            'Lt': Lt,
            'Kx': Kx,
            'Ky': Ky,
            'Kt': Kt}


class Section_Dimensions:
    def __init__(self, A: float, B: float, C: float, t: float, R: float):
        self.A = A
        self.B = B
        self.C = C
        self.t = t
        self.R = R


class Members:
    def __init__(self, Lx: float, Ly: float, Lt: float, Kx: float, Ky: float, Kt: float,
                 Support: Literal["S-S", "C-C", "S-C", "C-F", "C-G"]):
        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt
        self.Kx = Kx
        self.Ky = Ky
        self.Kt = Kt
        self.Support = Support
        self.lengths_data = None
        self.lengthRange()

    def lengthRange(self):
        self.lengths_data = np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36,
            38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
            132, 144, 156, 168, 180, 204, 228, 252, 276, 300])
        self.lengths_data = np.sort(np.append(self.lengths_data, self.Lx))


class Materials:
    def __init__(self, fy: float):
        self.fy = fy
        self.E = 29500  # ksi
        self.G = 11300  # ksi
        self.v = 0.3


class Case:
    def __init__(self, Analysis_case: Literal['Axial', 'Flexural']):
        self.Analysis_case = Analysis_case


# Main Input
parameters = input_parameters(3.5, 1.625, 0.50, 0.0451, 0.0712, 50, 110, 50, 50, 1.0, 1.0, 1.0)
A = parameters['A']
B = parameters['B']
C = parameters['C']
t = parameters['t']
R = parameters['R']
fy = parameters['fy']
Lx = parameters['Lx']
Ly = parameters['Ly']
Lt = parameters['Lt']
Kx = parameters['Kx']
Ky = parameters['Ky']
Kt = parameters['Kt']

Section_1 = Section_Dimensions(A, B, C, t, R)
Member_1 = Members(Lx, Ly, Lt, Kx, Ky, Kt, 'S-S')
Material_1 = Materials(fy)
# Analysis Case
Case_Axial = Case('Axial')
Case_Flexural = Case('Flexural')
# Section
Section_ang0 = sec.C_Section(Section_1.A, Section_1.B, Section_1.C, Section_1.t, Section_1.R, 0)
Section_ang90 = sec.C_Section(Section_1.A, Section_1.B, Section_1.C, Section_1.t, Section_1.R, 90)
Section_ang270 = sec.C_Section(Section_1.A, Section_1.B, Section_1.C, Section_1.t, Section_1.R, 270)
# Gross Properties
Gross_ang0 = sec.GrossProps(Section_ang0.nodes[:, 1], Section_ang0.nodes[:, 2], Section_ang0.t, Section_ang0.r)
Gross_ang90 = sec.GrossProps(Section_ang90.nodes[:, 1], Section_ang90.nodes[:, 2], Section_ang90.t, Section_ang90.r)
Gross_ang270 = sec.GrossProps(Section_ang270.nodes[:, 1], Section_ang270.nodes[:, 2], Section_ang270.t,
                              Section_ang270.r)
