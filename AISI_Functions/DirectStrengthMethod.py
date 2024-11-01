import math


def F211(Section, **kwargs):
    """
    AISI D100-17
    F2.1.1 - Singly or Doubly Symmetric Sections Bending About Symmetric Axis
    :param Section:
    :param kwargs:
    :return: Fcre - The elastic buckling stress.
    """
    # ==== Input ====
    Cb = kwargs['Cb']
    E = 29500.0
    G = 1000.0
    L = kwargs['L']
    K = kwargs['K']
    Lt = kwargs['Lt']
    Kt = kwargs['Kt']
    cx = Section[1][2]
    cy = Section[1][1]
    xo = Section[1][9]
    y0 = Section[1][10]
    A = Section[1][0]
    Ixx = Section[1][3]
    Wxx = Section[1][4]
    Ixy = Section[1][7]
    Iyy = Section[1][5]
    Wyy = Section[1][6]
    I11 = Section[1][3]
    I22 = Section[1][5]
    Cw = Section[1][11]
    J = Section[1][12]
    # ====   ====
    rx = math.sqrt(Ixx / A)
    ry = math.sqrt(Iyy / A)
    # Eq. F2.1.1-3
    ro = math.sqrt(math.pow(rx, 2) + math.pow(ry, 2) + math.pow(xo, 2))

    # Eq. F2.1.1-4
    sey = math.pow(math.pi, 2) * E / math.pow((K * L) / ry, 2)
    # Eq. F2.1.1-5
    p1 = 1 / (A * math.pow(ro, 2))
    p2 = G * J
    p3 = math.pow(math.pi, 2) * E * Cw
    p4 = math.pow(Kt * Lt, 2)
    set = p1*(p2+p3/p4)

    # Eq. F2.1.1-1
    Fcre = Cb * ro * A / Wxx * math.sqrt(sey * set)
    return Fcre



def flexuralStrength(fy: float, Sf: float, ratioFlxLocal: float, ratioFlxDist: float):
    # b. Yielding and global buckling (Section F2)
    My = Sf * fy
    Fn = fy
    Mne = Sf * Fn
    if Mne > My:
        Mne = My
    # ASD
    omega = 1.67
    oMne = omega * Mne
    # LRFD

    # F3 Local Buckling Interacting With Yielding and Global Buckling
    # F3.2.1 Members without holes
    Mcrl = ratioFlxLocal * My
    laml = math.sqrt(Mne / Mcrl)
