import math
import BucklingAnalysis as bucklAna

mem1 = bucklAna.C1


# print(mem1['Sect_Props'][1])


def F211(Section, **kwargs):
    """
    AISI S100-16
    F2.1.1 - Singly or Doubly Symmetric Sections Bending About Symmetric Axis
    :param Section: Gross section properties.
    :param kwargs:
    :return: Fcre - The elastic buckling stress.
    """
    # ==== Input ====
    Cb = kwargs['Cb']
    E = 29500.0
    G = 1000.0
    L = Section['Reference_Length']
    K = 1.0
    Lt = kwargs['Lt']
    Kt = kwargs['Kt']
    cx = Section['Sect_Props'][1][2]
    cy = Section['Sect_Props'][1][1]
    xo = Section['Sect_Props'][1][9]
    y0 = Section['Sect_Props'][1][10]
    A = Section['Sect_Props'][1][0]
    Ixx = Section['Sect_Props'][1][3]
    Wxx = Section['Sect_Props'][1][4]
    Ixy = Section['Sect_Props'][1][7]
    Iyy = Section['Sect_Props'][1][5]
    Wyy = Section['Sect_Props'][1][6]
    I11 = Section['Sect_Props'][1][3]
    I22 = Section['Sect_Props'][1][5]
    Cw = Section['Sect_Props'][1][11]
    J = Section['Sect_Props'][1][12]
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
    set = p1 * (p2 + p3 / p4)

    # Eq. F2.1.1-1
    Fcre = Cb * ro * A / Wxx * math.sqrt(sey * set)
    return Fcre


def F21(Section, Fcre: float):
    """
    AISI S100-16
    F2.1 - Initiation of Yielding Strength
    :param Section: Section properties.
    :param Fcre: Critical elastic lateral-torsional buckling stress, determined in accordance
    with Section F2.1.1 or Appendix 2.
    :return: Mne, The nominal flexural strength [resistance].
    """

    fy = Section['Yield_stress']
    Wxx = Section['Sect_Props'][1][4]
    if Fcre >= 2.78 * fy:
        # Equation F2.1-3
        Fn = fy
    elif 0.56 * fy < Fcre < 2.78 * fy:
        # Equation F2.1-4
        Fn = 10.0 / 9.0 * fy * (1 - (10 * fy) / (36 * Fcre))
    else:
        # Equation F2.1-5
        Fn = Fcre
    # Equation F2.1-2
    My = Wxx * fy
    print(f'My = {My}')
    # Equation F2.1-1
    Mne = Wxx * Fn
    if Mne > My:
        Mne = My
    return Mne


def F32(Section, Mne: float, ratioFlxLocal: float):
    """
    AISI S100-16
    Section F3 Local Buckling Interacting with Yielding and Global Buckling
    Section F3.2 Direct Strength Method
    For the direct Strength Method, the nominal flexural strength, Mnl, for local buckling shall be calculated in
    accordance with Section F3.2.1 through F3.2.3.
    :param Section: Section properties.
    :param Mne: Mne, The nominal flexural strength [resistance]. From F2.1.
    :param ratioFlxLocal: Load factor for local buckling mode.
    :return: Mnl, The nominal flexural strength, for considering interaction of local buckling and global buckling.
    """
    fy = Section['Yield_stress']
    Wxx = Section['Sect_Props'][1][4]
    # Section F3.2 Direct strength method.
    # Section F3.2.1 Members without holes.
    My = Wxx * fy
    Mcrl = ratioFlxLocal * My
    # Equation F3.2.1-3
    laml = math.sqrt(Mne / Mcrl)
    if laml <= 0.776:
        # Equation F3.2.1-1
        Mnl = Mne
    else:
        # Equation F3.2.1-2
        Mnl = (1 - 0.15 * math.pow(Mcrl / Mne, 0.4)) * math.pow(Mcrl / Mne, 0.4) * Mne

    # ASD
    omega = 1.67
    # LRFD
    ff = 0.90
    FlexuralStrength = {'oMne': Mne / omega, 'oMnl': Mnl / omega, 'ffMne': ff * Mne, 'ffMnl': ff * Mnl}
    return FlexuralStrength


def F41(Section, ratioFlxDist: float):
    """
    AISI S100-16
    Section F4 Distortional Buckling
    Section F4.1 Direct Strength Method
    :param Section: Section properties.
    :param ratioFlxDist: Load factor for distortional buckling mode.
    :return: Mnd
    """
    fy = Section['Yield_stress']
    Wxx = Section['Sect_Props'][1][4]
    # Section F4.1 Members without holes.
    # Equation F4.1-4
    My = Wxx * fy
    Fcrd = ratioFlxDist * fy
    # Equation F4.1-5
    Mcrd = Wxx * Fcrd
    # Equation F4.1-3
    lamd = math.sqrt(My / Mcrd)
    if lamd <= 0.673:
        # Equation F4.1-1
        Mnd = My
    else:
        # Equation F4.1-2
        Mnd = (1 - 0.22 * math.pow(Mcrd / My, 0.5)) * math.pow(Mcrd / My, 0.5) * My

    # ASD
    omega = 1.67
    # LRFD
    ff = 0.90
    FlexuralStrength = {'oMnd': Mnd / omega, 'ffMnd': ff * Mnd}
    return FlexuralStrength


Fcre = F211(mem1, Cb=1.0, Kt=1.0, Lt=mem1['Reference_Length'])
Mne = F21(mem1, Fcre)
Mnl = F32(mem1, Mne, 0.647)
Mnd = F41(mem1, 0.84)

print(Fcre)
print(Mne)
print(Mnl)
print(Mnd)
