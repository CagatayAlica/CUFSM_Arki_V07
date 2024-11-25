import math
import Constants.Constants as cons


def stregnths(Minimas, Material, Section, Member):
    # ==== Input ====
    Cb = 1.67
    E = Material.E
    G = Material.G
    fy = Material.fy
    Lx = Member.Lx
    Kx = Member.Kx
    Ly = Member.Ly
    Ky = Member.Ky
    Lt = Member.Lt
    Kt = Member.Kt
    cx = Section['gross'].cx
    cy = Section['gross'].cy
    xo = Section['gross'].xsc
    y0 = Section['gross'].ysc
    A = Section['gross'].Ar
    Ixx = Section['gross'].Ix
    Wxx = Section['gross'].Wx
    Ixy = Section['gross'].Ixy
    Iyy = Section['gross'].Iy
    Wyy = Section['gross'].Wy
    I11 = Section['gross'].Ix
    I22 = Section['gross'].Iy
    Cw = Section['gross'].Cw
    J = Section['gross'].It
    # ====   ====
    rx = math.sqrt(Ixx / A)
    ry = math.sqrt(Iyy / A)

    # Finding the minimas
    Critical_Length_Distortional = None
    Critical_Length_Local = None
    Critical_Length_Global = None
    ratio_Local = None
    ratio_Distortional = None
    ratio_Global = None

    # check point for empty minimas list
    if not Minimas:
        raise Exception("There is no minima in the curve!")

    if len(Minimas) == 1:
        # Global buckling case.
        Critical_Length_Global = Minimas[0][0]
        ratio_Global = Minimas[0][2]
    if len(Minimas) == 2:
        # First minima for local buckling case.
        Critical_Length_Local = Minimas[0][0]
        ratio_Local = Minimas[0][2]
        # There is no second minima. Therefore, distortional case is taken equal to local case.
        Critical_Length_Distortional = Minimas[0][0]
        ratio_Distortional = Minimas[0][2]
        # Global buckling case.
        Critical_Length_Global = Minimas[1][0]
        ratio_Global = Minimas[1][2]
    if len(Minimas) > 2:
        # First minima for local buckling case.
        Critical_Length_Local = Minimas[0][0]
        ratio_Local = Minimas[0][2]
        # Second minima for distortional buckling case.
        Critical_Length_Distortional = Minimas[1][0]
        ratio_Distortional = Minimas[1][2]
        # Global buckling case.
        Critical_Length_Global = Minimas[-1][0]
        ratio_Global = Minimas[-1][2]

    # ======================================================================================================================
    # E. MEMBERS IN COMPRESSION
    # ======================================================================================================================
    def E21():
        # Fcre, flexural buckling stress.
        # Eq. E2.1-1
        Fcre = min(math.pow(cons.PI, 2) * E / math.pow(Kx * Lx / rx, 2),
                   math.pow(cons.PI, 2) * E / math.pow(Ky * Ly / ry, 2))
        return Fcre

    def E22():
        # Eq. E2.2-4
        ro = math.sqrt(math.pow(rx, 2) + math.pow(ry, 2) + math.pow(xo, 2))
        # Eq. E2.2-3
        beta = 1 - math.pow(xo / ro, 2)
        # Eq. E2.2-6
        sex = math.pow(cons.PI, 2) * E / math.pow(Kx * Lx / rx, 2)
        # Eq. E2.2-5
        p1 = 1 / (A * math.pow(ro, 2))
        p2 = G * J
        p3 = math.pow(cons.PI, 2) * E * Cw
        p4 = math.pow(Kt * Lt, 2)
        set = p1 * (p2 + p3 / p4)
        # Eq. E2.2-1
        m1 = 1 / (2 * beta)
        m2 = sex + set
        m3 = math.pow(sex + set, 2)
        m4 = 4 * beta * sex * set
        Fcre = m1 * (m2 - math.sqrt(m3 - m4))
        return Fcre

    def E2(Fcrexy, Fcret):
        # Eq. E2-4
        Fcre = min(Fcrexy, Fcret)
        lamc = math.sqrt(fy / Fcre)
        if lamc <= 1.5:
            # Eq. E2-2
            Fn = math.pow(0.658, math.pow(lamc, 2)) * fy
        else:
            # Eq. E2-3
            Fn = (0.877 / math.pow(lamc, 2)) * fy
        # Eq. E2-1
        Pne = A * Fn
        return Pne

    def E32(Pne: float, ratioAxialLocal: float):
        Py = A * fy
        Pcrl = ratioAxialLocal * Py
        Rep = f'==== Section E3 Local Buckling Interacting With Yielding and Global Buckling.\n==== Section E3.2.1 Members without holes.\n'
        Rep += (f'Py = {Py:.3f} kips. Eq.E4.1-4\n'
                f'P/Py = {ratioAxialLocal:.3f}\n'
                f'Pcrl = {Pcrl:.3f} kips.\n')
        # Eq. E3.2.1-3
        laml = math.sqrt(Pne / Pcrl)
        Rep += f'lamd = {laml:.3f} Eq.E3.2.1-3\n'
        if laml <= 0.776:
            Pnl = Pne
            Rep += f'   laml <= 0.776\nPnl = {Pnl:.3f} kips. Eq.E3.2-1\n'
        else:
            Pnl = (1 - 0.15 * math.pow(Pcrl / Pne, 0.4)) * math.pow(Pcrl / Pne, 0.4) * Pne
            Rep += f'   laml > 0.776\nPnl = {Pnl:.3f} kips. Eq.E3.2-2\n'
        # ASD
        Rep += f'_____________\nDesign Strengths:\n'
        omega = 1.80
        Rep += f'omega = {omega}.\nPnl,o = {Pnl / omega:.3f} kips.\n'
        # LRFD
        ff = 0.85
        Rep += f'ff = {ff}.\nff,Pnl = {Pnl * ff:.3f} kips.\n'
        Results = {'ASD': {'oPne': Pne / omega, 'oPnl': Pnl / omega},
                   'LRFD': {'ffPne': ff * Pne, 'ffPnl': ff * Pnl}}
        print(Rep)
        return Results

    def E41(Pne: float, ratioAxialDist: float):
        # Eq. E4.1-4
        Py = A * fy
        Pcrd = ratioAxialDist * Py
        Rep = f'==== Section E4 Distortional Buckling.\n==== Section F4.1 Members without holes.\n'
        Rep += (f'Py = {Py:.3f} kips. Eq.E4.1-4\n'
                f'P/Py = {ratioAxialDist:.3f}\n'
                f'Pcrd = {Pcrd:.3f} kips.\n')
        # Eq. E4.1-3
        lamd = math.sqrt(Pne / Pcrd)
        Rep += f'lamd = {lamd:.3f} Eq.E4.1-3\n'
        if lamd <= 0.561:
            Pnd = Pne
            Rep += f'   lamd <= 0.561\nPnd = {Pnd:.3f} kips. Eq.E4.1-1\n'
        else:
            Pnd = (1 - 0.25 * math.pow(Pcrd / Py, 0.6)) * math.pow(Pcrd / Py, 0.6) * Py
            Rep += f'   lamd > 0.561\nPnd = {Pnd:.3f} kips. Eq.E4.1-2\n'
        # ASD
        Rep += f'_____________\nDesign Strengths:\n'
        omega = 1.80
        Rep += f'omega = {omega}.\nPnd,o = {Pnd / omega:.3f} kips.\n'
        # LRFD
        ff = 0.85
        Rep += f'ff = {ff}.\nff,Pnd = {Pnd * ff:.3f} kips.\n'
        Results = {'ASD': {'oPnd': Pnd / omega},
                   'LRFD': {'ffPnd': ff * Pnd}}
        print(Rep)
        return Results

    # ======================================================================================================================
    # F. MEMBERS IN FLEXURE
    # ======================================================================================================================
    def F211():
        """
        AISI S100-16
        F2.1.1 - Singly or Doubly Symmetric Sections Bending About Symmetric Axis
        :param Section: Gross section properties.
        :param kwargs:
        :return: Fcre - The elastic buckling stress.
        """
        # Eq. F2.1.1-3
        ro = math.sqrt(math.pow(rx, 2) + math.pow(ry, 2) + math.pow(xo, 2))
        # Eq. F2.1.1-4
        sey = math.pow(math.pi, 2) * E / math.pow((Ky * Ly) / ry, 2)
        # Eq. F2.1.1-5
        p1 = 1 / (A * math.pow(ro, 2))
        p2 = G * J
        p3 = math.pow(math.pi, 2) * E * Cw
        p4 = math.pow(Kt * Lt, 2)
        set = p1 * (p2 + p3 / p4)
        # Eq. F2.1.1-1
        Fcre = Cb * ro * A / Wxx * math.sqrt(sey * set)

        Rep = (
            f'{cons.secDivider}\n CALCULATION OF CRITICAL BUCKLING LOAD\n    CHAPTER F. MEMBERS IN FLEXURE\n{cons.secDivider}\n'
            f'The global elastic buckling stress.\nFcre: {Fcre:.3f} ksi. Eq.F2.1.1-1 ')
        print(Rep)
        return Fcre

    def F21(Fcre: float):
        """
        AISI S100-16
        F2.1 - Initiation of Yielding Strength
        :param Section: Section properties.
        :param Fcre: Critical elastic lateral-torsional buckling stress, determined in accordance
        with Section F2.1.1 or Appendix 2.
        :return: Mne, The nominal flexural strength [resistance].
        """
        Rep = f'==== Section F2 Yielding and Global Buckling.\n==== Section F2.1 Initiation of Yielding Strength.\n'

        if Fcre >= 2.78 * fy:
            # Equation F2.1-3
            Fn = fy
            Rep += f'Fcre >= 2.78 x fy\nFn = {Fn:.3f} ksi. Eq.F2.1-3\n'
        elif 0.56 * fy < Fcre < 2.78 * fy:
            # Equation F2.1-4
            Fn = 10.0 / 9.0 * fy * (1 - (10 * fy) / (36 * Fcre))
            Rep += f'0.56 x fy < Fcre < 2.78 x fy\nFn = {Fn:.3f} ksi. Eq.F2.1-4\n'
        else:
            # Equation F2.1-5
            Fn = Fcre
            Rep += f'Fcre > 2.78 x fy\nFn = {Fn:.3f} ksi. Eq.F2.1-5\n'
        # Equation F2.1-2
        My = Wxx * fy
        Rep += f'My = {My:.3f} kip-in. Eq.F2.1-2\n'
        # Equation F2.1-1
        Mne = Wxx * Fn
        if Mne > My:
            Mne = My
        Rep += f'Mne = {Mne:.3f} kip-in. Eq.F2.1-1\n'

        # ASD
        Rep += f'_____________\nDesign Strengths:\n'
        omega = 1.67
        Rep += f'omega = {omega}.\nMne,o = {Mne / omega:.3f} kip-in.\n'
        # LRFD
        ff = 0.90
        Rep += f'ff = {ff}.\nff,Mne = {Mne * ff:.3f} kip-in.\n'
        print(Rep)
        return Mne

    def F32(Mne: float, ratioFlxLocal: float):
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

        # Section F3 Local Buckling Interacting With Yielding and Global Buckling.
        # Section F3.2.1 Members without holes.
        Rep = f'==== Section F3 Local Buckling Interacting With Yielding and Global Buckling.\n==== Section F3.2.1 Members without holes.\n'
        My = Wxx * fy
        Mcrl = ratioFlxLocal * My
        Rep += f'My = {My:.3f} kip-in.\nP/Py = {ratioFlxLocal:.3f}.\nMcrl = {Mcrl:.3f} kip-in.\n'
        # Equation F3.2.1-3
        laml = math.sqrt(Mne / Mcrl)
        Rep += f'laml = {laml:.3f}. Eq.F3.2.1-3\n'
        if laml <= 0.776:
            # Equation F3.2.1-1
            Mnl = Mne
            Rep += f'   laml <= 0.776\nMnl = {Mnl:.3f} kip-in. Eq.F3.2.1-1\n'
        else:
            # Equation F3.2.1-2
            Mnl = (1 - 0.15 * math.pow(Mcrl / Mne, 0.4)) * math.pow(Mcrl / Mne, 0.4) * Mne
            Rep += f'   laml > 0.776\nMnl = {Mnl:.3f} kip-in. Eq.F3.2.1-2\n'

        # ASD
        Rep += f'_____________\nDesign Strengths:\n'
        omega = 1.67
        Rep += f'omega = {omega}.\nMnl,o = {Mnl / omega:.3f} kip-in.\n'
        # LRFD
        ff = 0.90
        Rep += f'ff = {ff}.\nff,Mnl = {Mnl * ff:.3f} kip-in.\n'
        FlexuralStrength = {'ASD': {'oMne': Mne / omega, 'oMnl': Mnl / omega},
                            'LRFD': {'ffMne': ff * Mne, 'ffMnl': ff * Mnl}}
        print(Rep)
        return FlexuralStrength

    def F41(ratioFlxDist: float):
        """
        AISI S100-16
        Section F4 Distortional Buckling
        Section F4.1 Direct Strength Method
        :param Section: Section properties.
        :param ratioFlxDist: Load factor for distortional buckling mode.
        :return: Mnd
        """

        # Section F4 Distortional Buckling.
        # Section F4.1 Members without holes.
        # Equation F4.1-4
        Rep = f'==== Section F4 Distortional Buckling.\n==== Section F4.1 Members without holes.\n'
        My = Wxx * fy
        Fcrd = ratioFlxDist * fy
        Rep += f'My = {My:.3f} kip-in.\nP/Py = {ratioFlxDist:.3f}.\nFcrd = {Fcrd:.3f} ksi.\n'
        # Equation F4.1-5
        Mcrd = Wxx * Fcrd
        Rep += f'Mcrd = {Mcrd:.3f} kip-in.\n'
        # Equation F4.1-3
        lamd = math.sqrt(My / Mcrd)
        Rep += f'lamd = {lamd:.3f}. Eq.F4.1-3\n'
        if lamd <= 0.673:
            # Equation F4.1-1
            Mnd = My
            Rep += f'   lamd <= 0.673\nMnd = {Mnd:.3f} kip-in. Eq.F4.1-1\n'
        else:
            # Equation F4.1-2
            Mnd = (1 - 0.22 * math.pow(Mcrd / My, 0.5)) * math.pow(Mcrd / My, 0.5) * My
            Rep += f'   lamd > 0.673\nMnd = {Mnd:.3f} kip-in. Eq.F4.1-2\n'

        # ASD
        Rep += f'_____________\nDesign Strengths:\n'
        omega = 1.67
        Rep += f'omega = {omega}.\nMnd,o = {Mnd / omega:.3f} kip-in.\n'
        # LRFD
        ff = 0.90
        Rep += f'ff = {ff}.\nff,Mnd = {Mnd * ff:.3f} kip-in.\n'
        FlexuralStrength = {'ASD': {'oMnd': Mnd / omega},
                            'LRFD': {'ffMnd': ff * Mnd}}
        print(Rep)
        return FlexuralStrength


    if Section['case'] == 'Axial':
        Pne = E2(E21(), E22())
        Pnl = E32(Pne, ratio_Local)
        Pnd = E41(Pne, ratio_Distortional)
        print(f'Pne = {Pne}')
        print(f'Pnl = {Pnl}')
        print(f'Pnd = {Pnd}')
        Strength_ASD = min(min(Pnl['ASD']['oPne'], Pnl['ASD']['oPnl']), Pnd['ASD']['oPnd'])
        Strength_LRFD = min(min(Pnl['LRFD']['ffPne'], Pnl['LRFD']['ffPnl']), Pnd['LRFD']['ffPnd'])
        print(f'Strength_ASD = {Strength_ASD}')
        print(f'Strength_LRFD = {Strength_LRFD}')
    else:
        Fcre = F211()
        Mne = F21(Fcre)
        Mnl = F32(Mne, ratio_Local)
        Mnd = F41(ratio_Distortional)

        print(f'Mnl = {Mnl}')
        print(f'Mnd = {Mnd}')
        Strength_ASD = min(min(Mnl['ASD']['oMne'], Mnl['ASD']['oMnl']), Mnd['ASD']['oMnd'])
        Strength_LRFD = min(min(Mnl['LRFD']['ffMne'], Mnl['LRFD']['ffMnl']), Mnd['LRFD']['ffMnd'])
        print(f'Strength_ASD = {Strength_ASD}')
        print(f'Strength_LRFD = {Strength_LRFD}')

    Results = {'Strength_ASD':Strength_ASD,
               'Strength_LRFD': Strength_LRFD}
    return Results


