import math
import Constants.Constants as cons
import Input.Definitions as Inp
import Solver.BucklingAnalysis as buckle


class DSM_Strengths:
    def __init__(self, Minimas, Case, Material, Gross, Member):
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
        cx = Gross.cx
        cy = Gross.cy
        xo = Gross.xsc
        y0 = Gross.ysc
        A = Gross.Ar
        Ixx = Gross.Ix
        Wxx = Gross.Wx
        Ixy = Gross.Ixy
        Iyy = Gross.Iy
        Wyy = Gross.Wy
        I11 = Gross.Ix
        I22 = Gross.Iy
        Cw = Gross.Cw
        J = Gross.It
        # ====   ====
        rx = math.sqrt(Ixx / A)
        ry = math.sqrt(Iyy / A)

        # Finding the minimas
        Critical_Length_Distortional = None
        Critical_Length_Local = None
        Critical_Length_Global = None
        ratio_Local = None
        ratio_Distortional = None

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
        if len(Minimas) > 2:
            # First minima for local buckling case.
            Critical_Length_Local = Minimas[0][0]
            ratio_Local = Minimas[0][2]
            # Second minima for distortional buckling case.
            Critical_Length_Distortional = Minimas[1][0]
            ratio_Distortional = Minimas[1][2]

        # ==================================================================================================================
        # E. MEMBERS IN COMPRESSION
        # ==================================================================================================================
        def E21():
            # Fcre, flexural buckling stress.
            # Eq. E2.1-1
            Fcre = min(math.pow(cons.PI, 2) * E / math.pow(Kx * Lx / rx, 2),
                       math.pow(cons.PI, 2) * E / math.pow(Ky * Ly / ry, 2))
            Rep = (
                f'{cons.secDivider}\n CALCULATION OF CRITICAL BUCKLING LOAD\n    CHAPTER E. MEMBERS IN COMPRESSION\n{cons.secDivider}\n'
                f'The global elastic buckling stress.\nFcre: {Fcre:.3f} ksi. Eq.E2.1-1')
            print(Rep)
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
            Rep += f'laml = {laml:.3f} Eq.E3.2.1-3\n'
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

        # ==================================================================================================================
        # F. MEMBERS IN FLEXURE
        # ==================================================================================================================
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

        # ==================================================================================================================
        # OUTPUT
        # ==================================================================================================================
        if Case.Analysis_case == 'Axial':
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

        self.Results = {'Strength_ASD': Strength_ASD,
                        'Strength_LRFD': Strength_LRFD}


Str_Axial_0 = DSM_Strengths(buckle.Signa_Axial_0.MinimasList, Inp.Case_Axial, Inp.Material_1, Inp.Gross_ang0,
                            Inp.Member_1)
Str_Flx_0 = DSM_Strengths(buckle.Signa_Flx_0.MinimasList, Inp.Case_Flexural, Inp.Material_1, Inp.Gross_ang0,
                          Inp.Member_1)
Str_Flx_90 = DSM_Strengths(buckle.Signa_Flx_90.MinimasList, Inp.Case_Flexural, Inp.Material_1, Inp.Gross_ang90,
                           Inp.Member_1)
Str_Flx_270 = DSM_Strengths(buckle.Signa_Flx_270.MinimasList, Inp.Case_Flexural, Inp.Material_1, Inp.Gross_ang270,
                            Inp.Member_1)

Curve_Axial_0 = [buckle.Signa_Axial_0.signaCurve_X, buckle.Signa_Axial_0.signaCurve_Y]
Curve_Flx_0 = [buckle.Signa_Flx_0.signaCurve_X, buckle.Signa_Flx_0.signaCurve_Y]
Curve_Flx_90 = [buckle.Signa_Flx_90.signaCurve_X, buckle.Signa_Flx_90.signaCurve_Y]
Curve_Flx_270 = [buckle.Signa_Flx_270.signaCurve_X, buckle.Signa_Flx_270.signaCurve_Y]

class Tension:
    def __init__(self, Material, Gross):
        self.Ag = Gross.Ar
        self.fy = Material.fy
        self.omega = 1.67
        self.ff = 0.90

    def strength(self):
        Tn = self.Ag * self.fy
        Tno = Tn / self.omega
        ffTn = Tn * self.ff
        Strength = {'Tn': Tn, 'Tno': Tno, 'ffTn': ffTn}
        return Strength


class Shear:
    def __init__(self, Material, Section):
        self.Vy = None
        self.Vcr = None
        self.Fcr = None
        self.Vn = None
        self.ffVn = None
        self.Vno = None
        self.omega = 1.60
        self.ff = 0.95
        self.h = Section.a
        self.b = Section.b
        self.t = Section.t
        self.fy = Material.fy
        self.E = Material.E
        self.v = Material.v
        self.kv = 5.34

    def strengthStrong(self):
        Aw = self.h * self.t
        Fcr = (math.pow(math.pi, 2) * self.E * self.kv) / (
                12 * (1 - math.pow(self.v, 2)) * math.pow(self.h / self.t, 2))
        Vcr = Aw * Fcr
        Vy = 0.6 * Aw * self.fy
        lamv = math.sqrt(Vy / Vcr)
        if lamv <= 0.815:
            Vn = Vy
        elif 0.815 < lamv <= 1.227:
            Vn = 0.815 * math.sqrt(Vcr * Vy)
        else:
            Vn = Vcr
        Vno = Vn / self.omega
        ffVn = Vn * self.ff
        StrengthStrong = {'Vn': Vn, 'Vno': Vno, 'ffVn': ffVn}
        return StrengthStrong

    def strengthWeak(self):
        Aw = self.b * self.t
        Fcr = (math.pow(math.pi, 2) * self.E * self.kv) / (
                12 * (1 - math.pow(self.v, 2)) * math.pow(self.h / self.t, 2))
        Vcr = Aw * Fcr
        Vy = 0.6 * Aw * self.fy
        lamv = math.sqrt(Vy / Vcr)
        if lamv <= 0.815:
            Vn = 2 * Vy
        elif 0.815 < lamv <= 1.227:
            Vn = 2 * 0.815 * math.sqrt(Vcr * Vy)
        else:
            Vn = 2 * Vcr
        Vno = Vn / self.omega
        ffVn = Vn * self.ff
        StrengthWeak = {'Vn': Vn, 'Vno': Vno, 'ffVn': ffVn}
        return StrengthWeak


class Compression:
    def __init__(self, Material, Gross, Lx: float, Ly: float, Lt: float, Kx: float, Ky: float, Kt: float):
        # ==== Input ====
        self.Pne = None
        self.Py = None
        self.Cb = 1.67
        self.E = Material.E
        self.G = Material.G
        self.fy = Material.fy
        self.Lx = Lx
        self.Kx = Kx
        self.Ly = Ly
        self.Ky = Ky
        self.Lt = Lt
        self.Kt = Kt
        self.cx = Gross.cx
        self.cy = Gross.cy
        self.xo = Gross.xsc
        self.y0 = Gross.ysc
        self.A = Gross.Ar
        self.Ixx = Gross.Ix
        self.Wxx = Gross.Wx
        self.Ixy = Gross.Ixy
        self.Iyy = Gross.Iy
        self.Wyy = Gross.Wy
        self.I11 = Gross.Ix
        self.I22 = Gross.Iy
        self.Cw = Gross.Cw
        self.J = Gross.It
        # ====   ====
        self.rx = math.sqrt(self.Ixx / self.A)
        self.ry = math.sqrt(self.Iyy / self.A)
        # ====   ====
        self.omega = 1.80
        self.ff = 0.85

        # ==================================================================================================================
        # E. MEMBERS IN COMPRESSION
        # ==================================================================================================================

    def E21(self, Lxi, Lyi):
        # Fcre, flexural buckling stress.
        # Eq. E2.1-1
        Fcre = min(math.pow(cons.PI, 2) * self.E / math.pow(self.Kx * Lxi / self.rx, 2),
                   math.pow(cons.PI, 2) * self.E / math.pow(self.Ky * Lyi / self.ry, 2))
        return Fcre

    def E22(self, Lxi, Lti):
        # Eq. E2.2-4
        ro = math.sqrt(math.pow(self.rx, 2) + math.pow(self.ry, 2) + math.pow(self.xo, 2))
        # Eq. E2.2-3
        beta = 1 - math.pow(self.xo / ro, 2)
        # Eq. E2.2-6
        sex = math.pow(cons.PI, 2) * self.E / math.pow(self.Kx * Lxi / self.rx, 2)
        # Eq. E2.2-5
        p1 = 1 / (self.A * math.pow(ro, 2))
        p2 = self.G * self.J
        p3 = math.pow(cons.PI, 2) * self.E * self.Cw
        p4 = math.pow(self.Kt * Lti, 2)
        set = p1 * (p2 + p3 / p4)
        # Eq. E2.2-1
        m1 = 1 / (2 * beta)
        m2 = sex + set
        m3 = math.pow(sex + set, 2)
        m4 = 4 * beta * sex * set
        Fcre = m1 * (m2 - math.sqrt(m3 - m4))
        return Fcre

    def E2(self, Fcrexy, Fcret):
        # Eq. E2-4
        Fcre = min(Fcrexy, Fcret)
        lamc = math.sqrt(self.fy / Fcre)
        if lamc <= 1.5:
            # Eq. E2-2
            Fn = math.pow(0.658, math.pow(lamc, 2)) * self.fy
        else:
            # Eq. E2-3
            Fn = (0.877 / math.pow(lamc, 2)) * self.fy
        # Eq. E2-1
        Pne = self.A * Fn
        return Pne

    # ==================================================================================================================
    # OUTPUT
    # ==================================================================================================================
    def strength(self):
        self.Py = self.A * self.fy
        self.Pne = self.E2(self.E21(self.Lx, self.Ly), self.E22(self.Lx, self.Lt))
        Pno = self.Pne / self.omega
        ffPn = self.Pne * self.ff
        Strength = {'Pne': self.Pne, 'Pno': Pno, 'ffPn': ffPn}
        return Strength


class Flexure:
    def __init__(self, Material, Gross, Lx: float, Ly: float, Lt: float, Kx: float, Ky: float, Kt: float):
        # ==== Input ====
        self.Mne = None
        self.Fcre = None
        self.My = None
        self.Cb = 1.67
        self.E = Material.E
        self.G = Material.G
        self.fy = Material.fy
        self.Lx = Lx
        self.Kx = Kx
        self.Ly = Ly
        self.Ky = Ky
        self.Lt = Lt
        self.Kt = Kt
        self.cx = Gross.cx
        self.cy = Gross.cy
        self.xo = Gross.xsc
        self.y0 = Gross.ysc
        self.A = Gross.Ar
        self.Ixx = Gross.Ix
        self.Wxx = Gross.Wx
        self.Ixy = Gross.Ixy
        self.Iyy = Gross.Iy
        self.Wyy = Gross.Wy
        self.I11 = Gross.Ix
        self.I22 = Gross.Iy
        self.Cw = Gross.Cw
        self.J = Gross.It
        # ====   ====
        self.rx = math.sqrt(self.Ixx / self.A)
        self.ry = math.sqrt(self.Iyy / self.A)
        # ====   ====
        self.omega = 1.80
        self.ff = 0.85

        # ==================================================================================================================
        # F. MEMBERS IN FLEXURE
        # ==================================================================================================================

    def F211(self, Lyi, Lti):
        """
            AISI S100-16
            F2.1.1 - Singly or Doubly Symmetric Sections Bending About Symmetric Axis
            :param Section: Gross section properties.
            :param kwargs:
            :return: Fcre - The elastic buckling stress.
            """
        # Eq. F2.1.1-3
        ro = math.sqrt(math.pow(self.rx, 2) + math.pow(self.ry, 2) + math.pow(self.xo, 2))
        # Eq. F2.1.1-4
        sey = math.pow(math.pi, 2) * self.E / math.pow((self.Ky * Lyi) / self.ry, 2)
        # Eq. F2.1.1-5
        p1 = 1 / (self.A * math.pow(ro, 2))
        p2 = self.G * self.J
        p3 = math.pow(math.pi, 2) * self.E * self.Cw
        p4 = math.pow(self.Kt * Lti, 2)
        set = p1 * (p2 + p3 / p4)
        # Eq. F2.1.1-1
        Fcre = self.Cb * ro * self.A / self.Wxx * math.sqrt(sey * set)
        return Fcre

    def F21(self, Fcre: float):
        """
            AISI S100-16
            F2.1 - Initiation of Yielding Strength
            :param Section: Section properties.
            :param Fcre: Critical elastic lateral-torsional buckling stress, determined in accordance
            with Section F2.1.1 or Appendix 2.
            :return: Mne, The nominal flexural strength [resistance].
            """
        if Fcre >= 2.78 * self.fy:
            # Equation F2.1-3
            Fn = self.fy
        elif 0.56 * self.fy < Fcre < 2.78 * self.fy:
            # Equation F2.1-4
            Fn = 10.0 / 9.0 * self.fy * (1 - (10 * self.fy) / (36 * Fcre))
        else:
            # Equation F2.1-5
            Fn = Fcre
        # Equation F2.1-2
        My = self.Wxx * self.fy
        # Equation F2.1-1
        Mne = self.Wxx * Fn
        if Mne > My:
            Mne = My

        return Mne

    # ==================================================================================================================
    # OUTPUT
    # ==================================================================================================================
    def strength(self):
        self.My = self.Wxx * self.fy
        self.Fcre = self.F211(self.Ly, self.Lt)
        self.Mne = self.F21(self.Fcre)
        Mno = self.Mne / self.omega
        ffMn = self.Mne * self.ff
        Strength = {'Mne': self.Mne, 'Mno': Mno, 'ffMn': ffMn}
        return Strength
