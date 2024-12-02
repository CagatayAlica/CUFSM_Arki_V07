from typing import Literal
import math
from typing import Dict
import pandas as pd
import numpy as np
import Input.CreateSection as sec
import Input.Material as mat
import Input.Member as mem

from pycufsm.CUFSM_Functions.fsm import strip
from pycufsm.CUFSM_Functions.preprocess import stress_gen
from pycufsm.CUFSM_Functions.types import GBT_Con, Sect_Props
import matplotlib.pyplot as plt
from pycufsm.SectionProps.sectionDraw import lengthRange

import Constants.Constants as cons


class MainInput:
    def __init__(self, **kwargs):
        self.A = kwargs['A']
        self.B = kwargs['B']
        self.C = kwargs['C']
        self.t = kwargs['t']
        self.R = kwargs['R']
        self.Lx = kwargs['Lx']
        self.Ly = kwargs['Ly']
        self.Lt = kwargs['Lt']
        self.Kx = kwargs['Kx']
        self.Ky = kwargs['Ky']
        self.Kt = kwargs['Kt']
        self.fy = kwargs['fy']

        # ======================================================================================================================
        # EXPLANATION OF INPUT TERMS
        # ======================================================================================================================
        # C_sign_solver(A, B, C, t, angle, Fy, Case, MemLength)
        # Units [in, ksi]
        # A : Web height.
        # B : Flange width.
        # C : Lip length.
        # t : Steel thickness.
        # R : Inner radius.
        # angle : Orientation of the section.
        #                 "0": """
        #                        ┌-┐
        #                        |
        #                        └-┘
        #                        """,
        #                "270": """
        #                        ┌   ┐
        #                        └---┘
        #                        """,
        #                "90": """
        #                        ┌---┐
        #                        └   ┘
        #                        """
        # Fy : Steel yield stress.
        # Case : 'Axial' for uniform axial compression.
        #           'Flexural' for bending creating compression at top fiber.
        # MemLength : Total member length
        # ======================================================================================================================
        def section_dimensions():
            A = self.A
            B = self.B
            C = self.C
            t = self.t
            R = self.R
            section_dim = {'A': A,
                           'B': B,
                           'C': C,
                           't': t,
                           'R': R}
            return section_dim

        def section_input(**kwargs):
            """
            This function calculates the member strength as per AISI using Direct Strength Method.
            :param kwargs:
            :return:
            """
            Section_dims = kwargs['Section_Dimensions']
            A = Section_dims['A']
            B = Section_dims['B']
            C = Section_dims['C']
            t = Section_dims['t']
            R = Section_dims['R']
            Angle = kwargs['ang']
            Analysis_case: Literal['Axial', 'Flexural'] = kwargs['case']

            # Creating a section and calculates the nodes and elements.
            section = sec.C_Section(A=A, B=B, C=C, t=t, R=R, angle=Angle)
            # Calculate the gross-section properties
            gross = sec.GrossProps(section.nodes[:, 1], section.nodes[:, 2], section.t, section.r)
            # Define an analysis case
            case = Analysis_case

            # Results in dictionary
            section_dict = {'section': section,
                            'gross': gross,
                            'case': case}

            return section_dict

        def material_input():
            fy = self.fy
            # Define the material
            material = mat.Material(fy)
            return material

        def member_input(**kwargs):
            Lx = self.Lx
            Ly = self.Ly
            Lt = self.Lt
            Kx = self.Kx
            Ky = self.Ky
            Kt = self.Kt
            Support: Literal["S-S", "C-C", "S-C", "C-F", "C-G"] = kwargs['support']
            # Define a member
            defined_member = mem.Member(Lx=Lx, Ly=Ly, Lt=Lt,
                                        Kx=Kx, Ky=Ky, Kt=Kt,
                                        support=Support)
            return defined_member

        # ======================================================================================================================
        # MAIN DEFINITIONS
        # ======================================================================================================================
        # Member
        # -------------------------------------------------------
        member = member_input(support='S-S')

        # Material
        # -------------------------------------------------------
        material = material_input()

        # Sections
        # -------------------------------------------------------
        Section_Shape = section_dimensions()
        C_Axial = section_input(Section_Dimensions=Section_Shape, ang=0, case='Axial')
        C_ang0_Flex = section_input(Section_Dimensions=Section_Shape, ang=0, case='Flexural')
        C_ang90_Flex = section_input(Section_Dimensions=Section_Shape, ang=90, case='Flexural')
        C_ang270_Flex = section_input(Section_Dimensions=Section_Shape, ang=270, case='Flexural')

        self.Input_Dict = {'Section': Section_Shape,
                           'Gross': C_Axial['gross'],
                           'Material': material,
                           'Member': member,
                           'C_Axial': C_Axial,
                           'C_ang0_Flex': C_ang0_Flex,
                           'C_ang90_Flex': C_ang90_Flex,
                           'C_ang270_Flex': C_ang270_Flex}


class BucklingAnalysis:
    def __init__(self, Section, Material, Member):
        """

                    :param Section: C_####['section']
                    :param Material: C_####['material']
                    :param Member: member
                    :return:
                    """
        # Define an isotropic material with E = 29,500 ksi and nu = 0.3
        E = Material.E
        nu = Material.v
        props = np.array([np.array([0, E, E, nu, nu, E / (2 * (1 + 0.3))])])
        # Steel yield stress
        fy = Material.fy  # ksi
        # Nodes IDs for strips
        nodes = Section['section'].nodes
        # Elements IDs for strips
        elements = Section['section'].elements
        # Steel thickness
        thickness = Section['section'].t
        # Section name
        descp = Section['section'].descp_rep
        # Analysis case
        case = Section['case']
        # Section orientation
        angle = Section['section'].angle
        orientationShape = Section['section'].ang_shape
        # Calculation the gross section properties
        properties = Section['gross']

        # These lengths will generally provide sufficient accuracy for
        # local, distortional, and global buckling modes
        # Length units are inches
        ReferenceLength = Member.Lx  # inches
        lengths = Member.lengths_data

        flag = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # No special springs or constraints
        springs = np.array([])
        constraints = np.array([])

        # Values here correspond to signature curve basis and orthogonal based upon geometry
        gbt_con: GBT_Con = {
            'glob': [0],
            'dist': [0],
            'local': [0],
            'other': [0],
            'o_space': 1,
            'couple': 1,
            'orth': 2,
            'norm': 0,
        }

        # Simply-supported boundary conditions
        b_c = Member.support

        # For signature curve analysis, only a single array of ones makes sense here
        m_all = np.ones((len(lengths), 1))

        # Solve for 10 eigenvalues
        n_eigs = 12
        # Set the section properties
        sect_props: Sect_Props = {
            'cx': properties.cx,
            'cy': properties.cy,
            'x0': properties.xsc,
            'y0': properties.ysc,
            'phi': 0,
            'A': properties.Ar,
            'Ixx': properties.Ix,
            'Ixy': properties.Ixy,
            'Iyy': properties.Iy,
            'I11': properties.Ix,
            'I22': properties.Iy,
            'Cw': properties.Cw,
            'J': properties.It,
            'B1': 0,
            'B2': 0,
            'wn': np.array([])
        }

        # Generate the stress points
        if case == 'Axial':
            nodes_p = stress_gen(
                nodes=nodes,
                forces={
                    'P': fy * sect_props['A'],
                    'Mxx': 0,  # fy * sect_props['Ixx'] / sect_props['cy'],
                    'Myy': 0,
                    'M11': 0,
                    'M22': 0,
                    'restrain': False,
                    'offset': [-thickness / 2, -thickness / 2]
                },
                sect_props=sect_props,
            )
        else:
            nodes_p = stress_gen(
                nodes=nodes,
                forces={
                    'P': 0,  # fy * sect_props['A'],
                    'Mxx': fy * sect_props['Ixx'] / sect_props['cy'],
                    'Myy': 0,
                    'M11': 0,
                    'M22': 0,
                    'restrain': False,
                    'offset': [-thickness / 2, -thickness / 2]
                },
                sect_props=sect_props,
            )

        # Perform the Finite Strip Method analysis
        signature, curve, shapes = strip(
            props=props,
            nodes=nodes_p,
            elements=elements,
            lengths=lengths,
            springs=springs,
            constraints=constraints,
            gbt_con=gbt_con,
            b_c=b_c,
            m_all=m_all,
            n_eigs=n_eigs,
            sect_props=sect_props
        )

        signature = np.array(signature)
        curves = np.array(curve)
        shapes = np.array(shapes)
        curve = np.zeros((len(lengths), n_eigs, 2))
        for j in range(len(lengths)):
            for i in range(n_eigs):
                curve[j, i, 0] = lengths[j]
                curve[j, i, 1] = curves[j, i]

        self.BucklingAnalysis = {
            'curve': curve,
            'nodes': nodes,
            'shapes': shapes,
            'thickness': thickness,
            'elements': elements,
            'flag': flag,
            'springs': springs,
            'BC': b_c,
            'constraints': constraints,
            'X_values': lengths,
            'Y_values': signature,
            'Y_values_allmodes': curve,
            'Orig_coords': nodes_p,
            'Deformations': shapes,
            'Reference_Length': ReferenceLength,
            'Section_Def': descp,
            'Yield_stress': fy,
            'Case': case,
            'Angle': angle,
            'orientationShape': orientationShape,
            'Sect_Props': properties
        }


class PlotSignatureCurve:
    def __init__(self, Section, plot: bool):
        # Inputs:
        sign_nodes = Section['nodes']
        thk = Section['thickness']
        X_Values = Section['X_values']
        Y_Values = Section['Y_values']
        RefLen = Section['Reference_Length']
        descp = Section['Section_Def']
        fy = Section['Yield_stress']
        case = Section['Case']
        angle = Section['Angle']

        lengths = lengthRange(RefLen, "imperial")
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2)
        minimas = []
        fig.suptitle(f'Signature Curve\n{case} case\nfy: {fy:.2f} ksi')
        # Finding the minima points
        for loadFactor in range(2, len(Y_Values)):
            if Y_Values[loadFactor - 1] < Y_Values[loadFactor - 2] and Y_Values[
                loadFactor - 1] < \
                    Y_Values[loadFactor]:
                text = f"P/Py: {Y_Values[loadFactor - 1]:.3f} \nL: {X_Values[loadFactor - 1]}"
                # minimas = [index, length, loadfactor]
                minimas.append([np.where(lengths == X_Values[loadFactor - 1])[0][0], X_Values[loadFactor - 1],
                                Y_Values[loadFactor - 1]])
                # Annotating the minima values
                ax1.annotate(text, xy=(X_Values[loadFactor - 1], Y_Values[loadFactor - 1]),
                             xytext=(X_Values[loadFactor - 1] * 0.3, Y_Values[loadFactor - 1] * 0.3),
                             arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)
        # Setting the plot for the signature curve.
        ax1.plot(X_Values, Y_Values, linewidth=2.0)

        # Drawing a global buckling curve
        # Find the index of the item
        h_index = np.where(lengths == RefLen)[0][0]
        h_value = Y_Values[h_index]
        h_text = f'P/Py: {h_value:.3f}\nL: {RefLen}'
        minimas.append([h_index, RefLen, h_value])
        # Annotating the minima values
        ax1.annotate(h_text, xy=(RefLen, h_value),
                     xytext=(RefLen * 0.3, h_value * 0.3),
                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)

        # print(signa['curve'][:, 1][:, 1])
        # print(minimas)
        # Formatting the signature curve plot.
        ax1.axis(ymin=0.0, ymax=np.min([np.max(Y_Values), 3 * np.median(Y_Values)]))
        ax1.grid(color='b', linestyle='-', linewidth=0.2)
        ax1.axes.set_xscale("log")
        ax1.axes.set_xlabel('length [in]')
        ax1.axes.set_ylabel('load factor [P/Py]')
        ax1.axes.set_title('Buckling curve')
        # Drawing the cross section shape
        # IDs of the nodes.
        id_nodes = sign_nodes[:, 0]
        # X values of the nodes.
        x_nodes = sign_nodes[:, 1]
        # Y values of the nodes.
        y_nodes = sign_nodes[:, 2]
        # Thickness
        thk = thk
        for i in range(len(x_nodes)):
            ax2.axes.annotate(f'{id_nodes[i] + 1:.0f}', xy=(x_nodes[i] * 1.03, y_nodes[i]), xycoords='data',
                              fontsize=8)
        ax2.plot(x_nodes, y_nodes, linewidth=thk * 30, color='green', marker="o", markersize=2)
        ax2.axis('equal')
        if angle == 0:
            midX = (min(x_nodes) + max(x_nodes)) / 2.0
            midY = (min(y_nodes) + max(y_nodes)) / 2.0
            ax2.text(1.01 * midX, midY, descp, ha='left', rotation=0, wrap=True,
                     bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        elif angle == 90:
            midX = (min(x_nodes) + max(x_nodes)) / 2.0
            midY = (min(y_nodes) + max(y_nodes)) / 2.0
            ax2.text(midX, -1.05 * midY, descp, ha='center', rotation=0, wrap=True,
                     bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        else:
            midX = (min(x_nodes) + max(x_nodes)) / 2.0
            midY = (min(y_nodes) + max(y_nodes)) / 2.0
            ax2.text(midX, 1.05 * midY, descp, ha='center', rotation=0, wrap=True,
                     bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

        ax2.axes.set_xlabel('length [in]')
        ax2.axes.set_ylabel('length [in]')
        ax2.axes.set_title('Cross Section')
        # Show the plot
        if plot:
            plt.show()
        self.MinimasList = minimas


class FSMA_Report:
    def __init__(self, Section, minimas):
        # Inputs:
        nodes = Section['nodes']
        elements = Section['elements']
        Boundary = Section['BC']
        RefLen = Section['Reference_Length']
        descp = Section['Section_Def']
        fy = Section['Yield_stress']
        case = Section['Case']
        lengthsData = lengthRange(RefLen, "imperial")
        GrossData = Section['Sect_Props']
        angle = Section['Angle']
        ang_shape = Section['orientationShape']

        # Create a dataframe for nodes
        dfNodes = []
        for i in nodes:
            dfNodes.append([i[0] + 1, i[1], i[2]])
        Nodes = pd.DataFrame(dfNodes, columns=['Node', 'X', 'Y'])

        # Create a dataframe for elements
        dfElements = []
        for i in elements:
            dfElements.append([i[0] + 1, i[1], i[2], i[3]])
        Elements = pd.DataFrame(dfElements, columns=['Element', 'iNode', 'jNode', 'Thickness'])

        # Gross section properties
        dfGross = []
        for key, value in GrossData.propDict.items():
            dfGross.append([key, value])
        Gross = pd.DataFrame(dfGross, columns=['Type', 'Value / Unit'])

        # DataFrame for halfwave lengths
        Lengths = pd.DataFrame(lengthsData, columns=['Length [in]'])

        # Create a dataframe for critical buckling length
        dfMinima = []
        for i in minimas:
            dfMinima.append([i[1], i[2]])
        Minima = pd.DataFrame(dfMinima, columns=['Critical Length [in]', 'P/Py'])

        # ==== CREATE A REPORT ====
        Rep = (
            f'{cons.secDivider}\n CALCULATION OF CRITICAL BUCKLING LOAD\n         USING SIGNATURE CURVE\n{cons.secDivider}\n'
            f'Units are in Imperial [in, ksi]\n'
            f'{descp}\n'
            f'Steel yield stress:\n{cons.sp3}Fy: {fy:.3f} ksi\n'
            f'Member length:\n{cons.sp3}L: {RefLen:.3f} in\n'
            f'Orientation:\n'
            f'{cons.sp3}Angle: {angle}\n{ang_shape}'
            f'Check case:\n'
            f'{cons.sp3}Case: {case}\n'
            f'Boundary Condition:\n'
            f'{cons.sp3}Boundary: {Boundary}\n'
            f'Sectional nodes for center line:\n{Nodes}\n'
            f'Sectional elements:\n{Elements}\n'
            f'Gross section properties:\n{Gross}\n'
            f'Reference lengths:\n'
            f'{Lengths}\n'
            f' ==== \nCritical load ratios:\n'
            f'{Minima}\n ==== \n')
        print(Rep)


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
        self.h = Section['section'].a
        self.b = Section['section'].b
        self.t = Section['section'].t
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
    def __init__(self, Material, Section, Member):
        # ==== Input ====
        self.Pne = None
        self.Py = None
        self.Cb = 1.67
        self.E = Material.E
        self.G = Material.G
        self.fy = Material.fy
        self.Lx = Member.Lx
        self.Kx = Member.Kx
        self.Ly = Member.Ly
        self.Ky = Member.Ky
        self.Lt = Member.Lt
        self.Kt = Member.Kt
        self.cx = Section['gross'].cx
        self.cy = Section['gross'].cy
        self.xo = Section['gross'].xsc
        self.y0 = Section['gross'].ysc
        self.A = Section['gross'].Ar
        self.Ixx = Section['gross'].Ix
        self.Wxx = Section['gross'].Wx
        self.Ixy = Section['gross'].Ixy
        self.Iyy = Section['gross'].Iy
        self.Wyy = Section['gross'].Wy
        self.I11 = Section['gross'].Ix
        self.I22 = Section['gross'].Iy
        self.Cw = Section['gross'].Cw
        self.J = Section['gross'].It
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
    def __init__(self, Material, Section, Member):
        # ==== Input ====
        self.Mne = None
        self.Fcre = None
        self.My = None
        self.Cb = 1.67
        self.E = Material.E
        self.G = Material.G
        self.fy = Material.fy
        self.Lx = Member.Lx
        self.Kx = Member.Kx
        self.Ly = Member.Ly
        self.Ky = Member.Ky
        self.Lt = Member.Lt
        self.Kt = Member.Kt
        self.cx = Section['gross'].cx
        self.cy = Section['gross'].cy
        self.xo = Section['gross'].xsc
        self.y0 = Section['gross'].ysc
        self.A = Section['gross'].Ar
        self.Ixx = Section['gross'].Ix
        self.Wxx = Section['gross'].Wx
        self.Ixy = Section['gross'].Ixy
        self.Iyy = Section['gross'].Iy
        self.Wyy = Section['gross'].Wy
        self.I11 = Section['gross'].Ix
        self.I22 = Section['gross'].Iy
        self.Cw = Section['gross'].Cw
        self.J = Section['gross'].It
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
