from typing import Literal
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
import AISI_Functions.DirectStrengthMethod as strength


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

def section_input(**kwargs):
    '''
    This function calculates the member strength as per AISI using Direct Strength Method.
    :param kwargs:
    :return:
    '''
    A = kwargs['A']
    B = kwargs['B']
    C = kwargs['C']
    t = kwargs['t']
    R = kwargs['R']
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


def material_input(**kwargs):
    fy = kwargs['fy']
    # Define the material
    material = mat.Material(fy)
    return material


def member_input(**kwargs):
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    Lt = kwargs['Lt']
    Kx = kwargs['Kx']
    Ky = kwargs['Ky']
    Kt = kwargs['Kt']
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
member = member_input(Lx=110.0, Ly=110.0, Lt=110.0, Kx=1.0, Ky=1.0, Kt=1.0, support='S-S')

# Material
# -------------------------------------------------------
material = material_input(fy=33.0)

# Sections
# _______________________________________________________
C_Axial = section_input(A=9.0, B=2.5, C=1.625, t=0.075, R=0.1870, ang=0, case='Axial')
C_ang0_Flex = section_input(A=9.0, B=2.5, C=1.625, t=0.075, R=0.1870, ang=0, case='Flexural')
C_ang90_Flex = section_input(A=9.0, B=2.5, C=1.625, t=0.075, R=0.1870, ang=90, case='Flexural')
C_ang270_Flex = section_input(A=9.0, B=2.5, C=1.625, t=0.075, R=0.1870, ang=270, case='Flexural')
# List for iteration for all the cases:
Analysis_Cases = [C_Axial, C_ang0_Flex, C_ang90_Flex, C_ang270_Flex]


# ======================================================================================================================
# PERFORM THE FINITE STRIP ANALYSIS
# ======================================================================================================================
def C_sign_solver(Section, Material, Member) -> Dict[str, np.ndarray]:
    '''

    :param Section: C_####['section']
    :param Material: C_####['material']
    :param Member: member
    :return:
    '''
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

    return {
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


# ======================================================================================================================
# FIND THE MINIMAS / CRITICAL LOAD RATIOS
# ======================================================================================================================
def plot_Sign_Curve(Section, plot: bool):
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

    # Find the index of the item
    h_index = np.where(lengths == RefLen)[0][0]
    h_value = Y_Values[h_index]
    h_text = f'P/Py: {h_value:.3f}\nL: {RefLen}'
    minimas.append([h_index, RefLen, h_value])
    # Annotating the minima values
    ax1.annotate(h_text, xy=(RefLen, h_value),
                 xytext=(RefLen * 0.3, h_value * 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)
    # Setting the plot for the signature curve.
    ax1.plot(X_Values, Y_Values, linewidth=2.0)
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
        ax2.axes.annotate(f'{id_nodes[i] + 1:.0f}', xy=(x_nodes[i] * 1.03, y_nodes[i]), xycoords='data', fontsize=8)
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
    return minimas


# ======================================================================================================================
# EXPORT A REPORT
# ======================================================================================================================
def export_report(Section, minimas):
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


# ======================================================================================================================
# OUTPUT FOR BUCKLING ANALYSIS
# ======================================================================================================================
# Creation of a member to solve
C1 = C_sign_solver(C_Axial, material, member)
C2 = C_sign_solver(C_ang0_Flex, material, member)
C3 = C_sign_solver(C_ang90_Flex, material, member)
C4 = C_sign_solver(C_ang270_Flex, material, member)
# Creation of graph if True plot will be shown
pC1 = plot_Sign_Curve(C1, True)
pC2 = plot_Sign_Curve(C2, True)
pC3 = plot_Sign_Curve(C3, True)
pC4 = plot_Sign_Curve(C4, True)
# Print minimas
print(pC1)
print(pC2)
print(pC3)
print(pC4)
# Export the report. (Section definition, Curve)
export_report(C1, pC1)
export_report(C2, pC2)
export_report(C3, pC3)
export_report(C4, pC4)

# ======================================================================================================================
# OUTPUT FOR DIRECT STRENGTH METHOD IN AISI
# ======================================================================================================================
DSM1 = strength.stregnths(pC1, material, C_Axial, member)
DSM2 = strength.stregnths(pC2, material, C_ang0_Flex, member)
DSM3 = strength.stregnths(pC3, material, C_ang90_Flex, member)
DSM4 = strength.stregnths(pC4, material, C_ang270_Flex, member)
print(DSM1)
print(DSM2)
print(DSM3)
print(DSM4)
