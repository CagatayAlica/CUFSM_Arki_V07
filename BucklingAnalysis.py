from typing import Dict
import numpy as np
from pycufsm.CUFSM_Functions.fsm import strip
from pycufsm.CUFSM_Functions.preprocess import stress_gen
from pycufsm.CUFSM_Functions.types import GBT_Con, Sect_Props
import matplotlib.pyplot as plt
from pycufsm.SectionProps.sectionDraw import lengthRange
import pandas as pd
import Constants.Constants as cons
import Definitions as defin


# ======================================================================================================================
# PERFORM THE FINITE STRIP ANALYSIS
# ======================================================================================================================
def C_sign_solver() -> Dict[str, np.ndarray]:
    # Define an isotropic material with E = 29,500 ksi and nu = 0.3
    E = defin.material.E
    nu = defin.material.v
    props = np.array([np.array([0, E, E, nu, nu, E / (2 * (1 + 0.3))])])
    # Steel yield stress
    fy = defin.material.fy  # ksi
    # Nodes IDs for strips
    nodes = defin.section.nodes
    # Elements IDs for strips
    elements = defin.section.elements
    # Steel thickness
    thickness = defin.section.t
    # Section name
    descp = defin.section.descp_rep
    # Analysis case
    case = defin.case
    # Section orientation
    angle = defin.section.angle
    orientationShape = defin.section.ang_shape
    # Calculation the gross section properties
    properties = defin.gross

    # These lengths will generally provide sufficient accuracy for
    # local, distortional, and global buckling modes
    # Length units are inches
    ReferenceLength = defin.member.Lx  # inches
    lengths = defin.member.lengths_data

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
    b_c = defin.member.support

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
    if case.case == 'Axial':
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
    fig.suptitle(f'Signature Curve\n{case.case} case, {case.explanation}\nfy: {fy:.2f} ksi')
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
# EXPLANATION OF INPUT TERMS
# ======================================================================================================================
# C_sign_solver(A, B, C, t, angle, Fyield, Case, MemLength)
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
# Fyield : Steel yield stress.
# Case : 'Axial' for uniform axial compression.
#           'Flexural' for bending creating compression at top fiber.
# MemLength : Total member length
# ======================================================================================================================


# ======================================================================================================================
# OUTPUT
# ======================================================================================================================
# Creation of a member to solve
C1 = C_sign_solver()
# Creation of graph if True plot will be shown
pC1 = plot_Sign_Curve(C1, True)
# Print minimas
print(pC1)
# Export the report. (Section definition, Curve)
export_report(C1, pC1)
