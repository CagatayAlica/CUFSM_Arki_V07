from typing import Dict
import numpy as np
from pycufsm.CUFSM_Functions.fsm import strip
from pycufsm.CUFSM_Functions.preprocess import stress_gen
from pycufsm.CUFSM_Functions.types import BC, GBT_Con, Sect_Props
from pycufsm.CUFSM_Functions.plotters import thecurve3
import matplotlib.pyplot as plt
from pycufsm.SectionProps.sectionDraw import ceeSection, lengthRange, grossProp


# This example presents a very simple Cee section,
# solved for pure compression,
# in the Imperial unit system


def C_sign_solver(A: float, B: float, C: float, t: float, angle: float, Fyield: float, Case: str,
                  MemLength: float) -> Dict[str, np.ndarray]:
    # Define an isotropic material with E = 29,500 ksi and nu = 0.3
    props = np.array([np.array([0, 29500, 29500, 0.3, 0.3, 29500 / (2 * (1 + 0.3))])])

    # Define a lightly-meshed Cee shape
    # (1 element per lip, 2 elements per flange, 3 elements on the web)
    # Nodal location units are inches
    # section = hatSection(4.724, 2.362, 2.953, 0.787, 0.079)
    # section = ceeSection(5.905, 3.80, 0.630, 0.0393, 0)
    section = ceeSection(A, B, C, t, angle)
    fy = Fyield  # ksi
    nodes = section[0]
    elements = section[1]
    thickness = section[2]
    descp = section[3]
    properties = grossProp(nodes[:, 1], nodes[:, 2], thickness, thickness)

    # These lengths will generally provide sufficient accuracy for
    # local, distortional, and global buckling modes
    # Length units are inches
    ReferenceLength = MemLength  # inches

    lengths = lengthRange(ReferenceLength, "imperial")
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
    b_c: BC = 'S-S'

    # For signature curve analysis, only a single array of ones makes sense here
    m_all = np.ones((len(lengths), 1))

    # Solve for 10 eigenvalues
    n_eigs = 10
    print(f"gross: {properties[0]}")
    # Set the section properties for this simple section
    # Normally, these might be calculated by an external package
    sect_props: Sect_Props = {
        'cx': properties[1][2],
        'cy': properties[1][1],
        'x0': properties[1][9],
        'y0': properties[1][10],
        'phi': 0,
        'A': properties[1][0],
        'Ixx': properties[1][3],
        'Ixy': properties[1][7],
        'Iyy': properties[1][5],
        'I11': properties[1][3],
        'I22': properties[1][5],
        'Cw': properties[1][11],
        'J': properties[1][12],
        'B1': 0,
        'B2': 0,
        'wn': np.array([])
    }

    # Generate the stress points assuming 50 ksi yield and pure compression
    if Case == 'Axial':
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

    fileindex = 1
    fileddisplay = [1]
    clas = 0
    clasopt = 0
    xmin = np.min(lengths) * 10 / 11
    xmax = np.max(lengths) * 11 / 10
    ymin = 0
    ymax = np.min([np.max(signature), 3 * np.median(signature)])
    modeindex = 1
    length_index = 40
    fileindex = 1
    picpoint = [lengths[length_index - 1], curves[length_index - 1, modeindex - 1]]
    # thecurve3(curve, clas, fileddisplay, 1, 1, clasopt, xmin, xmax, ymin, ymax, [1],
    # fileindex, modeindex, picpoint)

    # Return the important example results
    # The signature curve is simply a matter of plotting the
    # 'signature' values against the lengths
    # (usually on a logarithmic axis)

    return {
        'curve': curve,
        'nodes': nodes,
        'shapes': shapes,
        'thickness': thickness,
        'elements': elements,
        'flag': flag,
        'springs': springs,
        'constraints': constraints,
        'X_values': lengths,
        'Y_values': signature,
        'Y_values_allmodes': curve,
        'Orig_coords': nodes_p,
        'Deformations': shapes,
        'Reference_Length': ReferenceLength,
        'Section_Def': descp,
        'Yield_stress': fy,
        'Case': Case,
        'Sect_Props': properties
    }


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

    lengths = lengthRange(RefLen, "imperial")
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    minimas = []
    fig.suptitle(f'Signature Curve, {case}\n{descp}, {fy:.2f} ksi')
    # Finding the minima points
    for loadFactor in range(2, len(Y_Values)):
        if Y_Values[loadFactor - 1] < Y_Values[loadFactor - 2] and Y_Values[
            loadFactor - 1] < \
                Y_Values[loadFactor]:
            text = f"P/Py: {Y_Values[loadFactor - 1]:.3f} \nL: {X_Values[loadFactor - 1]}"
            minimas.append([X_Values[loadFactor - 1], Y_Values[loadFactor - 1]])
            # Annotating the minima values
            ax1.annotate(text, xy=(X_Values[loadFactor - 1], Y_Values[loadFactor - 1]),
                         xytext=(X_Values[loadFactor - 1] * 0.3, Y_Values[loadFactor - 1] * 0.3),
                         arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)

    # Find the index of the item
    h_index = np.where(lengths == RefLen)[0]
    h_value = Y_Values[h_index][0]
    h_text = f'P/Py: {h_value:.3f}\nL: {RefLen}'
    minimas.append([RefLen, h_value])
    # Annotating the minima values
    ax1.annotate(h_text, xy=(RefLen, h_value),
                 xytext=(RefLen * 0.3, h_value * 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)
    # Setting the plot for the signature curve.
    ax1.plot(X_Values, Y_Values, linewidth=2.0)
    # print(signa['curve'][:, 1][:, 1])
    print(minimas)
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
    ax2.axes.set_xlabel('length [in]')
    ax2.axes.set_ylabel('length [in]')
    ax2.axes.set_title('Cross Section')
    # Show the plot
    if plot:
        plt.show()
    return minimas


# C_sign_solver(A, B, C, t, angle, Fyield, Case, MemLength)
# Units [in, ksi]
# A : Web height
# B : Flange width
# C : Lip length
# t : Steel thickness
# angle : Orientation of the section
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
# Fyield : Steel yield stress
# Case : 'Axial' for uniform axial compression
#           'Flx' for bending
# MemLength : Total member length

C1 = C_sign_solver(9.0, 2.5, 0.773, 0.059, 0, 55.0, 'Axial', 150.0)
C2 = C_sign_solver(9.0, 2.5, 0.773, 0.059, 0, 55.0, 'Flx', 150.0)

plot_Sign_Curve(C1, True)
plot_Sign_Curve(C2, False)
