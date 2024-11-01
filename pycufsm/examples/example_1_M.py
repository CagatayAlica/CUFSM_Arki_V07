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
# ------------------------------------------------------------------------
# Define units
# ------------------------------------------------------------------------
# Basic Units
mm = 1.0
N = 1.0
sec = 1.0

# Length
m = mm * 1000.0
cm = m / 100.0
inch = 25.4 * mm
ft = 12.0 * inch

# Force
kN = N * 1000.0
kips = kN * 4.448221615
lb = kips / 1.0e3

# Stress (kN/m2 or kPa)
Pa = N / (m ** 2)
kPa = Pa * 1.0e3
MPa = Pa * 1.0e6
GPa = Pa * 1.0e9
ksi = 6.8947573 * MPa
psi = 1e-3 * ksi

# Mass - Weight
tonne = kN * sec ** 2 / m
kg = N * sec ** 2 / m
lbs = psi * inch ** 2

# Gravitational acceleration
g = 9.81 * m / sec ** 2

# Time
mins = 60 * sec
hour = 60 * mins

# -------------------------------------------------------------------------
# Material information
# ------------------------------------------------------------------------
E = 29500
v = 0.3
fy = 55
# ------------------------------------------------------------------------
# Section dimensions
# ------------------------------------------------------------------------
A = 9.0
B = 2.5
C = 1.773
t = 0.059

isImperial = True
if isImperial:
    # -------------------------------------------------------------------------
    # Material information
    # ------------------------------------------------------------------------
    E = E
    v = v
    fy = fy
    # ------------------------------------------------------------------------
    # Section dimensions
    # ------------------------------------------------------------------------
    A = A
    B = B
    C = C
    t = t
else:
    # Set all values to Imperial
    # -------------------------------------------------------------------------
    # Material information
    # ------------------------------------------------------------------------
    E = E * ksi
    v = v
    fy = fy * ksi
    # ------------------------------------------------------------------------
    # Section dimensions
    # ------------------------------------------------------------------------
    A = A * inch
    B = B * inch
    C = C * inch
    t = t * inch





def __main__() -> Dict[str, np.ndarray]:
    # Define an isotropic material with E = 29,500 ksi and nu = 0.3

    props = np.array([np.array([0, E, E, v, v, E / (2 * (1 + v))])])

    # Define a lightly-meshed Cee shape
    # (1 element per lip, 2 elements per flange, 3 elements on the web)
    # Nodal location units are inches
    # section = hatSection(4.724, 2.362, 2.953, 0.787, 0.079)
    section = ceeSection(A, B, C, t)
    nodes = section[0]['0']
    elements = section[1]
    thickness = section[2]
    properties = grossProp(nodes[:, 1], nodes[:, 2], thickness, thickness)

    # These lengths will generally provide sufficient accuracy for
    # local, distortional, and global buckling modes
    # Length units are inches
    unit = ''
    if isImperial:
        unit = "imperial"
    else:
        unit = "metric"

    lengths = lengthRange(unit)
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
    nodes_p = stress_gen(
        nodes=nodes,
        forces={
            'P': 0,
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
    thecurve3(curve, clas, fileddisplay, 1, 1, clasopt, xmin, xmax, ymin, ymax, [1],
              fileindex, modeindex, picpoint)

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
        'Deformations': shapes
    }


if __name__ == '__main__':
    signa = __main__()
    lengths = lengthRange("imperial")
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    minimas = []
    fig.suptitle('Signature Curve')
    for i in range(2, len(signa['Y_values'])):
        slope = (signa['Y_values'][i - 1] - signa['Y_values'][i - 2]) / (
                signa['X_values'][i - 1] - signa['X_values'][i - 2])
        # with open("shapes.txt", mode='a') as file:
        # file.write(f"\nLength: {signa['X_values'][i]} ==> slope: {slope:.7f}")

    # Finding the minima points
    for loadFactor in range(2, len(signa['Y_values'])):
        if signa['Y_values'][loadFactor - 1] < signa['Y_values'][loadFactor - 2] and signa['Y_values'][loadFactor - 1] < \
                signa['Y_values'][loadFactor]:
            text = f"P/Py: {signa['Y_values'][loadFactor - 1]:.3f} \nL: {signa['X_values'][loadFactor - 1]}"
            minimas.append([signa['X_values'][loadFactor - 1], signa['Y_values'][loadFactor - 1]])
            print(f"Minima of the curve = {signa['Y_values'][loadFactor - 1]:.3f}")
            print(signa['Y_values'])
            print(signa['X_values'])

            with open("shapes.txt", mode='w') as file:
                file.write(str(signa['curve']))
            l_index = loadFactor

            # Annotating the minima values
            ax1.annotate(text, xy=(signa['X_values'][loadFactor - 1], signa['Y_values'][loadFactor - 1]),
                         xytext=(signa['X_values'][loadFactor - 1] * 0.3, signa['Y_values'][loadFactor - 1] * 0.3),
                         arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1), fontsize=8)
    # Setting the plot for the signature curve.
    ax1.plot(signa['X_values'], signa['Y_values'], linewidth=2.0)
    ax3.plot(signa['X_values'], signa['curve'][:, 1][:, 1], linewidth=2.0)
    print(signa['curve'][:, 1][:, 1])
    print(minimas)
    # Formatting the signature curve plot.
    ax1.axis(ymin=0.0, ymax=np.min([np.max(signa['Y_values']), 3 * np.median(signa['Y_values'])]))
    ax1.grid(color='b', linestyle='-', linewidth=0.2)
    ax1.axes.set_xscale("log")
    ax3.axes.set_xscale("log")
    ax1.axes.set_xlabel('length')
    ax1.axes.set_ylabel('load factor [P/Py]')
    ax1.axes.set_title('Buckling curve')

    # Drawing the cross section shape
    # IDs of the nodes.
    id_nodes = signa['nodes'][:, 0]
    # X values of the nodes.
    x_nodes = signa['nodes'][:, 1]
    # Y values of the nodes.
    y_nodes = signa['nodes'][:, 2]
    # Thickness
    thk = signa['thickness']

    for i in range(len(x_nodes)):
        ax2.axes.annotate(id_nodes[i], xy=(x_nodes[i] * 1.03, y_nodes[i]), xycoords='data', fontsize=8)
    ax2.plot(x_nodes, y_nodes, linewidth=thk * 10, color='green', marker="o", markersize=2)
    ax2.axis('equal')
    ax2.axes.set_xlabel('length')
    ax2.axes.set_ylabel('length')
    ax2.axes.set_title('Cross Section')
    # Show the plot
    plt.show()
