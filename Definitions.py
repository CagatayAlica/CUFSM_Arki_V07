import Input.CreateSection as sec
import Input.Material as mat
import Input.Cases as ca
import Input.Member as mem

# Creation of a member to solve
#section = sec.C_Section(8.0, 1.625, 0.625, 0.1017, 0.1525, 0)
section = sec.C_Section(9.0, 2.50, 0.773, 0.059, 0.1875, 0)
#section = sec.U_Section(8.0, 2.0, 0.1017, 0.1525, 0)
# Calculate the gross-section properties
gross = sec.GrossProps(section.nodes[:, 1], section.nodes[:, 2], section.t, section.r)
# Define the material
material = mat.Material(55.0)
# Define an analysis case
case = ca.Cases('Flexural')
# Define a member
member = mem.Member(56.3, 56.3, 56.3, 1, 1, 1, 'S-S')
