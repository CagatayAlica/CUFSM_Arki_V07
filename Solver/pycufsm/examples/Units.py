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