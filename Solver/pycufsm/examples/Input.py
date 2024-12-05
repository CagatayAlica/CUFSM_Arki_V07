class Calculation_input:
    def __init__(self):
        self.A = None       # Web height
        self.B = None       # Flange width
        self.C = None       # Lip length
        self.t = None       # Thickness
        self.E = None       # Modulus of elasticity
        self.fy = None      # Yield stress
        self.fu = None      # Ultimate stress
        self.v = None       # Poisson's ratio
        self.section()
        self.material()

    def section(self):
        # ------------------------------------------------------------------------
        # Section dimensions
        # ------------------------------------------------------------------------
        self.A = 9.0
        self.B = 2.5
        self.C = 1.773
        self.t = 0.059

    def material(self):
        # ------------------------------------------------------------------------
        # Material information
        # ------------------------------------------------------------------------
        self.E = 29500
        self.v = 0.3
        self.fy = 55.0
        self.fu = 62.0


sec = Calculation_input()
A = sec.A
fy = sec.fy
print(A)
print(fy)
