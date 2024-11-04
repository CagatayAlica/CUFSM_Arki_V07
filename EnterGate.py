class Calculation_input:
    def __init__(self, A: float, B: float, C: float, R: float, t: float, E: float, fy: float, v:float):
        self.A = A       # Web height
        self.B = B       # Flange width
        self.C = C       # Lip length
        self.R = R       # Inner Radius
        self.t = t       # Thickness
        self.E = E       # Modulus of elasticity
        self.fy = fy      # Yield stress
        self.v = v       # Poisson's ratio

