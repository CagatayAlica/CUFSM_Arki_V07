import math


class CombinedForces:
    def __init__(self, Mx, My, N, Vx, Vy, Mxa, Mya, Ta, Pa, Vxa, Vya):
        self.Mx = Mx
        self.My = My
        self.N = N
        self.Vx = Vx
        self.Vy = Vy
        self.Mxa = Mxa
        self.Mya = Mya
        self.Ta = Ta
        self.Pa = Pa
        self.Vxa = Vxa
        self.Vya = Vya

    def TensionBending(self):
        Ratios = {'H1.1-1': self.Mx / self.Mxa + self.My / self.Mya + self.N / self.Ta,
                  'H1.1-2': self.Mx / self.Mxa + self.My / self.Mya - self.N / self.Ta}
        return Ratios

    def CompressionBending(self):
        Ratios = {'H1.2-1': self.Mx / self.Mxa + self.My / self.Mya + self.N / self.Pa}
        return Ratios

    def BendingShear(self):
        Ratios = {'H2-1_x': math.sqrt(math.pow(self.Mx/self.Mxa, 2)+math.pow(self.Vy/self.Vya, 2)),
                  'H2-1_y': math.sqrt(math.pow(self.My/self.Mya, 2)+math.pow(self.Vx/self.Vxa, 2))}
        return Ratios

