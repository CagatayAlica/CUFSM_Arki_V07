import math


class Tension:
    def __init__(self, Material, Section):
        self.Ag = Section['gross'].Ar
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
            Vn = 2*Vy
        elif 0.815 < lamv <= 1.227:
            Vn = 2*0.815 * math.sqrt(Vcr * Vy)
        else:
            Vn = 2*Vcr
        Vno = Vn / self.omega
        ffVn = Vn * self.ff
        StrengthWeak = {'Vn': Vn, 'Vno': Vno, 'ffVn': ffVn}
        return StrengthWeak
