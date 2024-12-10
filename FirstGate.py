from typing import Literal


class Base_Unit:
    def __init__(self, unit: Literal['Metric', 'Imperial']):
        self.Unit = unit
        if self.Unit == 'Metric':
            self.length_converting = 1.0 / 25.4
            self.stress_converting = 1.0 / 6.8947573
            self.force_converting = 4.44822
            self.moment_converting = 4.44822 * 0.0254
        else:
            self.length_converting = 1.0
            self.stress_converting = 1.0
            self.force_converting = 1.0
            self.moment_converting = 1.0

    def convert_length(self, x):
        return x * self.length_converting

    def convert_stress(self, x):
        return x * self.stress_converting

    def convert_force(self, x):
        return x * self.force_converting

    def convert_moment(self, x):
        return x * self.moment_converting


class Analysis:

    def __init__(self, method: Literal['ASD', 'LRFD']):
        self.Method = method

    def method(self):
        return self.Method


class Main_Input_Parameters:
    def __init__(self, A: float, B: float, C: float, t: float, R: float, fy: float, Lxg: float, Lyg: float, Ltg: float,
                 Kxg: float, Kyg: float, Ktg: float, Lxs: float, Lys: float, Lts: float, Kxs: float, Kys: float,
                 Kts: float, Unit_Def):
        """
        Main entrance gate of the program. Units should be [mm, MPa] or [in, ksi].
        :param A: Web height
        :param B: Flange width
        :param C: Lip length
        :param t: Nominal thickness
        :param R: Inner bend radius
        :param fy: Steel yield stress
        :param Lxg: Critical member length in global state
        :param Lyg: Critical member length in global state
        :param Ltg: Critical member length in global state
        :param Kxg: Critical member length factor in global state
        :param Kyg: Critical member length factor in global state
        :param Ktg: Critical member length factor in global state
        :param Lxs: Critical member length in station state
        :param Lys: Critical member length in station state
        :param Lts: Critical member length in station state
        :param Kxs: Critical member length factor in station state
        :param Kys: Critical member length factor in station state
        :param Kts: Critical member length factor in station state
        """

        conv = Unit_Def

        self.A = conv.convert_length(A)
        self.B = conv.convert_length(B)
        self.C = conv.convert_length(C)
        self.t = conv.convert_length(t)
        self.R = conv.convert_length(R)
        self.fy = conv.convert_stress(fy)

        self.Lxg = conv.convert_length(Lxg)
        self.Lyg = conv.convert_length(Lyg)
        self.Ltg = conv.convert_length(Ltg)
        self.Kxg = Kxg
        self.Kyg = Kyg
        self.Ktg = Ktg

        self.Lxs = conv.convert_length(Lxs)
        self.Lys = conv.convert_length(Lys)
        self.Lts = conv.convert_length(Lts)
        self.Kxs = Kxs
        self.Kys = Kys
        self.Kts = Kts


StrengthMethod = Analysis('LRFD').method()

Unit_Definition = Base_Unit('Imperial')

Input = Main_Input_Parameters(2.5, 1.25, 0.188, 0.0451, 0.0712,
                              33,
                              110, 50, 50, 1.0, 1.0, 1.0,
                              110, 50, 50, 1.0, 1.0, 1.0,
                              Unit_Definition)

Input_M = Main_Input_Parameters(120.0, 55.0, 15.0, 1.0, 2.5,
                                350.0,
                                2800.0, 1220.0, 1220.0, 1.0, 1.0, 1.0,
                                2800.0, 1220.0, 1220.0, 1.0, 1.0, 1.0,
                                Unit_Definition)

Analysis_Section = Input
