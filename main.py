import Input.Definitions as Inp
import AISI_Functions.Strengths as Strength


def member_base_calculation():
    """
    This function will be called one time per member.
    :return:Signature curve [x,y] in a list.
    """
    print(Strength.Str_Axial_0.Results)
    print(Strength.Str_Flx_0.Results)
    print(Strength.Str_Flx_90.Results)
    print(Strength.Str_Flx_270.Results)
    Curves = [Strength.Curve_Axial_0, Strength.Curve_Flx_0, Strength.Curve_Flx_90, Strength.Curve_Flx_270]
    return Curves


def station_base_calculation(Lx: float, Ly: float, Lt: float, Kx: float, Ky: float, Kt: float):
    """
    This function will be called for every station.
    :param Lx:
    :param Ly:
    :param Lt:
    :param Kx:
    :param Ky:
    :param Kt:
    :return:
    """
    print(Strength.Tension(Inp.Material_1, Inp.Gross_ang0).strength())
    print(Strength.Compression(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strength())
    print(Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strength())
    print(Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthStrong())
    print(Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthWeak())


if __name__ == '__main__':
    # This function will be called for members.
    member_base_calculation()
    station_base_calculation(110, 50, 50, 1, 1, 1)
