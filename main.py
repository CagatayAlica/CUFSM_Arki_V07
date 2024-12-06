import Input.Definitions as Inp
import AISI_Functions.Strengths as Strength
import FirstGate as gate


def member_base_calculation():
    """
    This function will be called one time per member.
    :return:Signature curve [x,y] in a list.
    """
    print(f'Direct Strength Method:\nAxial Compression:')
    print(Strength.Str_Axial_0.Results)
    print(f'Bending About Strong Axis:')
    print(Strength.Str_Flx_0.Results)
    print(f'Bending About Weak Axis / Web is under compression:')
    print(Strength.Str_Flx_90.Results)
    print(f'Bending About Weak Axis / Lips are under compression:')
    print(Strength.Str_Flx_270.Results)
    Results = {'ASD': {'DSM_AxialCompression': Strength.Str_Axial_0.Results['Strength_ASD'],
                       'DSM_Flexure0': Strength.Str_Flx_0.Results['Strength_ASD'],
                       'DSM_Flexure90': Strength.Str_Flx_90.Results['Strength_ASD'],
                       'DSM_Flexure270': Strength.Str_Flx_270.Results['Strength_ASD']},
               'LRFD': {'DSM_AxialCompression': Strength.Str_Axial_0.Results['Strength_LRFD'],
                        'DSM_Flexure0': Strength.Str_Flx_0.Results['Strength_LRFD'],
                        'DSM_Flexure90': Strength.Str_Flx_90.Results['Strength_LRFD'],
                        'DSM_Flexure270': Strength.Str_Flx_270.Results['Strength_LRFD']}
               }
    Curves = [Strength.Curve_Axial_0, Strength.Curve_Flx_0, Strength.Curve_Flx_90, Strength.Curve_Flx_270]
    return Curves, Results


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
    print(f'Global Strengths:\nAxial Tension:')
    print(Strength.Tension(Inp.Material_1, Inp.Gross_ang0).strength())
    print(f'Axial Compression:')
    print(Strength.Compression(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strength())
    print(f'Bending About Strong Axis:')
    print(Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthStrong())
    print(f'Bending About Weak Axis:')
    print(Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthWeak())
    print(f'Shear Along Strong Axis:')
    print(Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthStrong())
    print(f'Shear Along Weak Axis:')
    print(Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthWeak())

    Results = {'ASD': {'Global_AxialTension': Strength.Tension(Inp.Material_1, Inp.Gross_ang0).strength()['Tno'],
                       'Global_AxialCompression':
                           Strength.Compression(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strength()[
                               'Pno'],
                       'Global_Flexure0':
                           Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthStrong()[
                               'Mno'],
                       'Global_Flexure90':
                           Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthWeak()[
                               'Mno'],
                       'Global_Flexure270':
                           Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthWeak()[
                               'Mno'],
                       'Global_ShearStrong': Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthStrong()['Vno'],
                       'Global_ShearWeak': Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthWeak()['Vno']},
               'LRFD': {'Global_AxialTension': Strength.Tension(Inp.Material_1, Inp.Gross_ang0).strength()['ffTn'],
                        'Global_AxialCompression':
                            Strength.Compression(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strength()[
                                'ffPn'],
                        'Global_Flexure0':
                            Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthStrong()[
                                'ffMn'],
                        'Global_Flexure90':
                            Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthWeak()[
                                'ffMn'],
                        'Global_Flexure270':
                            Strength.Flexure(Inp.Material_1, Inp.Gross_ang0, Lx, Ly, Lt, Kx, Ky, Kt).strengthWeak()[
                                'ffMn'],
                        'Global_ShearStrong': Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthStrong()['ffVn'],
                        'Global_ShearWeak': Strength.Shear(Inp.Material_1, Inp.Section_ang0).strengthWeak()['ffVn']}
               }

    return Results


if __name__ == '__main__':
    print(f'++++++++++\nCALCULATION, Unit [{gate.Unit_Definition.Unit}]\n++++++++++\n')
    analysisMethod = gate.StrengthMethod
    print(f'Method {analysisMethod}')
    Parameter = gate.Analysis_Section
    # This function will be called for members.
    calc_1 = member_base_calculation()[1]
    calc_2 = station_base_calculation(Parameter.Lxs,
                                      Parameter.Lys,
                                      Parameter.Lts,
                                      Parameter.Kxs,
                                      Parameter.Kys,
                                      Parameter.Kts)
    # Axial Compression Strength
    Analysis_Results = {
        'T_strength': calc_2[analysisMethod]['Global_AxialTension'],
        'P_strength': min(calc_1[analysisMethod]['DSM_AxialCompression'],
                          calc_2[analysisMethod]['Global_AxialCompression']),
        'M_Strong_strength': min(calc_1[analysisMethod]['DSM_Flexure0'], calc_2[analysisMethod]['Global_Flexure0']),
        'M_Weak_Lip_strength': min(calc_1[analysisMethod]['DSM_Flexure90'], calc_2[analysisMethod]['Global_Flexure90']),
        'M_Weak_Web_strength': min(calc_1[analysisMethod]['DSM_Flexure270'],
                                   calc_2[analysisMethod]['Global_Flexure270']),
        'V_AlongWeb': calc_2[analysisMethod]['Global_ShearStrong'],
        'V_AlongFlange': calc_2[analysisMethod]['Global_ShearWeak']}
    print(f'Method {analysisMethod}')
    for key, value in Analysis_Results.items():
        print(f'{key}: {value:.4f}')

