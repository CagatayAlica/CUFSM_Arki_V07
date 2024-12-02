from AISI_Functions import DSM_AISI as aisi
from AISI_Functions import CombinedForces as ratio
from AISI_Functions import DirectStrengthMethod as strength


def member_base_calculation(**kwargs):
    """
    Element strength calculation as per AISI S100-17.
    :param kwargs: Element parameters.
        A = Web height [in]
        B = Flange width [in]
        C = Lip length [in]
        t = Thickness [in]
        R = Inner radius [in]
        Lx = Critical member length for bending about x-x axis [in]
        Ly = Critical member length for bending about y-y axis [in]
        Lt = Critical member length for torsion [in]
        Kx = Buckling length factor for bending about x-x axis [in]
        Ky = Buckling length factor for bending about y-y axis [in]
        Kt = Buckling length factor for torsion [in]
        fy = Steel yield stress [ksi]
    :return: Strength values as per AISI S100-17 using DSM.
    """
    A = kwargs['A']
    B = kwargs['B']
    C = kwargs['C']
    t = kwargs['t']
    R = kwargs['R']
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    Lt = kwargs['Lt']
    Kx = kwargs['Kx']
    Ky = kwargs['Ky']
    Kt = kwargs['Kt']
    fy = kwargs['fy']
    # CREATING A MEMBER
    ff = aisi.MainInput(A=A, B=B, C=C, t=t, R=R, Lx=Lx, Ly=Ly, Lt=Lt, Kx=Kx, Ky=Ky, Kt=Kt, fy=fy)

    # BUCKLING ANALYSIS
    FSMA_Axial = aisi.BucklingAnalysis(ff.Input_Dict['C_Axial'],
                                       ff.Input_Dict['Material'],
                                       ff.Input_Dict['Member'])
    FSMA_C_ang0_Flex = aisi.BucklingAnalysis(ff.Input_Dict['C_ang0_Flex'],
                                             ff.Input_Dict['Material'],
                                             ff.Input_Dict['Member'])
    FSMA_C_ang90_Flex = aisi.BucklingAnalysis(ff.Input_Dict['C_ang90_Flex'],
                                              ff.Input_Dict['Material'],
                                              ff.Input_Dict['Member'])
    FSMA_C_ang270_Flex = aisi.BucklingAnalysis(ff.Input_Dict['C_ang270_Flex'],
                                               ff.Input_Dict['Material'],
                                               ff.Input_Dict['Member'])

    # PLOTTING THE SIGNATURE CURVE AND GETTING MINIMAS
    p_C_Axial = aisi.PlotSignatureCurve(FSMA_Axial.BucklingAnalysis, True)
    p_C_ang0_Flex = aisi.PlotSignatureCurve(FSMA_C_ang0_Flex.BucklingAnalysis,  True)
    p_C_ang90_Flex = aisi.PlotSignatureCurve(FSMA_C_ang90_Flex.BucklingAnalysis,  True)
    p_C_ang270_Flex = aisi.PlotSignatureCurve(FSMA_C_ang270_Flex.BucklingAnalysis,  True)

    # PRINTING THE ANALYSIS RESULTS OF FSM
    aisi.FSMA_Report(FSMA_Axial.BucklingAnalysis, p_C_Axial.MinimasList)
    aisi.FSMA_Report(FSMA_C_ang0_Flex.BucklingAnalysis, p_C_ang0_Flex.MinimasList)
    aisi.FSMA_Report(FSMA_C_ang90_Flex.BucklingAnalysis, p_C_ang90_Flex.MinimasList)
    aisi.FSMA_Report(FSMA_C_ang270_Flex.BucklingAnalysis, p_C_ang270_Flex.MinimasList)

    # SECTION STRENGTHS / Local / Distortional Buckling
    S_Axial = strength.stregnths(p_C_Axial.MinimasList, ff.Input_Dict['Material'], ff.Input_Dict['C_Axial'],
                                 ff.Input_Dict['Member'])
    S_C_ang0_Flex = strength.stregnths(p_C_ang0_Flex.MinimasList, ff.Input_Dict['Material'],
                                       ff.Input_Dict['C_ang0_Flex'], ff.Input_Dict['Member'])
    S_C_ang90_Flex = strength.stregnths(p_C_ang90_Flex.MinimasList, ff.Input_Dict['Material'],
                                        ff.Input_Dict['C_ang90_Flex'], ff.Input_Dict['Member'])
    S_C_ang270_Flex = strength.stregnths(p_C_ang270_Flex.MinimasList, ff.Input_Dict['Material'],
                                         ff.Input_Dict['C_ang270_Flex'], ff.Input_Dict['Member'])

    # SECTION STRENGTHS / Tension and Shear

    return [S_Axial, S_C_ang0_Flex, S_C_ang90_Flex, S_C_ang270_Flex]


def station_base_calculation(**kwargs):
    """
        Element strength calculation as per AISI S100-17.
        :param kwargs: Element parameters.
            A = Web height [in]
            B = Flange width [in]
            C = Lip length [in]
            t = Thickness [in]
            R = Inner radius [in]
            Lx = Critical member length for bending about x-x axis [in]
            Ly = Critical member length for bending about y-y axis [in]
            Lt = Critical member length for torsion [in]
            Kx = Buckling length factor for bending about x-x axis [in]
            Ky = Buckling length factor for bending about y-y axis [in]
            Kt = Buckling length factor for torsion [in]
            fy = Steel yield stress [ksi]
        :return: Strength values as per AISI S100-17 using DSM.
        """
    A = kwargs['A']
    B = kwargs['B']
    C = kwargs['C']
    t = kwargs['t']
    R = kwargs['R']
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    Lt = kwargs['Lt']
    Kx = kwargs['Kx']
    Ky = kwargs['Ky']
    Kt = kwargs['Kt']
    fy = kwargs['fy']
    # CREATING A MEMBER
    ff = aisi.MainInput(A=A, B=B, C=C, t=t, R=R, Lx=Lx, Ly=Ly, Lt=Lt, Kx=Kx, Ky=Ky, Kt=Kt, fy=fy)
    gross = ff.Input_Dict['Gross']
    member = ff.Input_Dict['Member']
    Tn = aisi.Tension(ff.Input_Dict['Material'], gross).strength()
    VnStrong = aisi.Shear(ff.Input_Dict['Material'], gross).strengthStrong()
    VnWeak = aisi.Shear(ff.Input_Dict['Material'], gross).strengthWeak()
    Pn = aisi.Compression(ff.Input_Dict['Material'], gross, member).strength()
    Mn = aisi.Flexure(ff.Input_Dict['Material'], gross, member).strength()
    Strength = {'Tn': Tn, 'VnStrong': VnStrong, 'VnWeak': VnWeak, 'Pn': Pn, 'Mn': Mn}
    return Strength


def member_check_ratios():
    # CHECKING THE COMBINED FORCES AND RATIOS
    member = ratio.CombinedForces(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    ratioTensionBending = member.TensionBending()
    ratioCompressionBending = member.CompressionBending()
    print(ratioTensionBending)
    print(ratioCompressionBending)


if __name__ == '__main__':
    # This function will be called for members.
    member_base_calculation(A=3.625, B=1.625, C=0.500, t=0.0566, R=0.0849, Lx=120.0, Ly=60.0, Lt=60.0, Kx=1.0,
                            Ky=1.0, Kt=1.0, fy=50.0)
    station_base_calculation(A=3.625, B=1.625, C=0.500, t=0.0566, R=0.0849, Lx=120.0, Ly=60.0, Lt=60.0, Kx=1.0,
                            Ky=1.0, Kt=1.0, fy=50.0)
    # This function will be called for every station.
    member_check_ratios()
