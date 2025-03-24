import numpy as np

def solve_stream(Uinf, r1_R, r2_R, rootradius_R, tipradius_R, Omega, Radius,
                 NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, use_prandtl=True):
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_R = (r1_R + r2_R) / 2

    a = 0.3  # axial induction
    a_line = 0.0  # tangential induction

    Niterations = 100
    Erroriterations = 0.00001

    for _ in range(Niterations):
        Urotor = Uinf * (1 - a)
        Utan = (1 + a_line) * Omega * r_R * Radius

        fnorm, ftan, gamma, alpha, phi = blade_loading(Urotor, Utan, r_R, chord, twist,
                                                       polar_alpha, polar_cl, polar_cd)

        load3Daxial = fnorm * Radius * (r2_R - r1_R) * NBlades
        CT = load3Daxial / (0.5 * Area * Uinf**2)

        a_new = induction(CT)

        Prandtl, Prandtltip, Prandtlroot = prandtl_correction(
            r_R, rootradius_R, tipradius_R, Omega * Radius / Uinf, NBlades, a_new)

        if Prandtl < 0.0001:
            Prandtl = 0.0001

        if use_prandtl and Prandtl > 0.01:
            a_new = a_new / Prandtl

        a = 0.75 * a + 0.25 * a_new

        a_line = ftan * NBlades / (2 * np.pi * Uinf * (1 - a) * Omega * 2 * (r_R * Radius)**2)
        if use_prandtl and Prandtl > 0.01:
            a_line = a_line / Prandtl

        if np.abs(a - a_new) < Erroriterations:
            break

    return [a, a_line, r_R, fnorm, ftan, gamma], alpha, phi

def blade_loading(vnorm, vtan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd):
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)
    alpha = twist + inflowangle * 180 / np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord
    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, inflowangle

def prandtl_correction(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    temp1 = -NBlades / 2 * (tipradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Ftip = 2 / np.pi * np.arccos(np.exp(temp1))
    Ftip = np.nan_to_num(Ftip)

    temp1 = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Froot = 2 / np.pi * np.arccos(np.exp(temp1))
    Froot = np.nan_to_num(Froot)

    return Froot * Ftip, Ftip, Froot

def induction(CT):
    CT = np.array(CT, ndmin=1)
    a = np.zeros(np.shape(CT))
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1
    a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(CT1) - 1))
    a[CT < CT2] = 0.5 - 0.5 * np.sqrt(1 - CT[CT < CT2])
    return a[0] if a.size == 1 else a

def CT_function(a, glauert=False):
    CT = 4 * a * (1 - a)
    if glauert:
        CT1 = 1.816
        a1 = 1 - np.sqrt(CT1) / 2
        CT = np.where(a > a1, CT1 - 4 * (np.sqrt(CT1) - 1) * (1 - a), CT)
    return CT
