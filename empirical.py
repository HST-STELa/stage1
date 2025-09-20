import numpy as np
from astropy import units as u

import Mors as mors

import utilities as utils


def stellar_Mdot_from_Xray_wood21(Fx_sfc):
    """

    Parameters
    ----------
    Fx_sfc : soft X-ray flux at the surface of the star,

    Returns
    -------
    Mdot_sfc : mass loss rate per unit sfc area of star
    """
    Mdot_sun = 2e-14 * u.Unit('Msun yr-1')
    Mdot_sfc_sun = Mdot_sun / (4 * np.pi * u.Rsun**2)
    pivot = 1e6 * u.Unit('erg s-1 cm-2')
    Mdot_sfc = 15.89 * Mdot_sfc_sun * (Fx_sfc/pivot)**0.77
    return Mdot_sfc.to('g s-1 cm-2')


def Lya_from_galex_schneider19(mag, dist, band='nuv'):
    # mag to flux density from  https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
    # Weff from SVO filter service http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=GALEX&asttype=
    dfac = (dist*u.pc/u.AU)**2
    dfac = dfac.to('').value
    if band == 'nuv':
        f = 10**((mag - 20.08)/-2.5)*2.06e-16
        F = f * 768.31
        F1AU = F*dfac
        logFlya = 0.7 * np.log10(F1AU) + 0.193
    elif band == 'fuv':
        f = 10**((mag - 18.82)/-2.5)*1.4e-15
        F = f * 265.57
        F1AU = F*dfac
        logFlya = 0.742 * np.log10(F1AU) + 0.945
    else:
        raise ValueError
    return 10**logFlya


def Lya_from_Teff_schneider19(Teff):
    """
    Relationship only valid for field-age stars.
    """
    logF = 6.754e-4 * Teff - 2.639
    Flya_au = 10 ** logF * u.Unit('erg s-1 cm-2')
    return Flya_au


def Lya_from_Teff_linsky13(Teff, Prot):
    fast = Prot < 10
    normal = (Prot >= 10) & (Prot < 25)
    slow = Prot >= 25

    logFlya = np.zeros(len(Teff))

    logFlya[fast] = 0.37688 + 0.0002061*Teff[fast]
    logFlya[normal] = 0.48243 + 0.0001632*Teff[normal]
    logFlya[slow] = -1.5963 + 0.0004732*Teff[slow]

    return 10**logFlya


def EUV_Linsky14(Flya_at_1AU, Teff, return_spec=False):
    if type(Teff) is not np.ndarray:
        Teff = np.array([Teff])
        Flya_at_1AU = np.array([Flya_at_1AU])
    logFEUV_bands = np.zeros((9, len(Teff)))
    Ms = Teff < 4000
    logFlya = np.log10(Flya_at_1AU)
    logFEUV_bands[0, Ms] = -0.491 + logFlya[Ms]
    logFEUV_bands[1, Ms] = -0.548 + logFlya[Ms]
    logFEUV_bands[2, Ms] = -0.602 + logFlya[Ms]
    logFEUV_bands[0, ~Ms] = -1.357 + 0.344*logFlya[~Ms] + logFlya[~Ms]
    logFEUV_bands[1, ~Ms] = -1.300 + 0.309*logFlya[~Ms] + logFlya[~Ms]
    logFEUV_bands[2, ~Ms] = -0.882 + logFlya[~Ms]
    logFEUV_bands[3, :] = -2.294+0.258*logFlya + logFlya
    logFEUV_bands[4, :] = -2.098+0.572*logFlya + logFlya
    logFEUV_bands[5, :] = -1.920+0.240*logFlya + logFlya
    logFEUV_bands[6, :] = -1.894+0.518*logFlya + logFlya
    logFEUV_bands[7, :] = -1.811+0.764*logFlya + logFlya
    logFEUV_bands[8, :] = -1.025+ logFlya
    FEUV_bands = 10**logFEUV_bands
    FEUV = np.sum(FEUV_bands, 0)
    if return_spec:
        bin_edges = np.array([100, 200, 300, 400, 500, 600, 700, 800, 912, 1170])
        dw = np.diff(bin_edges)
        Fdens = FEUV_bands/dw[:,None]
        w = utils.midpts(bin_edges)
        return w, dw, Fdens.squeeze()
    else:
        return FEUV


mors_bin_edges = [0.517, 12.4, 36, 92] * u.nm  # actually the x-ray and euv bands overlap by 2 nm, but I'm ignoring that


def age_from_Prot_johnstone21(Prot, M):
    Prot = Prot.to_value('d')
    M = M.to_value('Msun')
    track = mors.Star(Mstar=M, percentile=50)
    logagegrid = np.log10(track.AgeTrack)
    Pgrid = track.ProtTrack
    logage = np.interp(Prot, Pgrid, logagegrid)
    age = 10 ** logage
    return age * u.Myr


def Prot_from_age_johnstone21(age, M):
    age = age.to_value('Myr')
    logage = np.log10(age)
    M = M.to_value('Msun')
    track = mors.Star(Mstar=M, percentile=50)
    logagegrid = np.log10(track.AgeTrack)
    Pgrid = track.ProtTrack
    P = np.interp(logage, logagegrid, Pgrid)
    return P * u.d


def XUV_from_Prot_johnstone21(Prot, M, age):
    """

    Parameters
    ----------
    Prot : stellar rotation period as Quantity
    M : stellar mass as Quantity
    age : 'stellar age as Quantity
        Age is important because it sets the stellar radius.

    Returns
    -------
    w : wavelength of bin midpoints
    dw : widths of bins
    F : "luminosity density" of the star
        i.e. the luminosity in each bin divided by the bin width, units of erg s-1 AA-1
    """
    age = age.to_value('Myr')
    if age > 1e4:
        age = 1e4
    Prot = Prot.to_value('d')
    M = M.to_value('Msun')

    star = mors.Star(Mstar=M, Prot=Prot)
    dw = np.diff(mors_bin_edges)
    w = utils.midpts(mors_bin_edges)
    L = [star.Value(Age=age, Quantity=key) for key in 'Lx Leuv1 Leuv2'.split()]
    L = np.asarray(L) * u.erg / u.s
    F = L / dw

    return w.to('AA'), dw.to('AA'), F.to('erg s-1 AA-1')


def ho_gap_lowlim(P, Mstar):
    A, B, C = -0.09, 0.21, 0.35
    width = 0.1
    logP = np.log10(P)
    logM = np.log10(Mstar)
    logR = A*logP + B*logM + C - width/2
    return 10**logR


def turnover_time(Msun):
    # eqn 6 from Wright+ 2018
    log_tau = 2.33 - 1.50 + 0.31*Msun**2
    return 10**log_tau * u.d
