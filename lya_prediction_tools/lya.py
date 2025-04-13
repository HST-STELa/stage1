from math import pi, nan
import warnings

import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.special import wofz
from scipy.stats import norm
from tqdm import tqdm

import utilities as utils
from lya_prediction_tools import ism
from lya_prediction_tools import etc


# region intrinsic lya line shape
fwhm_broad = 400*u.km/u.s # rough average from youngblood+ 2016
fwhm_narrow = 125*u.km/u.s
amp_ratio_broad_narrow = 0.04
sig_ratio_reverse_narrow = 0.5 # just a by-eye calibration using the results of Wood+ 2005
amp_ratio_reverse_narrow = 0.3/0.5
#endregion

# region parameters you  might care about
Tism = 10000*u.K
# endregion

#region parameters you probably don't care about
wlab_H = 1215.67*u.AA
wlab_D = 1215.34*u.AA
D_H = 1.5e-5  # from bourrier+ 2017 (kepler 444)
f = 4.1641e-01
A = 6.2649e8 / u.s
mH, mD = 1 * u.u, 2 * u.u
#endregion


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


def EUV_Linsky14(Flya_at_1AU, Teff):
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
    return FEUV


def lya_factor_percentile(percentile):
    lya_factor_dex = -norm.isf(percentile / 100) * 0.2
    lya_factor = 10 ** lya_factor_dex
    return lya_factor


def transmission(w, rv, Nh, T):
    w0s = doppler_shift((wlab_H, wlab_D)*u.AA, rv)
    xsections = [voigt_xsection(w, w0, f, A, T, m) for
                 w0, m in zip(w0s, (mH, mD))]
    tau = xsections[0]*Nh + xsections[1]*Nh*D_H
    return np.exp(-tau)


def w2v(w):
    return  (w/wlab_H.value - 1)*const.c.to('km/s').value


def v2w(v):
    return  (v/const.c.to('km/s').value + 1)*wlab_H.value


def reversed_lya_profile(w, rv, flux):
    w0 = doppler_shift(wlab_H, rv)

    def sig_from_fwhm(fwhm):
        sig = fwhm/2/np.sqrt(2*np.log(2))*w0/const.c
        return sig.to(w0.unit)
    sig_broad, sig_narrow = map(sig_from_fwhm, (fwhm_broad, fwhm_narrow))
    sig_reverse = sig_narrow*sig_ratio_reverse_narrow

    def gaussian_profile(amp, sigma):
        f = amp*np.exp(-(w - w0)**2/2/sigma**2)
        return f

    def gaussian_flux(amp, sigma):
        return np.sqrt(2*np.pi)*sigma*amp

    raw_flux = (gaussian_flux(1, sig_narrow)
                + gaussian_flux(amp_ratio_broad_narrow, sig_broad)
                - gaussian_flux(amp_ratio_reverse_narrow, sig_reverse))
    raw_profile = (gaussian_profile(1, sig_narrow)
                   + gaussian_profile(amp_ratio_broad_narrow, sig_broad)
                   - gaussian_profile(amp_ratio_reverse_narrow, sig_reverse))
    profile = flux/raw_flux * raw_profile
    return profile


def lya_with_ISM(wgrid, Flya, rv_star, rv_ism, Nh, Tism):
    intrinsic = reversed_lya_profile(wgrid, rv_star, Flya)
    transmitted = transmission(wgrid, rv_ism, Nh, Tism)
    observed = intrinsic * transmitted
    return observed


wgrid_std = np.arange(1210., 1220., 0.005) * u.AA
def lya_at_earth_auto(catalog, n_H, default_rv=nan, lya_factor=1.0, lya_1AU_colname='Flya_1AU_adopted',
                      show_progress=False):
    get_nanfilled = lambda colname: catalog[colname].filled(np.nan).quantity
    ra, dec, d, Flya_1AU = map(get_nanfilled, ('ra', 'dec', 'sy_dist', lya_1AU_colname))
    Flya_at_earth_no_ISM = Flya_1AU * (u.AU/d)**2 * lya_factor
    Nh = n_H * d
    rv_ism = ism.ism_velocity(ra, dec)
    rv_star = __default_rv_handler(catalog, default_rv)

    lya_spec_at_earth_w_ISM = []
    sets = list(zip(Flya_at_earth_no_ISM, rv_star, rv_ism, Nh))
    if show_progress:
        sets = tqdm(sets)
    for F, rvs, rvi, Nh_ in sets:
        F_at_earth = lya_with_ISM(wgrid_std, F, rvs, rvi, Nh_, Tism)
        lya_spec_at_earth_w_ISM.append(F_at_earth)
    lya_spec_at_earth_w_ISM = u.Quantity(lya_spec_at_earth_w_ISM)
    return lya_spec_at_earth_w_ISM.to('erg s-1 cm-2 AA-1')


def __default_rv_handler(catalog, default_rv):
    ra = catalog['ra'].filled(nan).quantity
    dec = catalog['dec'].filled(nan).quantity
    rv_ism = ism.ism_velocity(ra, dec)
    rv_col = catalog['st_radv']
    if type(default_rv) in (float, int):
        rv_star = rv_col.filled(default_rv).quantity
        if np.any(rv_col.mask) and np.isnan(default_rv):
            warnings.warn('Some st_radv values are missing and are being filled with NaN. NaN output will result. '
                          'Set default_rv to a real value if you want to avoid this.')
    elif default_rv == 'ism':
        rv_star = rv_col.filled(nan).quantity
        mask = rv_col.mask
        rv_star[mask] = rv_ism[mask]
    else:
        raise ValueError
    return rv_star



# region ETC reference info for SNR calcs
etc_ref = etc.g140m_etc_countrates
w_etc = etc_ref['wavelength'] * u.AA # EF = earth frame
we_etc = utils.mids2edges(w_etc.value, simple=True) * u.AA
v_etc = w2v(w_etc.value)

etc_ref['flux2cps'] = etc_ref['target_counts'] / etc.g140m_expt_ref / etc.g140m_flux_ref
etc_ref['bkgnd_cps'] = (etc_ref['total_counts'] - etc_ref['target_counts'])/etc.g140m_expt_ref
# endregion


def sim_g140m_obs(f, expt):
    fpix = utils.intergolate(we_etc, wgrid_std, f)
    src = fpix * etc_ref['flux2cps'] * expt
    bkgnd = etc_ref['bkgnd_cps'] * expt
    total = bkgnd + src
    err_counts = np.sqrt(total)
    err_flux = err_counts / expt / etc_ref['flux2cps']
    return fpix, err_flux


def doppler_shift(w, velocity):
    return (1 + velocity/const.c)*w


def voigt_xsection(w, w0, f, gamma, T, mass, b=None):
    """
    Compute the absorption cross section using hte voigt profile for a line.

    Parameters
    ----------
    w : astropy quantity array or scalar
        Scalar or vector of wavelengths at which to compute cross section.
    w0: quanitity
        Line center wavelength.
    f: scalar
        Oscillator strength.
    gamma: quantity
        Sum of transition rates (A values) out of upper and lower states. Just Aul for a resonance line where only
        spontaneous decay is an issue.
    T: quantity
        Temperature of the gas. Can be None if you provide a b value instead.
    mass: quantity
        molecular mass of the gas
    b : quantity
        Doppler b value (in velocity units) of the line
    Returns
    -------
    x : quantity
        Cross section of the line at each w.
    """

    nu = const.c / w
    nu0 = const.c / w0
    if T is None:
        sigma_dopp = b/const.c*nu0/np.sqrt(2)
    else:
        sigma_dopp = np.sqrt(const.k_B*T/mass/const.c**2) * nu0
    dnu = nu - nu0
    gauss_sigma = sigma_dopp.to(u.Hz).value
    lorentz_FWHM = (gamma/2/np.pi).to(u.Hz).value
    phi = voigt(dnu.to(u.Hz).value, gauss_sigma, lorentz_FWHM) * u.s
    x = np.pi*const.e.esu**2/const.m_e/const.c * f * phi
    return x.to('cm2')


def voigt(x, gauss_sigma, lorentz_FWHM):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = gauss_sigma
    gamma = lorentz_FWHM/2.0
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)


