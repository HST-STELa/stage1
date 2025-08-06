"""
These use an error estimation method that has a bug which zeros out the
error in the airglow region of the spectrum.
"""

from math import pi

import numpy as np
from astropy import units as u
from tqdm import tqdm

import empirical
from lya_prediction_tools import lya
import utilities as utils
from lya_prediction_tools import ism


def is_pos_real(cat, col):
    if hasattr(cat[col], 'mask'):
        return ~cat[col].mask & (cat[col].filled(-999) > 0)
    else:
        return cat[col] > 0


def lya_flux_estimate(cat):
    isgood = lambda col: is_pos_real(cat, col)

    filled = np.zeros(len(cat), bool)
    Flya_au_adopted = np.zeros(len(cat))

    # the original target search did not use any GALEX mags, so I removed the corresponding block

    # Lya based on Teff from Linsky+ 2013
    rep = isgood('st_teff') & isgood('st_rotp') & ~filled
    filled = filled | rep
    print('Using Teff per Linsky+ 2013 to predict Lya for {} rows.'.format(sum(rep)))
    Flya_au = empirical.Lya_from_Teff_linsky13(cat['st_teff'], cat['st_rotp'])
    Flya_au_adopted[rep] = Flya_au[rep]

    # Lya based on Teff from Schneider
    rep = isgood('st_teff') & ~filled
    print('Using Teff per Schneider+ 2019 to predict Lya for {} rows.'.format(sum(rep)))
    Teff = cat['st_teff'].quantity.value
    logF = 6.754e-4 * Teff - 2.639
    Flya_au = 10**logF
    Flya_au_adopted[rep] = Flya_au[rep]

    return Flya_au_adopted * u.Unit('erg s-1 cm-2')


def sim_g140m_obs(f, expt):
    fpix = utils.intergolate(lya.we_etc, lya.wgrid_std, f)
    src = fpix * lya.etc_ref['flux2cps'] * expt
    bkgnd = lya.etc_ref['bkgnd_cps'] * expt
    total = bkgnd + src
    with np.errstate(divide='ignore', invalid='ignore'):
        err_counts = np.sqrt(total)
        err_flux = err_counts / src * fpix
        err_flux[~np.isfinite(err_flux)] = 0
    return fpix, err_flux


def lya_profile_at_earth_auto(catalog_row, n_H, lya_factor):
    planet = catalog_row
    Flya = planet['Flya_at_earth'] * u.Unit('erg s-1 cm-2') * lya_factor
    rv_star = planet['st_radv'] * u.km / u.s
    rv_ism = ism.ism_velocity(planet['ra'] * u.deg, planet['dec'] * u.deg)
    Nh = n_H * planet['sy_dist'] * u.pc
    observed = lya.lya_with_ISM(lya.wgrid_std, Flya, rv_star, rv_ism, Nh, lya.Tism)
    return observed


def depth_extended_hill_transit(planet_row):
    s = planet_row
    Mp = s['pl_bmasse'] * u.Mearth
    if Mp > 0.41*u.Mjup and s['pl_bmasse'] not in ['Mass', 'Msini']: # based on Chen and Kipping 2017
        Mp = 0.41*u.Mjup
    Ms = s['st_mass'] * u.Msun
    a = s['pl_orbsmax'] * u.AU
    Rs = s['st_rad'] * u.Rsun
    Rhill = (a * (Mp / 3 / Ms)**(1./3)).to('Rsun')
    Ahill = Rhill*2*Rs*2
    depth = Ahill/(2*pi*Rs**2)
    depth = depth.to('')
    depth[depth > 1] = 1
    return depth


def transit_snr(planet, expt_out=3500, expt_in=6000, add_jitter=False, optimistic=False, pessimistic=False):
    from copy import copy
    if (optimistic and add_jitter) or (pessimistic and add_jitter):
        raise ValueError
    wgrid = np.arange(1210., 1220., 0.005) * u.AA
    ism_columns = ism.ism_columns

    def jitter(value, err):
        return np.random.randn(1)*err + value
    def jitter_dex(value, errdex):
        return value*10**(np.random.randn(1) * errdex)

    if add_jitter:
        n_H = np.random.choice(ism_columns['n_H'])
        lya_factor = jitter_dex(1, 0.2)
    else:
        n_H = np.median(ism_columns['n_H'])
        lya_factor = 1
    if optimistic:
        n_H = np.percentile(ism_columns['n_H'], 16)
        lya_factor = 10**0.2
    if pessimistic:
        n_H = np.percentile(ism_columns['n_H'], 95)
        lya_factor = 10**-0.4
    n_H /= u.cm**3

    v_integrate = [-150, 100]

    rv_star = planet['st_radv'] * u.km / u.s
    v_trans, d_trans_temp = np.loadtxt('generic_transit_absorption_spectrum.csv', delimiter=',').T
    # broaden the absorption profile to better match GJ 436 transit
    v_trans *= 2
    d_trans_norm = d_trans_temp / np.max(d_trans_temp)
    v = (wgrid / lya.wlab_H - 1) * 3e5 - rv_star.value
    v_etc_PF = lya.v_etc - rv_star.value

    # RVing the planet parameters, if needed
    if add_jitter:
        planet = copy(planet)
        def jittercol(col, default_dex):
            if np.ma.is_masked(planet[col+'err1']):
                planet[col] = jitter_dex(planet[col], default_dex)
            else:
                planet[col] = jitter(planet[col], planet[col+'err1'])
        rdex = 0.075 if planet['pl_rade'] > 1.23 else 0.045
        jittercol('pl_bmasse', rdex)
        jittercol('st_mass', 0.04)
        jittercol('pl_orbsmax', 0.04)
        jittercol('st_rad', 0.04)

    max_depth = depth_extended_hill_transit(planet)
    d_trans = max_depth * d_trans_norm
    observed = lya_profile_at_earth_auto(planet, n_H, lya_factor)
    v_blue = v[np.argmax(observed[v < 0])]
    v_trans_shifted = v_trans + v_blue
    transit_depth = np.interp(v, v_trans_shifted, d_trans)
    intransit = observed * (1 - transit_depth)
    fo, eo = sim_g140m_obs(observed, expt_out)
    fi, ei = sim_g140m_obs(intransit, expt_in)
    d = fo - fi
    e = np.sqrt(eo ** 2 + ei ** 2)

    integrate = (v_etc_PF > v_integrate[0]) & (v_etc_PF < v_integrate[1])
    D = np.sum(d[integrate])
    E = np.sqrt(np.sum(e[integrate]**2))
    return D/E


def depth_extended_hill_transit_proposal(planet_row):
    s = planet_row
    Mp = s['pl_bmasse'] * u.Mearth
    Ms = s['st_mass'] * u.Msun
    a = s['pl_orbsmax'] * u.AU
    Rs = s['st_rad'] * u.Rsun
    Rhill = (a * (Mp / 3 / Ms)**(1./3)).to('Rsun')
    Ahill = Rhill*2*Rs*2
    depth = Ahill/(2*pi*Rs**2)
    depth = depth.to('')
    depth[depth > 1] = 1
    return depth


def transit_snr_proposal(cat, case='nominal'):
    # estimate SNR of transit in absorbed ISM profile
    if case == 'nominal':
        n_H = 0.025 / u.cm ** 3  # mid value
    elif case == 'optimistic':
        n_H = 0.01/u.cm**3
    else:
        raise ValueError
    Tism = 10000 * u.K
    wgrid = np.arange(1210., 1220., 0.005) * u.AA

    etc_flux = 1e-14
    etc_src = 105 * 2 / 10000.
    etc_bg = 10.5 * 2 / 10000.

    v_integrate = [-150, 100]

    v_trans, d_trans = np.loadtxt('generic_transit_absorption_spectrum.csv', delimiter=',').T
    d_trans_norm = d_trans / np.max(d_trans)

    SNRs = []
    for i in tqdm(list(range(0, len(cat)))):
        star = cat[i]
        # scale depth to match extended hill sphere transit
        max_depth = depth_extended_hill_transit_proposal(star)
        d_trans = max_depth * d_trans_norm
        Nh = n_H*star['sy_dist'] * u.pc
        rv_ism = ism.ism_velocity(star['ra'] * u.deg, star['dec'] * u.deg)
        rv_star = star['st_radv'] * u.km/u.s
        if np.ma.is_masked(rv_star):
            rv_star = rv_ism # assume no offset between star and ISM
        Flya = star['Flya_at_earth'] * u.Unit('erg s-1 cm-2')
        v = (wgrid / lya.wlab_H - 1) * 3e5 - rv_star.value

        intrinsic = lya.reversed_lya_profile(wgrid, rv_star, Flya)
        transmitted = lya.transmission(wgrid, rv_ism, Nh, Tism)
        observed = intrinsic * transmitted
        transit_depth = np.interp(v, v_trans, d_trans)
        transit = observed * (1 - transit_depth)

        src_cps = observed.value/etc_flux * etc_src
        src_cnts = src_cps* 3 * 2700.
        bg_cnts = etc_bg * 3 * 2700.
        err = np.sqrt(src_cnts + bg_cnts)/src_cnts*observed
        err[np.isnan(err)] = 0

        mask = (v > v_integrate[0]) & (v < v_integrate[1])
        Fout = np.trapezoid(observed[mask], v[mask])
        Fin = np.trapezoid(transit[mask], v[mask])
        Eout = np.sqrt(np.trapezoid(err[mask]**2, v[mask]))
        dF = Fout - Fin
        dE = np.sqrt(2)*Eout
        SNR = dF/dE

        SNRs.append(SNR)

    return SNRs