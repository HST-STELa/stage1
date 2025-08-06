from math import pi, nan

import numpy as np
from tqdm import tqdm
from astropy import units as u
from astropy import table as Table
from matplotlib import pyplot as plt
import h5py

import utilities as utils

from lya_prediction_tools import ism
from lya_prediction_tools import lya
from lya_prediction_tools import stis


def opaque_tail_depth(catalog):
    Mp = catalog['pl_bmasse'].filled(nan).quantity
    mass_provenance = catalog['pl_bmassprov'].filled('').astype(str)
    massless_giant = (Mp > 0.41*u.Mjup) & (np.char.count(mass_provenance, 'Mass') == 0)
    Mp[massless_giant] = 1*u.Mjup
    Ms = catalog['st_mass'].filled(nan).quantity
    a = catalog['pl_orbsmax'].filled(nan).quantity
    Rs = catalog['st_rad'].filled(nan).quantity
    Rhill = (a * (Mp / 3 / Ms)**(1./3))
    Rhill = Rhill.to('Rsun')
    Ahill = Rhill*2*Rs*2 # rectangular transit to approximate an opaque tail
    depth = Ahill/(2*pi*Rs**2)
    depth = depth.to('')
    depth[depth > 1] = 1
    return depth


def opaque_tail_transit_profile(catalog, default_rv=nan, add_jitter=False,
                                n_H_percentile=50, lya_percentile=50, lya_1AU_colname='Flya_1AU_adopted',
                                transit_range=(-100, 30), show_progress=False):
    if add_jitter:
        needed_columns = (f'ra dec sy_dist st_radv st_rad st_mass pl_orbsmax pl_bmasse pl_rade pl_bmasseerr1 '
                          f'st_masserr1 pl_orbsmaxerr1 st_raderr1 {lya_1AU_colname}').split()
        cat = catalog[needed_columns].copy()
    else:
        cat = catalog
    n = len(cat)
    if (n_H_percentile and add_jitter) or (n_H_percentile and add_jitter):
        raise ValueError
    wgrid = lya.wgrid_std

    if add_jitter:
        n_H = np.random.choice(ism.ism_columns['n_H'], n, replace=True)
        lya_factor = utils.jitter_dex(np.ones(n), 0.2)
    else:
        n_H = ism.ism_n_H_percentile(n_H_percentile)
        lya_factor = lya.lya_factor_percentile(lya_percentile)

    rv_star = lya.__default_rv_handler(cat, default_rv)

    # shift velocity grids for use in ETC calcs based on stellar rv
    v_starframe = (wgrid[None,:] / lya.wlab_H - 1) * 3e5 - rv_star.value[:, None]

    # RVing the planet parameters, if needed
    if add_jitter:
        def jittercol(col, default_dex):
            missing_error = cat[f'{col}err1'].mask
            cat[col][missing_error] = utils.jitter_dex(cat[col][missing_error], default_dex[missing_error])
            cat[col][~missing_error] = utils.jitter(cat[col][~missing_error], cat[f'{col}err1'][~missing_error])
        ones = np.ones(len(cat))
        rdex = ones*0.075
        rdex[cat['pl_rade'] <= 1.23] = 0.045
        jittercol('pl_bmasse', rdex)
        jittercol('st_mass', 0.04*ones)
        jittercol('pl_orbsmax', 0.04*ones)
        jittercol('st_rad', 0.04*ones)

    max_depth = opaque_tail_depth(cat)

    if show_progress:
        print('Estimating Lya profiles.')
    observed = lya.lya_at_earth_auto(cat, n_H, default_rv=default_rv,
                                     lya_factor=lya_factor, lya_1AU_colname=lya_1AU_colname,
                                     show_progress=show_progress)

    in_transit_rng = (v_starframe > transit_range[0]) & (v_starframe < transit_range[1])
    transit_depth = np.where(in_transit_rng, max_depth[:,None], 0)

    # apply transit
    intransit = observed * (1 - transit_depth)
    return v_starframe, observed, intransit


def opaque_tail_transit_plots(catalog, default_rv=nan, add_jitter=False,
                              n_H_percentile=50, lya_percentile=50, lya_1AU_colname='Flya_1AU_adopted',
                              transit_range=(-100, 30), show_progress=False):
    v_starframe, observed, intransit = opaque_tail_transit_profile(catalog, default_rv, add_jitter, n_H_percentile,
                                                                   lya_percentile, lya_1AU_colname, transit_range, show_progress)
    rv_star = lya.__default_rv_handler(catalog, default_rv)
    ra = catalog['ra'].filled(nan).quantity
    dec = catalog['dec'].filled(nan).quantity
    rv_ism = ism.ism_velocity(ra, dec)

    for i, row in enumerate(catalog):
        plt.figure()
        plt.title(row['id'])
        infobox = (
            f"system rv {rv_star[i]:.1f}"
            f"\nISM rv {rv_ism[i]:.1f}"
            f"\nsystem distance {row['sy_dist']:.1f} pc"
            f"\nFUV-based Flya {row['Flya_1AU_fuv']:.1f}"
            f"\nNUV-based Flya {row['Flya_1AU_nuv']:.1f}"
            f"\nTeff-linsky based Flya {row['Flya_1AU_Teff_linsky']:.1f}"
            f"\nTeff-schneider based Flya {row['Flya_1AU_Teff_schneider']:.1f}"
            f"\nadopted Flya {row['Flya_1AU_adopted']:.1f}"
            f"\nFlya at earth no ism {row['Flya_earth_no_ISM']:.1e}"
        )
        plt.plot(v_starframe[i], observed[i])
        plt.plot(v_starframe[i], intransit[i])
        plt.annotate(infobox, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize='small', )


def opaque_tail_transit_SNR(catalog, expt_out=3500, expt_in=6000, default_rv=nan, add_jitter=False,
                            n_H_percentile=50, lya_percentile=50, lya_1AU_colname='Flya_1AU_adopted',
                            transit_range=(-100,30), integrate_range='best', show_progress=False):

    _, observed, intransit = opaque_tail_transit_profile(catalog, default_rv, add_jitter, n_H_percentile, lya_percentile,
                                                         lya_1AU_colname, transit_range, show_progress)

    rv_star = lya.__default_rv_handler(catalog, default_rv)

    # apply transit
    def estimate_flux_err(specs, expt):
        if show_progress:
            specs = tqdm(specs)
        result = [stis.simple_sim_g140m_obs(spec, expt) for spec in specs]
        result = np.asarray(result)
        result = np.moveaxis(result, 1, 0)
        return result

    if show_progress:
        print('Binning out-of-transit spectra to estimate SNR.')
    w, we, fo, eo = estimate_flux_err(observed, expt_out)
    if show_progress:
        print('And now the-in-transit spectra.')
    fi, ei = estimate_flux_err(intransit, expt_in)
    d = fo - fi
    e = np.sqrt(eo ** 2 + ei ** 2)

    if integrate_range == 'best':
        # assume we only integrate areas where SNR is above some cut
        snr_cut = 1/3
        spec_snr = d/e
        max_snr = np.max(spec_snr, 1)
        integrate = spec_snr > (max_snr[:,None] * snr_cut)
    elif hasattr(integrate_range, '__iter__'):
        v_etc_earth_frame = lya.w2v(w)
        v_etc_planet_frame = v_etc_earth_frame - rv_star.value[:, None]
        integrate = (v_etc_planet_frame > integrate_range[0]) & (v_etc_planet_frame < integrate_range[1])
    else:
        raise ValueError

    d_for_integrating = np.where(integrate, d, 0)
    e_for_integrating = np.where(integrate, e, 0)
    D = np.sum(d_for_integrating, axis=1)
    E = np.sqrt(np.sum(e_for_integrating**2, axis=1))
    return D/E


def model_transit_snr(
        transit_simulation_file,
        lya_reconstruction_file,
        obstimes,
        exptimes,
        spectrograph_object,
        rotation_period,
        rotation_amplitude,
        jitter,
):


    lya_recon = Table.read(lya_reconstruction_file)
    lya_cases = 'low_2sig low_1sig median high_1sig high_2sig'.split()
    wlya = lya_recon['wave_lya']

    with h5py.File(transit_simulation_file) as f:
        tsim = f['tgrid'][:]
        wsim = f['wavgrid'][:]
        transmission_array = f['intensity'][:]

    # there seem to be odd "zero-point" offsets in some of the transmission vectors. correct these
    transmaxs = np.max(transmission_array, axis=2)
    offsets = 1 - transmaxs
    transmission_corrected = transmission_array + offsets[:,:,None]

    # figure out the start and end times of the observations so I can bin over them later
    obs_start = obstimes - exptimes/2
    obs_end = obstimes + exptimes/2
    obs_edges = np.hstack((obs_start, obs_end))
    obs_edges = np.sort(obs_edges)
    obs_edges = obs_edges.to_value('h')
    obs_mask = np.ones(len(obs_edges) - 1, bool)
    obs_mask[1::2] = False

    # merge the two wavelength grids
    w = np.sort(np.hstack((wlya, wsim)))

    # function to apply LSF and estimate uncties
    observe = spectrograph_object.fast_observe_function(w)

    snr, wa, wb = [], [], []
    for transmission_raw in transmission_corrected:
        for lya_case in lya_cases:
            # average the transits over the observation intervals using the "intergolate" function I wrote
            bin_to_obs = lambda trans: utils.intergolate(obs_edges, tsim, trans, left=1, right=1)
            trans_tbinned = np.apply_along_axis(bin_to_obs, 0, transmission_raw)
            trans_obs = trans_tbinned[obs_mask]

            # interp lya and transmission onto the same wavelength grid
            lya_raw = lya_recon[f'lya_intrinsic unconvolved_{lya_case}']
            lya = np.interp(w, wlya, lya_raw, left=0, right=0)
            interp_wave = lambda trans: np.interp(w, wsim, trans, left=1, right=1)
            trans = np.apply_along_axis(interp_wave, 1, trans_obs)

            # get stellar variability vector

            # multiply to get in transit fluxes
            lya_transit = lya[None,:] * trans

            # "observe" each spectrum
            obsvtns = [observe()]
            result = np.apply_along_axis(obs_std, 1, lya_transit_xp)