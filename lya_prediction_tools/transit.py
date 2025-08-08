import warnings
from math import pi, nan

import numpy as np
from tqdm import tqdm
from astropy import units as u
from astropy.table import Table
from matplotlib import pyplot as plt
import h5py

import utilities as utils

from lya_prediction_tools import ism
from lya_prediction_tools import lya
from lya_prediction_tools import stis
from lya_prediction_tools import variability


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


def max_snr_integration_range(x, f, e, search_rng=None):
    n = len(x)
    if search_rng is None:
        inrng = np.ones(len(f), bool)
    else:
        inrng = utils.is_in_range(x, *search_rng)

    if np.all(f[inrng] == 0):
        return 0, 0, 0, np.inf

    inrng_indices = np.flatnonzero(inrng)

    cF = utils.subinterval_cumsums(f[inrng])
    cE = np.sqrt(utils.subinterval_cumsums(e[inrng]**2))
    with np.errstate(invalid='ignore'):
        cSNR = cF/cE
        cSNR[cF == 0] = 0
    max_idx_flat = np.argmax(cSNR)
    a_inrng, b_inrng = np.unravel_index(max_idx_flat, (n+1, n+1))
    true_indices = inrng_indices[a_inrng:b_inrng]
    a_true = true_indices[0]
    b_true = true_indices[-1] + 1

    F = cF.flat[max_idx_flat]
    E = cE.flat[max_idx_flat]
    return a_true, b_true, F, E


def max_snr_red_blue(x, f, e, blue_rng, red_rng):
    masks, Fs, Es = [], [], []
    for rng in (blue_rng, red_rng):
        a, b, F, E = max_snr_integration_range(x, f, e, rng)
        mask = np.zeros(len(f), bool)
        mask[a:b] = True
        masks.append(mask)
        Fs.append(F)
        Es.append(E)

    Fs, Es = np.array(Fs), np.array(Es)
    SNRs = Fs/Es
    imx = np.argmax(SNRs)

    # either pick a range or combine them based on snr
    SNRcombo = sum(Fs)/utils.quadsum(Es)
    if SNRcombo > SNRs[imx]:
        return masks[0] | masks[1]
    else:
        return masks[imx]


def generic_transit_snr(
        obstimes,
        exptimes,
        in_transit_time_range,
        baseline_time_range,
        normalization_within_rvs,
        transit_within_rvs,
        transit_timegrid,
        transit_wavegrid,
        transit_transmission_ary,
        lya_recon_wavegrid,
        lya_recon_flux_ary,
        spectrograph_object,
        rv_star,
        rv_ism,
        rotation_period,
        rotation_amplitude,
        jitter,
):
    if lya_recon_flux_ary.ndim == 1:
        lya_recon_flux_ary = lya_recon_flux_ary[None,:]
    if transit_transmission_ary.ndim == 2:
        transit_transmission_ary = transit_transmission_ary[None, :, :]
    assert lya_recon_flux_ary.shape[0] == transit_transmission_ary.shape[0]

    # masks for observations that are in the baseline and transit ranges
    bx = utils.is_in_range(obstimes, *baseline_time_range)
    tx = utils.is_in_range(obstimes, *in_transit_time_range)

    # for later use picking integration ranges
    v0_ism = rv_ism - rv_star
    v0_ism = v0_ism.to_value('km s-1')
    normalization_within_rvs = normalization_within_rvs.to_value('km s-1')
    transit_within_rvs = transit_within_rvs.to_value('km s-1')

    # instrument wavelength and velocity grids
    dwspec = spectrograph_object.binwidth
    wspec = spectrograph_object.wavegrid
    vspec = lya.w2v(wspec.value) - rv_star.to_value('km s-1') # in star frame
    vspec_bins = utils.mids2bins(vspec)

    # figure out the start and end times of the observations so I can bin over them later
    obs_start = obstimes - exptimes/2
    obs_end = obstimes + exptimes/2
    obs_edges = np.hstack((obs_start, obs_end))
    obs_edges = np.sort(obs_edges)
    obs_edges = obs_edges.to_value('h')
    obs_mask = np.ones(len(obs_edges) - 1, bool)
    obs_mask[1::2] = False

    # merge the two wavelength grids
    w = np.sort(np.hstack((lya_recon_wavegrid, transit_wavegrid)))

    # function to apply LSF and estimate uncties
    observe = spectrograph_object.fast_observe_function(w)

    # scatter added due to stellar variability if not normalizing by line wings, in relative units
    np.random.seed(20250807)
    variability_scatter = variability.added_transit_uncertainty(
        obstimes,
        exptimes,
        jitter,
        rotation_period,
        rotation_amplitude,
        in_transit_time_range,
        baseline_time_range,
        1000
    )

    rows = []
    for lya_recon_flux, transit_transmission in zip(lya_recon_flux_ary, transit_transmission_ary):
        row = {}

        # average the transits over the observation intervals using the "intergolate" function I wrote
        bin_to_obs = lambda trans: utils.intergolate(obs_edges, transit_timegrid, trans, left=1, right=1)
        trans_tbinned = np.apply_along_axis(bin_to_obs, 0, transit_transmission)
        trans_obs = trans_tbinned[obs_mask]

        # interp lya and transmission onto the same wavelength grid
        f_lya = np.interp(w, lya_recon_wavegrid, lya_recon_flux, left=0, right=0)
        interp_wave = lambda trans: np.interp(w, transit_wavegrid, trans, left=1, right=1)
        trans = np.apply_along_axis(interp_wave, 1, trans_obs)

        # "observe" the line over the baseline exposures
        exptime_baseline = np.sum(exptimes[bx])
        fb, eb = observe(f_lya, exptime_baseline.to_value('s'))

        # pick transit and normalization integration ranges
        normrng_mask = max_snr_red_blue(vspec, fb, eb, *normalization_within_rvs)
        if v0_ism >= transit_within_rvs[0] and v0_ism <= transit_within_rvs[1]:
            absprng_mask = max_snr_red_blue(vspec, fb, eb,
                                            (transit_within_rvs[0], v0_ism),
                                            (v0_ism, transit_within_rvs[1]))
        else:
            a, b, _, _ = max_snr_integration_range(vspec, fb, eb, transit_within_rvs)
            absprng_mask = np.zeros(len(fb), bool)
            absprng_mask[a:b] = True

        # normalization flux and error
        Fnorm = np.sum(fb[normrng_mask] * dwspec[normrng_mask])
        Enorm = utils.quadsum(eb[normrng_mask] * dwspec[normrng_mask])

        # get associated ranges to store in the results table
        i_absp = utils.chunk_edges(absprng_mask)
        i_norm = utils.chunk_edges(normrng_mask)
        v_absp = [(vspec_bins[ii[0]], vspec_bins[ii[1]+1]) for ii in i_absp]
        v_norm = [(vspec_bins[ii[0]], vspec_bins[ii[1]+1]) for ii in i_norm]
        row['transit ranges'] = v_absp
        row['normalization ranges'] = v_norm

        # multiply to get in transit fluxes
        f_lya_transit = f_lya[None,:] * trans

        # "observe" the lya line over the exposures
        obsvtns = [observe(f, expt) for f, expt in zip(f_lya_transit, exptimes.to_value('s'))]
        fobs, eobs = zip(*obsvtns)
        fobs, eobs = map(np.asarray, (fobs, eobs))

        # in and out of transit fluxes and errors
        Fs, Es = [], []
        for x in (bx, tx):
            f, e = utils.flux_average(exptimes[x, None], fobs[x,:], eobs[x,:], axis=0)
            F = np.sum(f[absprng_mask] * dwspec[absprng_mask])
            E = utils.quadsum(e[absprng_mask] * dwspec[absprng_mask])
            Fs.append(F)
            Es.append(E)
        Fb, Eb = Fs[0], Es[0]
        Ft, Et = Fs[1], Es[1]

        # add noise from stellar variability
        terms = np.array([Et, Ft*variability_scatter])
        Et_var = utils.quadsum(terms)

        # versus noise from trying to normalize out that variability
        Eratio = np.sqrt(2) * Enorm/Fnorm
        terms = np.array([Et, Ft*Eratio])
        Et_norm = utils.quadsum(terms)

        # and pick the smaller of the two
        if Et_var < Et_norm:
            row['normalized'] = False
            Et_min = Et_var
        else:
            row['normalized'] = True
            Et_min = Et_norm

        # detection significance
        dF = Fb - Ft
        terms = np.array((Eb, Et_min))
        dE = utils.quadsum(terms)
        sigma = dF/dE

        row['transit sigma'] = sigma.to_value('')
        rows.append(row)

    tbl = Table(rows=rows)
    return tbl

    # footnote: algebra for the noise from normalizing
    # Ft_normed = Fin * Fnorm_out/Fnorm_in = Fin * normratio
    # E_ratio = sqrt( (Enorm_out/Fnorm_in)**2 + (Enorm_in * Fnorm_out/Fnorm_in**2)**2)
    #         = sqrt( (normratio * Enorm_out/Fnorm_out)**2 + (normratio * Enorm_in/Fnorm_in)**2 )
    # assume Enorm_out/Fnorm_out = Enorm_in/Fnorm_in, normratio = 1
    # E_ratio = sqrt( 2 * (Enorm_out/Fnorm_out)**2)
    #         = sqrt(2) * Enorm_out/Fnorm_out
    # E_normed = sqrt( (normratio * Ein)**2 + (Fin * Eratio) ** 2)