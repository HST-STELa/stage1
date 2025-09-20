from math import pi, nan

import numpy as np
from tqdm import tqdm
from astropy import units as u
from astropy.table import Row, QTable, Column
from matplotlib import pyplot as plt

from lya_prediction_tools import ism
from lya_prediction_tools import lya
from lya_prediction_tools import stis
from lya_prediction_tools import variability

import utilities as utils
import catalog_utilities as catutils


def opaque_tail_depth(catalog_or_row):
    catalog_or_row = catalog_or_row
    get = lambda name: catutils.get_quantity_flexible(name, catalog_or_row, catalog_or_row, True, nan)
    Mp = get('pl_bmasse')
    Ms = get('st_mass')
    a = get('pl_orbsmax')
    Rs = get('st_rad')
    mass_provenance = catutils.get_value_or_col_filled('pl_bmassprov', catalog_or_row, '')
    mass_provenance = mass_provenance.astype(str) if isinstance(mass_provenance, Column) else str(mass_provenance)
    massless_giant = (Mp > 0.41*u.Mjup) & (np.char.count(mass_provenance, 'Mass') == 0)
    Mp[massless_giant] = 1*u.Mjup
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


def opaque_tail_transit_SNR(catalog, expt_out=3500*u.s, expt_in=6000*u.s, default_rv=nan, add_jitter=False,
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
        result = list(zip(*result))
        result = list(map(u.Quantity, result))
        return result

    if show_progress:
        print('Binning out-of-transit spectra to estimate SNR.')
    w, we, fo, eo = estimate_flux_err(observed, expt_out)
    if show_progress:
        print('And now the-in-transit spectra.')
    _, _, fi, ei = estimate_flux_err(intransit, expt_in)
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

    if np.all(cSNR <= 0):
        return 0, 0, 0, np.inf

    max_idx_flat = np.argmax(cSNR)
    a_inrng, b_inrng = np.unravel_index(max_idx_flat, cSNR.shape)
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


def flat_transit_transmission(
        planet: Row,
        tgrid: u.Quantity,
        wgrid: u.Quantity,
        time_range: u.Quantity,
        rv_star: u.Quantity,
        rv_range:u.Quantity):
    timemask = utils.is_in_range(tgrid, *time_range)
    vgrid = lya.w2v(wgrid.to_value('AA')) * u.km/u.s - rv_star
    flat_depth = opaque_tail_depth(planet)

    flat_transit_wavemask = utils.is_in_range(vgrid, *rv_range)
    flat_transmission = np.ones((len(tgrid), len(wgrid)))
    flat_transmission[np.ix_(timemask, flat_transit_wavemask)] = 1 - flat_depth

    return flat_transmission, flat_depth


def generic_transit_snr(
        obstimes,
        exptimes,
        in_transit_time_range,
        baseline_time_range,
        normalization_search_rvs,
        transit_search_rvs,
        transit_timegrid,
        transit_wavegrid,
        transit_transmission_ary,
        lya_recon_wavegrid,
        lya_recon_flux,
        spectrograph_object,
        rv_star,
        rv_ism,
        rotation_period,
        rotation_amplitude,
        jitter,
        diagnostic_plots=False,
):
    """Compute transit snr iterating over the first dimension of the transmission arrays, gaining some efficiency
    by pre-computing wavelength and time gridded values and variability estimates that don't change across transit
    cases."""

    assert lya_recon_flux.ndim == 1
    if transit_transmission_ary.ndim == 2:
        transit_transmission_ary = transit_transmission_ary[None, :, :]

    if diagnostic_plots and len(transit_transmission_ary) > 20:
        raise ValueError('This would create over 40 plots. For your sanity, no.')

    # masks for observations that are in the baseline and transit ranges
    bx = utils.is_in_range(obstimes, *baseline_time_range)
    tx = utils.is_in_range(obstimes, *in_transit_time_range)

    # for later use picking integration ranges
    v0_ism = rv_ism - rv_star
    v0_ism = v0_ism.to_value('km s-1')
    normalization_search_rvs = normalization_search_rvs.to_value('km s-1')
    transit_search_rvs = transit_search_rvs.to_value('km s-1')

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

    # convenience tool to integrate fluxes
    def integrate(f, e, mask):
        F = np.sum(f[mask] * dwspec[mask])
        E = utils.quadsum(e[mask] * dwspec[mask])
        return F, E

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
    wavefigs = []
    timefigs = []
    for transit_transmission in transit_transmission_ary:
        row = {}

        # average the transits over the observation intervals using the "intergolate" function I wrote
        bin_to_obs = lambda trans: utils.intergolate(obs_edges, transit_timegrid, trans, left=1, right=1)
        trans_tbinned = np.apply_along_axis(bin_to_obs, 0, transit_transmission)
        trans_obs = trans_tbinned[obs_mask]

        # interp lya and transmission onto the same wavelength grid
        f_lya = np.interp(w, lya_recon_wavegrid, lya_recon_flux, left=0, right=0)
        interp_wave = lambda trans: np.interp(w, transit_wavegrid, trans, left=1, right=1)
        trans = np.apply_along_axis(interp_wave, 1, trans_obs)

        # multiply to get in transit fluxes
        f_lya_transit = f_lya[None,:] * trans

        # "observe" the lya line over the exposures
        obsvtns = [observe(f, expt) for f, expt in zip(f_lya_transit, exptimes.to_value('s'))]
        fobs, eobs = zip(*obsvtns)
        fobs, eobs = map(np.asarray, (fobs, eobs))

        # in and out of transit fluxes and their difference
        fb, eb = utils.flux_average(exptimes[bx, None], fobs[bx,:], eobs[bx,:], axis=0)
        ft, et = utils.flux_average(exptimes[tx, None], fobs[tx,:], eobs[tx,:], axis=0)
        tt, _ = utils.flux_average(exptimes[tx, None], trans_obs[tx,:], 0, axis=0)
        d = fb - ft
        de = utils.quadsum(np.vstack((eb, et)), axis=0)

        if np.all(d < 1e-25):
            negligible_transit = True
            row['transit ranges'] = np.ma.masked
            row['normalization ranges'] = np.ma.masked
            row['transit sigma'] = 0
            row['normalized'] = np.ma.masked
            rows.append(row)

        else:
            negligible_transit = False

            # pick transit and normalization integration ranges based on max SNR
            normrng_mask = max_snr_red_blue(vspec, fb, eb, *normalization_search_rvs)
            if v0_ism >= transit_search_rvs[0] and v0_ism <= transit_search_rvs[1]:
                # if mid-ism absorption falls within the transit range
                absprng_mask = max_snr_red_blue(vspec, d, de,
                                                (transit_search_rvs[0], v0_ism),
                                                (v0_ism, transit_search_rvs[1]))
            else:
                a, b, _, _ = max_snr_integration_range(vspec, d, de, transit_search_rvs)
                absprng_mask = np.zeros(len(fb), bool)
                absprng_mask[a:b] = True

            Fout, Eout = integrate(fb, eb, absprng_mask)
            Ftran, Etran = integrate(ft, et, absprng_mask)
            Fnorm, Enorm = integrate(fb, eb, normrng_mask)

            # get associated ranges to store in the results table
            i_absp = utils.chunk_edges(absprng_mask)
            i_norm = utils.chunk_edges(normrng_mask)
            v_absp = [(vspec_bins[ii[0]], vspec_bins[ii[1]+1]) for ii in i_absp]
            v_norm = [(vspec_bins[ii[0]], vspec_bins[ii[1]+1]) for ii in i_norm]
            row['transit ranges'] = v_absp
            row['normalization ranges'] = v_norm

            # add noise from stellar/instrument variability
            terms = np.array([Etran, Ftran*variability_scatter])
            E_var = utils.quadsum(terms)

            # versus noise from trying to normalize out that variability
            Eratio = np.sqrt(2) * Enorm/Fnorm
            terms = np.array([Etran, Ftran*Eratio])
            E_norm = utils.quadsum(terms)

            # and pick the smaller of the two
            if E_var < E_norm:
                row['normalized'] = False
                Etrans_min = E_var
            else:
                row['normalized'] = True
                Etrans_min = E_norm

            # detection significance
            dF = Fout - Ftran
            terms = np.array((Eout, Etrans_min))
            dE = utils.quadsum(terms)
            sigma = dF/dE

            row['transit sigma'] = sigma.to_value('')
            rows.append(row)

        if diagnostic_plots:
            # region wavelength diagnostic plot
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel('Velocity in System Frame (km s-1)')
            ax.set_ylabel('Flux Density (erg s-1 cm-2 Å-1)')
            ax2 = ax.twinx()
            ax2.set_ylabel('Transit Transmission')

            fbln = ax.errorbar(vspec, fb, eb, errorevery=5, label='baseline')
            ftln = ax.errorbar(vspec, ft, et, fmt='-C2', errorevery=5, label='transit')
            ag, = ax.plot(vspec, et, ':C3', label='uncty with airglow')

            transit_vgrid = lya.w2v(transit_wavegrid) - rv_star.to_value('km s-1')
            for i, t in enumerate(obstimes):
                ax2.plot(transit_vgrid, trans_obs[i,:], lw=0.5, color='0.5', label=f'{t.to('h'):.1f}')
            tln, = ax2.plot(transit_vgrid, tt, color='0.5', label='transit transmission')
            ax2.set_ylim(-0.01, 1.05)

            # integration ranges
            if negligible_transit:
                v_norm = (transit_search_rvs,)
                v_absp = normalization_search_rvs
                absprng_mask = utils.is_in_range(vspec, *transit_search_rvs)
                ax.annotate('Transit does not register.\nDefault integration ranges shown.', xy=(0.5, 0.5), ha='center', va='center')
            for rng in v_norm:
                nspan = ax.axvspan(*rng, color='0.5', alpha=0.2, ls='none', label='normalization')
            for rng in v_absp:
                aspan = ax.axvspan(*rng, color='C2', alpha=0.2, ls='none', label='transit')

            ax.legend(handles=(fbln, ftln, ag, nspan, aspan))
            ax2.legend(handles=(tln,))
            wavefigs.append(fig)
            # endregion

            # region time diagnostic plot
            fig, axs = plt.subplots(1, 2, figsize=(8,4), width_ratios=[2/5, 1])
            fig.supxlabel('Time from Mid-Transit (h)')
            fig.supylabel('Normalized flux (h)')

            Fout, Eout = integrate(fb, eb, absprng_mask)
            result = [integrate(ff, ee, absprng_mask) for ff,ee in zip(fobs,eobs)]
            Fs, Es = zip(*result)
            Fs, Es = np.array((Fs, Es))

            for ax, x in zip(axs, (bx, ~bx)):
                ta, tb =  obs_edges[::2][x][0] - 0.25, obs_edges[1::2][x][-1] + 0.25
                buffer = (tb - ta) * 0.05
                ax.set_xlim(ta - buffer, tb + buffer)

                ax.errorbar(obstimes.to_value('h'), Fs/Fout, Es/Fout, exptimes.to_value('h')/2, fmt='o')
                ax.axhline(1, color='0.5', lw=0.5, ls=':')

            int_edges = np.sort(np.hstack((obs_edges[::2][tx], obs_edges[1::2][tx])))
            int_y = np.repeat(Fs[tx]/Fout, 2)
            ax.fill_between(int_edges, int_y, 1.0, color='0.5', alpha=0.3, lw=0)

            # time ranges
            for tedge in in_transit_time_range.to_value('h'):
                axs[-1].axvline(tedge, color='0.5', ls='--')

            if negligible_transit:
                ax.annotate('Transit does not register.', xy=(0.5, 0.5), ha='center', va='center')

            # stellar rotation
            worst_phase = 3*np.pi/4
            dt = 0.1
            tsmpl = np.arange(axs[0].get_xlim()[0], axs[1].get_xlim()[1] + dt, dt) * u.h
            yrot = variability.rotation(tsmpl, rotation_amplitude, rotation_period, worst_phase)
            tbase = np.mean(obstimes[bx])
            ybase = np.interp(tbase, tsmpl, yrot)
            yrot_offset = yrot - ybase + 1
            for ax in axs:
                rotln, = ax.plot(tsmpl, yrot_offset, '-C1')

            # jitter
            for ax in axs:
                ax.plot(tsmpl, yrot_offset + jitter, ':C1')
                jitterln, = ax.plot(tsmpl, yrot_offset - jitter, ':C1')

            labels = (f'{rotation_amplitude*100:.1f}% stellar rotation, Prot = {rotation_period:.1f}',
                      f'{jitter*100:.1f}% instrument+stellar jitter')
            axs[1].legend((rotln, jitterln), labels)

            timefigs.append(fig)
            # endregion

    tbl = QTable(rows=rows)
    if diagnostic_plots:
        return tbl, wavefigs, timefigs
    else:
        return tbl

    # footnote: algebra for the noise from normalizing
    # Ft_normed = Fin * Fnorm_out/Fnorm_in = Fin * normratio
    # E_ratio = sqrt( (Enorm_out/Fnorm_in)**2 + (Enorm_in * Fnorm_out/Fnorm_in**2)**2)
    #         = sqrt( (normratio * Enorm_out/Fnorm_out)**2 + (normratio * Enorm_in/Fnorm_in)**2 )
    # assume Enorm_out/Fnorm_out = Enorm_in/Fnorm_in, normratio = 1
    # E_ratio = sqrt( 2 * (Enorm_out/Fnorm_out)**2)
    #         = sqrt(2) * Enorm_out/Fnorm_out
    # E_normed = sqrt( (normratio * Ein)**2 + (Fin * Eratio) ** 2)