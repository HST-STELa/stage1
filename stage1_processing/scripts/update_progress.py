from datetime import datetime
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.optimize import root_scalar, least_squares

from astropy import table
from astropy import time
from astropy.io import fits
from astropy import constants as const
from astropy import units as u

import database_utilities as dbutils
import paths
import catalog_utilities as catutils
import utilities as utils
import hst_utilities as hutils

from lya_prediction_tools import lya, ism, stis
from stage1_processing import preloads
from stage1_processing import target_lists


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets_for_lya_flux_calcs = target_lists.observed_since('2025-09-04', type='lya')
saveplots = True
have_a_look = True
obs_filters = dict(targets=targets_for_lya_flux_calcs, instruments=['hst-stis-g140m', 'hst-stis-e140m'], directory=paths.data_targets)

#%% plot backend

if have_a_look:
    mpl.use('qt5agg') # plots are shown
else:
    mpl.use('Agg') # plots in the backgrounds so new windows don't constantly interrupt my typing

#%% tics

tics = preloads.stela_names.loc['hostname_file', targets_for_lya_flux_calcs]['tic_id']

#%% --- PROGRESS TABLE UPDATES ---
pass

#%% copy progress table for revision

prog_update = preloads.progress_table.copy()

#%% copy planets table

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    planets = preloads.planets.copy()
planets.add_index('tic_id')


#%% load and filter latest target build

with catutils.catch_QTable_unit_warnings():
    cat = preloads.planets.copy()

stg1_mask = (
    cat['stage1'].filled(False)  # either selected for stage1
    | cat['stage1_backup'].filled(False)  # backup for stage1
    | cat['external_lya'].filled(False)  # or archival
)


# print an FYI about previously selected, now unselected targets
stg1_tics = cat['tic_id'][stg1_mask]
lost_target_mask = ~np.isin(prog_update['TIC ID'], stg1_tics)
lost_tics = prog_update['TIC ID'][lost_target_mask]
if len(lost_tics) > 0:
    print()
    print('Some targets in the Observing Progress table are no longer in the stage1 or stage1_backup samples but'
          ' will be kept for record keeping. They are:')
    lost_mask = np.isin(cat['tic_id'], lost_tics)
    lost = cat[lost_mask]
    viewcols = ['pl_name', 'toi', 'tic_id', 'transit_snr_nominal', 'stage1_rank', 'decision']
    lost[viewcols].pprint(-1,-1)
    print()
    mask = stg1_mask | lost_mask
else:
    mask = stg1_mask

roster = cat[mask]
roster.sort('stage1_rank')

# sum the number of transiting and gaseous planets
unq_tids, i_inverse = np.unique(roster['tic_id'], return_inverse=True)
roster.add_index('tic_id')
roster['tran_flag'] = roster['tran_flag'].filled(0)
roster['flag_gaseous'] = roster['flag_gaseous'].filled(0)
n_tnst = [np.sum(roster.loc[tid]['tran_flag']) for tid in unq_tids]
n_tnst_gas = [np.sum(roster.loc[tid]['flag_gaseous']) for tid in unq_tids]
roster['n_tnst'] = np.array(n_tnst)[i_inverse]
roster['n_tnst_gas'] = np.array(n_tnst_gas)[i_inverse]

# pick the highest transit SNR of planets in system
with catutils.catch_QTable_unit_warnings():
    roster_picked_transit = roster.copy()
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_nominal', np.max, 'transit_snr_nominal')
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_optimistic', np.max, 'transit_snr_optimistic')

# slim down to just the hosts
catutils.scrub_indices(roster_picked_transit)
with catutils.catch_QTable_unit_warnings():
    roster_hosts = catutils.planets2hosts(roster_picked_transit)
roster_picked_transit.add_index('tic_id')
roster_hosts.add_index('tic_id')


#%% tabulate basic info on targets

target_info = roster_hosts[['tic_id']].copy()
target_info.rename_column('tic_id', 'TIC ID')
target_info.add_index('TIC ID')

target_info['Global\nRank'] = roster_hosts['stage1_rank']
target_info['Target'] = dbutils.target_names_tic2stela(roster_hosts['tic_id'])
target_info['No.\nPlanets'] = roster_hosts['sy_pnum']
target_info['No. Tnst\nPlanets'] = roster_hosts['n_tnst']
target_info['No. Tnst\nGaseous'] = roster_hosts['n_tnst_gas']
target_info["Nominal Lya\nTransit SNR"] = roster_hosts['transit_snr_nominal']
target_info["Optimistic Lya\nTransit SNR"] = roster_hosts['transit_snr_optimistic']

# set Status to selected, backup, or archival
n = len(roster_hosts)
ext_lya_mask = roster_hosts['external_lya'].filled(False) == True
selected_mask = roster_hosts['stage1'].filled(False)
backup_mask = roster_hosts['stage1_backup'].filled(False)
target_info['Status'] = table.MaskedColumn(length=n, dtype='object', mask=True)
target_info['Status'][selected_mask] = 'target'
target_info['Status'][backup_mask] = 'backup'
target_info['Status'][target_info['Status'].mask & ext_lya_mask] = 'external data'
target_info['External\nLya'] = roster_hosts['external_lya']
target_info['External\nFUV'] = roster_hosts['external_fuv']

# modes used
def infer_modes(prioritized_countcol_label_map):
    col = table.MaskedColumn(length=n, mask=True, dtype='object')
    for key,lbl in prioritized_countcol_label_map.items():
        add_lbl = (roster_hosts[key].filled(0) > 0) & col.mask
        col[add_lbl] = lbl
    return col
cols_lbls_ext_lya = dict(
    n_stis_g140m_lya_obs='STIS-G140M',
    n_stis_e140m_lya_obs='STIS-E140M',
    n_cos_g130m_lya_obs='COS-G130M',
)
target_info['External\nLya Mode'] = infer_modes(cols_lbls_ext_lya)
cols_lbls_ext_fuv = dict(
    n_stis_g140l_obs='STIS-G140L',
    n_stis_e140m_obs='STIS-E140M',
    n_cos_g140l_obs='COS-G140L',
    n_cos_g130m_obs='COS-G130M',
)
target_info['External\nFUV Mode'] = infer_modes(cols_lbls_ext_fuv)
cols_lbls_lya = dict(
    stage1_g140m='STIS-G140M',
    stage1_e140m='STIS-E140M',
)
target_info['Lya Mode'] = infer_modes(cols_lbls_lya)
target_info['Lya Mode'].mask[roster_hosts['external_lya'].filled(False)] = True
cols_lbls_fuv = dict(
    stage1_g140m='STIS-G140L',
    stage1_e140m='STIS-E140M',
)
target_info['FUV Mode'] = infer_modes(cols_lbls_fuv)
target_info['FUV Mode'].mask[roster_hosts['external_fuv'].filled(False)] = True
e140m_for_lya = np.char.count(target_info['Lya Mode'].astype(str), 'E140M') > 0
e140m_for_lya_mask = roster_hosts['external_lya'].filled(False)
target_info["E140M Used\nfor Lya?"] = table.MaskedColumn(e140m_for_lya, mask=e140m_for_lya_mask, dtype=bool)

# tabulate labels (in case new targets added)
labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')
catutils.set_index(labeltbl, 'tic_id')
for col in ('base', 'pair'):
    allcols = [name for name in labeltbl.colnames if col in name]
    joined_label_col = []
    for i, row in enumerate(target_info):
        try:
            j = labeltbl.loc_indices[row['TIC ID']]
            labels = [labeltbl[name][j] for name in allcols if not np.ma.is_masked(labeltbl[name][j])]
            joined_labels = ','.join(labels)
            joined_label_col.append(joined_labels)
        except KeyError:
            joined_label_col.append('')
    progname = 'Lya Visit\nLabels' if col == 'base' else 'FUV Visit\nLabels'
    target_info[progname] = joined_label_col


#%% tabulate measured lya fluxes and updated transit SNRs, save diagnostic plots

measured = []
msmt_error = []
lya_factor_rcds = []
nH_rcds = []
updated_transit_snr_rcds = [] # TODO add this calculation. goals is to compare with archival transits,
# so need to get archival transit lya lines first
prediction_pctls = [16, 50, 84]
predicted_flux_records = {pctl:[] for pctl in prediction_pctls}
def add_empty_lya_row():
    measured.append(np.ma.masked)
    msmt_error.append(np.ma.masked)
    lya_factor_rcds.append(np.ma.masked)
    nH_rcds.append(np.ma.masked)
    updated_transit_snr_rcds.append(np.ma.masked)
    for pctl in prediction_pctls:
        predicted_flux_records[pctl].append(np.ma.masked)

fit_window_vcty = np.array([-400, 400])
compute_window = fit_window_vcty + [-100, 100] # need to encompass full LSF and avoid edge effects
error_floor_window = [1500, 5000]
insts = ['hst-stis-g140m', 'hst-stis-e140m']
vstd = (lya.wgrid_std/1215.67/u.AA - 1)*const.c.to('km s-1')
flux_units = u.Unit('erg s-1 cm-2')

for tic_id in tics:
    target_filename = preloads.stela_names.loc['tic_id', tic_id]['hostname_file']
    data_dir = paths.target_hst_data(target_filename)
    sys_planets = planets.loc[tic_id]

    # find the appropriate file
    files = dbutils.find_coadd_or_x1ds(target_filename, instruments=insts, directory=data_dir)
    if len(files) == 0:
        print(f'No x1d file found for {target_filename}. Moving on.')
        add_empty_lya_row()
        continue
    if len(files) > 1:
        warnings.warn(f'Multiple x1d files found for {target_filename}, but no coadds.')
    specfile = files[0]
    if 'stis-e140m' in specfile.name:
        assert 'coadd' in specfile.name
    x1dfiles = dbutils.find_data_files('x1d', instruments=insts, directory=data_dir)

    file_info = dbutils.parse_filename(specfile)
    config = file_info['config']
    _, _, grating = config.split('-')

    x1d_hdu = fits.open(specfile, ext=1)
    spec = x1d_hdu[1].data[0]
    wearth_data, flux_data, error_data = spec['wavelength'], spec['flux'], spec['error']

    # shift to velocity frame
    rv = roster_hosts.loc[tic_id]['st_radv'].filled(0)
    vsys_data = (spec['wavelength'] / 1215.67 - 1) * const.c - rv
    vsys_data = vsys_data.to_value('km s-1')

    # get information on airglow location and aperture from contributing x1ds
    v_helios = []
    apertures = []
    for xf in x1dfiles:
        component_hdu = fits.open(xf)
        v_helios.append(component_hdu[1].header['v_helio'])
        apertures.append(component_hdu[0].header['aperture'])
    v_helio_range = -np.array([max(v_helios), min(v_helios)])
    v_helio_range_sys = v_helio_range - rv.to_value('km s-1')
    unique_apertures, aperture_counts = np.unique(apertures, return_counts=True)
    main_aperture = unique_apertures[np.argmax(aperture_counts)]

    # infer range where airglow is significant
    ag_width_scale_factor_dic = dict(g140m=0.26, e140m=0.07) # 0.07 value determined by eye from looking at E140M spec
    ag_width_scale_factor = ag_width_scale_factor_dic[grating]
    ap_width = float(main_aperture.split('X')[1])
    dw = np.interp(1215.67, spec['wavelength'][:-1], np.diff(spec['wavelength']))
    dv = np.interp(1215.67, spec['wavelength'][:-1], np.diff(vsys_data))
    vbuffer = (ag_width_scale_factor * dv / dw * ap_width / 0.2
               + 7.5) # hst orbital velocity
    vbuffer_safe = 125
    airglow_contam_range_sys = v_helio_range_sys + np.array((-vbuffer, vbuffer))
    airglow_contam_range_sys_safe = v_helio_range_sys + np.array((-vbuffer_safe, vbuffer_safe))

    # plot spectrum
    fig = plt.figure()
    plt.title(specfile.name, fontsize='small')
    plt.step(vsys_data, flux_data, color='C0', where='mid', label='signal', alpha=0.5)
    plt.step(vsys_data, error_data, color='C0', ls=':', where='mid', label='uncty', alpha=0.5)
    plt.xlim(fit_window_vcty * 1.5)

    # plot airglow
    if grating == 'g140m' and 'coadd' not in specfile.name:
        bk_flux = hutils.get_background_flux(spec)
        plt.step(vsys_data, bk_flux, color='C2', where='mid', label='bkgnd', alpha=0.5)
    else:
        spn = plt.axvspan(*airglow_contam_range_sys, color='0.5', alpha=0.2)

    # function for lya predictions
    compute_mask = utils.is_in_range(vsys_data, *compute_window)
    dw = wearth_data[1] - wearth_data[0]
    wave_compute_window = wearth_data[compute_mask][[0,-1]] + [-dw/2, dw/2]
    optic = stis.Spectrograph.from_x1d(x1d_hdu, wave_window=wave_compute_window,
                                       grating=grating, aperture=main_aperture.lower())
    itarget = roster_hosts.loc_indices[tic_id]
    broaden_and_bin = optic.fast_observe_function(lya.wgrid_std.to_value('AA'))
    def predict_lya(lya_factor, n_H,):
        profile, = lya.lya_at_earth_auto(
            roster_hosts[[itarget]],
            n_H*u.cm**-3,
            lya_factor=lya_factor,
            default_rv='ism')
        flux_lsf, _ = broaden_and_bin(profile.to_value('erg s-1 cm-2 AA-1'), 1000.)
        return profile, flux_lsf
    def integrate_profile(profile):
        return np.trapz(profile, lya.wgrid_std)

    # plot 16th, 50th, 84th profiles, store fluxes
    labelmap = {pctl:lbl for pctl, lbl in zip(prediction_pctls, range(-1,2))}
    def predict_lya_pctl(pctl):
        lya_factor = lya.lya_factor_percentile(pctl)
        n_H = ism.ism_n_H_percentile(100 - pctl)
        return predict_lya(lya_factor, n_H.to_value('cm-3'))
    def predict_plot_and_integrate(pctl):
        profile, flux_lsf = predict_lya_pctl(pctl)
        plt.step(vsys_data[compute_mask], flux_lsf, color='0.8', label=f'{labelmap[pctl]:+}σ pdctn')
        F = integrate_profile(profile)
        return F
    Flya_predicted = {pctl:predict_plot_and_integrate(pctl) for pctl in prediction_pctls}
    for pctl in prediction_pctls:
        predicted_flux_records[pctl].append(Flya_predicted[pctl].to_value(flux_units))

    # get ranges to use in fit
    if grating == 'g140m':
        snr = flux_data / error_data
        n_smooth = 3
        smoothed_snr = np.convolve(snr, np.ones(n_smooth)/n_smooth, mode='same')

        # find contiguous range near airglow where smoothed snr is v low
        suspect_pixels = smoothed_snr < 2
        i_start = np.searchsorted(vsys_data, np.mean(v_helio_range_sys))
        jump = 1
        while not suspect_pixels[i_start]:
            i_start += jump
            jump = -jump if jump > 0 else -jump - 1
        contam_slice = utils.contiguous_true_range(suspect_pixels, i_start)

        # dont let contaminated range extend beyond plausible range of airglow
        contam_range = vsys_data[list(contam_slice)]
        contam_range = np.clip(contam_range, *airglow_contam_range_sys_safe)
    elif grating == 'e140m':
        contam_range = airglow_contam_range_sys_safe
    else:
        raise NotImplementedError
    uncontaminated = ~utils.is_in_range(vsys_data, *contam_range)
    in_window = utils.is_in_range(vsys_data, *fit_window_vcty)
    data_fit_mask = in_window & uncontaminated
    assert sum(data_fit_mask) > 4
    mod_fit_mask = np.isin(optic.wavegrid, wearth_data[data_fit_mask])
    flux_fit_plot = flux_data.copy()
    flux_fit_plot[~data_fit_mask] = np.nan
    plt.step(vsys_data, flux_fit_plot, color='C0', where='mid', lw=2, label='used in fit')

    flux_fit = flux_data[data_fit_mask]

    def get_residuals(logparams):
        params = np.power(10, logparams)
        _, ymod = predict_lya(*params)
        return (flux_fit - ymod[mod_fit_mask]) / flux_fit.max()

    # find best fitting lya prediction
    result = least_squares(get_residuals, (0, -1.5), bounds=((-2, -3), (2, 0)))
    assert result.success
    lyafac, nH = 10**result.x
    profile_best, prof_lsf_best = predict_lya(lyafac, nH)
    Flya = integrate_profile(profile_best)

    # store results
    measured.append(Flya.to_value(flux_units))
    lya_factor_rcds.append(lyafac)
    nH_rcds.append(nH)

    # plot best fit
    plt.step(vsys_data[compute_mask], prof_lsf_best, color='0.4', label='prelim fit')

    # estimate data errors (thus avoiding known pipeline bugs)
    errors_est = np.std(flux_fit - prof_lsf_best[mod_fit_mask])
    def get_chi2(ymod):
        residuals = (flux_fit - ymod[mod_fit_mask]) / errors_est
        return np.sum(residuals**2)

    # scale up and down to ∆chi2 of ± 1 as a simple means of sampling uncty range of different fluxes
    chi2_best = get_chi2(prof_lsf_best)
    def chi2_diff(scalefac):
        prof_lsf = prof_lsf_best * scalefac
        chi2 = get_chi2(prof_lsf)
        return chi2 - chi2_best
    def get_1sigma(bracket):
        result = root_scalar(lambda x: chi2_diff(x) - 1, bracket=bracket)
        assert result.converged
        return result.root
    peak_sigma_above_zero = max(prof_lsf_best) / errors_est
    lofac = get_1sigma((1 - 10/peak_sigma_above_zero, 1))
    hifac = get_1sigma((1, 1 + 10/peak_sigma_above_zero))
    sigfac = (hifac - lofac)/2
    Fsnr = 1/sigfac
    Flya_err = sigfac*Flya
    msmt_error.append(Flya_err.to_value(flux_units))

    # print some diagnostics on the plot
    disposition = 'PASS' if Fsnr >= 3 else 'DROP'
    O_C = (Flya - Flya_predicted[50])
    pred_uncty_hi = Flya_predicted[84] - Flya_predicted[50]
    pred_uncty_lo = Flya_predicted[50] - Flya_predicted[16]
    pred_uncty = pred_uncty_lo if O_C < 0 else pred_uncty_hi
    O_C_s = O_C / pred_uncty
    flux_lbl = (f'{disposition}\n'
                f'\n'
                f'measured flux:\n'
                f'  {Flya.to_value(flux_units):.2e} ± {Flya_err.to_value(flux_units):.1e} ({Fsnr:.1f}σ)\n'
                f'predicted flux:\n'
                f'  {Flya_predicted[50].to_value(flux_units):.1e} +{pred_uncty_hi.to_value(flux_units):.1e} / -{pred_uncty_lo.to_value(flux_units):.1e}\n' 
                f'(measured - predicted)/sigma:\n'
                f'  {O_C_s:.1f}\n'
                )
    leg = plt.legend(loc='upper right')
    plt.annotate(flux_lbl, xy=(0.02, 0.98), xycoords='axes fraction', va='top',
                 bbox=dict(fc='w', ls='none', pad=1, alpha=0.7))

    ylo = -2 * np.std(flux_fit - prof_lsf_best[mod_fit_mask])
    pred_maxs = np.array([np.max(predict_lya_pctl(pctl)[1]) for pctl in prediction_pctls])
    fit_max = np.max(flux_fit)
    if np.any(pred_maxs > fit_max):
        plot_max = min(pred_maxs[pred_maxs > fit_max])
    else:
        plot_max = fit_max
    yhi = 1.2 * plot_max
    plt.ylim(ylo, yhi)
    plt.xlabel('Velocity in System Frame (km s-1)')
    plt.ylabel('Flux Density (cgs)')

    if saveplots:
        pngfile = str(specfile).replace('.fits', '.plot.png')
        fig.savefig(pngfile, dpi=300)

        htmlfile = str(specfile).replace('.fits', '.plot.html')
        utils.save_standard_mpld3(fig, htmlfile)

    if have_a_look:
        _ = utils.click_coords() # script will pause until user clicks off the plot

    plt.close()

#%% add Lya flux info to target_info table

iloc = target_info.loc_indices['TIC ID', tics]
n = len(target_info)
def addfluxcol(name, values):
    target_info[name] = table.MaskedColumn(length=n, mask=True, dtype=float, unit=flux_units)
    target_info[name][iloc] = values

for pctl in prediction_pctls:
    addfluxcol(f'Predicted\nLya Flux {pctl}%', predicted_flux_records[pctl]*flux_units)
addfluxcol("Integrated\nLya Flux", measured*flux_units)
addfluxcol("Integrated\nLya Flux Error", msmt_error*flux_units)


#%% join with the existing table

prog_update = table.join(prog_update, target_info, keys='TIC ID', join_type='outer')


#%% setup visit info cols

cols_to_update = 'Lya Visit\nin Phase II, Planned\nLya Obs, Last Lya\nObs, FUV Visit\nin Phase II, Planned\nFUV Obs, Last FUV\nObs'.split(', ')
for name in cols_to_update:
    name1 = name + '_1'
    name2 = name + '_2'
    prog_update.rename_column(name, name1)
    prog_update[name2] = ''
    prog_update[name2] = prog_update[name2].astype('object')

for stage in ('Lya', 'FUV'):
    prog_update[f'{stage} Visit\nin Phase II_2'] = False


#%% merge observation detailts from STScI visit status

status = preloads.visit_status.copy()
status.add_index('visit')
used = np.zeros(len(status), bool) # for tracking that all visits are accounted for in progress table

# go row by row rather than doing a cross match because there may be multiple Lya or FUV visits
# for the same target due to repeats
for i, row in enumerate(prog_update):
    for stage in ('Lya', 'FUV'):
        visit_labels = row[f'{stage} Visit\nLabels_2']
        if np.ma.is_masked(visit_labels):
            visit_labels = [None]
        else:
            visit_labels = visit_labels.split(',')

        if not np.any(np.isin(visit_labels, status['visit'])):
            continue
        prog_update[f'{stage} Visit\nin Phase II_2'][i] = True

        j = status.loc_indices[visit_labels]
        j = np.atleast_1d(j)
        n_visits = len(j)
        # note that not all visit labels in the progress update table will be in HST's status report, such as
        # FUV visits we removed from the plan or backup targets
        used[j] = True
        status_rows = status[j]

        # update dates
        date_info_sets = (('obsdate', f'Last {stage}\nObs_2', max, datetime(1, 1, 1)),
                          ('next', f'Planned\n{stage} Obs_2', min, datetime(3000, 1, 1)))
        for status_col, prog_col, slctn_fn, date_fill in date_info_sets:
            if np.all(status_rows[status_col].mask):
                continue
            if n_visits > 1:
                slctd_date = slctn_fn(status_rows[status_col].filled(date_fill))
            else:
                slctd_date, = status_rows[status_col]
            # Format the date as "Mon DD, YYYY" (e.g., "Mar 31, 2025")
            prog_update[prog_col][i] = slctd_date.strftime("%b %d, %Y")

if not np.all(used):
    msg = ("The following visits didn't get added to the prog_update table: "
           f"{', '.join(status['visit'][~used])}"
           "\nYou probably need to add some repeat visits by hand to the target_visit_labels.ecsv table,"
           "or it might be that these belong to a target removed from the roster in the last build but"
           "who are still in the visit status xml this script is using (from preloads.status).")
    warnings.warn(msg)


#%% update external data status from verified obs table

vfd_path, = paths.checked.glob('*verified*')
vfd = catutils.load_and_mask_ecsv(vfd_path)
vfd = vfd[['tic_id', 'lya', 'fuv']]
vfd['lya'][vfd['lya'] == 'none'] = ''
vfd['fuv'][vfd['fuv'] == 'none'] = ''
vfd.rename_columns(('lya', 'fuv'), ("External\nLya Good?", "External\nFUV Good?"))
prog_update = table.join(prog_update, vfd, keys_left='TIC ID', keys_right='tic_id', join_type='left')


#%% sort progress table columns for easy comparison
sorted_names = []
for name in prog_update.colnames:
    if '_1' in name:
        name2 = name.replace('_1', '_2')
        namediff = name.replace('_1','_diff')
        diff = prog_update[name] != prog_update[name2]
        prog_update[namediff] = diff
        sorted_names.extend((name, name2, namediff))
    elif ('_2' in name) or ('_diff' in name):
        pass
    else:
        sorted_names.append(name)
prog_update = prog_update[sorted_names]


#%% save a csv of progress table to inspect and copy columns back into official table
pass

# sort to match the original table
catutils.set_index(prog_update, 'TIC ID')
idx, _, _ = catutils.loc_indices_and_unmatched(prog_update, preloads.progress_table['TIC ID'])
mask_new = ~np.in1d(prog_update['TIC ID'], preloads.progress_table['TIC ID'])
i_new, = np.nonzero(mask_new)
isort = idx + i_new.tolist()
prog_update = prog_update[isort]

today = datetime.today().strftime("%Y-%m-%d")
prog_update.write(paths.status_output / f'progress_table_updates_{today}.csv', overwrite=True)


#%% --- PROGRESS PLOTS ---

for stage in ['Lya', 'FUV']:
    mask = prog_update[f'{stage} Visit\nin Phase II_2']
    mask = mask.astype(bool)
    datelists = []
    for datekey in ['Planned\n{} Obs_2', 'Last {}\nObs_2']:
        date_strings = prog_update[datekey.format(stage)][mask]
        dates = []
        for datestr in date_strings:
            if datestr in ['', 'SNAP']:
                continue
            try:
                date = time.Time(datestr)
            except ValueError:
                _ = datetime.strptime(datestr, "%b %d, %Y")
                date = time.Time(_)
            dates.append(date)
        if dates:
            dates = time.Time(sorted(dates))
        datelists.append(dates)
    plandates, obsdates = datelists

    plt.figure()
    nobs = len(obsdates)
    nplan = len(plandates)
    ivec = np.arange(nobs + nplan) + 1
    if obsdates:
        plt.plot(obsdates.decimalyear, ivec[:nobs], 'k-', lw=2)
    if plandates:
        plt.plot(plandates.decimalyear, ivec[nobs:nobs+nplan], '--', lw=2, color='0.5')
    plt.axhline(nobs + nplan, color='C2', lw=2)
    plt.xlim(2025.0, 2027.0)
    plt.ylim(-5, 135)
    plt.xlabel('Date')
    plt.ylabel(f'{stage.upper()} Observations Executed')
    plt.tight_layout()
    plt.savefig(paths.status_output / f'{stage} progress chart.pdf')
    plt.savefig(paths.status_output / f'{stage} progress chart.png', dpi=300)


#%% ISR table cycle 32 status columns

from target_selection_tools import reference_tables as ref

if hasattr(prog_update['Target_2'], 'mask'):
    prog_valid = prog_update[~prog_update['Target_2'].mask].copy()
else:
    prog_valid = prog_update.copy()

combo = table.join(ref.mdwarf_isr, prog_valid, join_type='left', keys_left='Target', keys_right='Target_2')
combo.add_index('Target')
combo = combo.loc['Target', ref.mdwarf_isr['Target'].tolist()]
for stage in ('Lya', 'FUV'):
    in_phase2 = combo[f"{stage} Visit\nin Phase II_2"].filled(False).astype(bool)
    observed = combo[f"Last {stage}\nObs_2"].filled('') != ''
    if stage == 'FUV':
        e140m_for_lya = combo["E140M Used\nfor Lya?_2"].filled(False)
        lya_observed = combo[f"Last Lya\nObs_2"].filled('') != ''
        observed[e140m_for_lya & lya_observed] = True
    obs_status = table.MaskedColumn(length=len(combo), mask=True, dtype=object)
    obs_status[:] = 'not in plan'
    obs_status[in_phase2] = 'planned'
    obs_status[observed] = 'observed'

    print(f"{stage}:")
    for x in obs_status:
        print(x)
    print("")
    print("")