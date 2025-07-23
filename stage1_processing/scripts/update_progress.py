import xml.etree.ElementTree as ET
from datetime import datetime
import re
import warnings

import numpy as np
from matplotlib import pyplot as plt
import mpld3
from mpld3 import plugins
from scipy.optimize import minimize

from astropy import table
from astropy import time
from astropy.io import fits
from astropy import constants as const
from astropy import units as u

import database_utilities as dbutils
import paths
import catalog_utilities as catutils
import utilities as utils

from lya_prediction_tools import lya, ism, stis
from stage1_processing import preloads
from stage1_processing import target_lists


#%% settings

saveplots = False

targets = target_lists.observed_since('2025-06-05')
obs_filters = dict(targets=targets, instruments=['hst-stis-g140m', 'hst-stis-e140m'], directory=paths.data_targets)


#%% --- PROGRESS TABLE UPDATES ---
pass

#%% copy progress table for revision

prog_update = preloads.progress_table.copy()


#%% load and filter latest target build

cat = preloads.planets.copy()
mask = (cat['stage1'].filled(False) # either selected for stage1
        | cat['stage1_backup'].filled(False) # backup for stage1
        | cat['external_lya'].filled(False)) # or archival
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
roster_picked_transit = roster.copy()
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_nominal', np.max, 'transit_snr_nominal')
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_optimistic', np.max, 'transit_snr_optimistic')

# slim down to just the hosts
catutils.scrub_indices(roster_picked_transit)
roster_hosts = catutils.planets2hosts(roster_picked_transit)
roster_picked_transit.add_index('tic_id')
roster_hosts.add_index('tic_id')
target_info = roster_hosts[['tic_id']].copy()
target_info.rename_column('tic_id', 'TIC ID')


#%% tabulate basic info on targets
pass

target_info['Global\nRank'] = roster_hosts['stage1_rank']
target_info['Target'] = roster_hosts['hostname']
target_info['No.\nPlanets'] = roster_hosts['sy_pnum']
target_info['No. Tnst\nPlanets'] = roster_hosts['n_tnst']
target_info['No. Tnst\nGaseous'] = roster_hosts['n_tnst_gas']
target_info["Nominal Lya\nTransit SNR"] = roster_hosts['transit_snr_nominal']
target_info["Optimistic Lya\nTransit SNR"] = roster_hosts['transit_snr_optimistic']

# set Status to selected, backup, or archival
n = len(roster_hosts)
lya_obs_mask = roster_hosts['lya_verified'].filled('') == 'pass'
fuv_obs_mask = roster_hosts['fuv_verified'].filled('') == 'pass'
selected_mask = roster_hosts['stage1'].filled(False)
backup_mask = roster_hosts['stage1_backup'].filled(False)
target_info['Status'] = table.MaskedColumn(length=n, dtype='object', mask=True)
target_info['Status'][lya_obs_mask & fuv_obs_mask] = '2 candidate'
target_info['Status'][lya_obs_mask & ~fuv_obs_mask] = '1b candidate'
target_info['Status'][selected_mask & ~lya_obs_mask] = '1a target'
target_info['Status'][backup_mask & ~lya_obs_mask] = '1a backup'
target_info['Status'][~selected_mask & ~backup_mask & lya_obs_mask] = 'archival lya'
target_info['External\nLya'] = roster_hosts['external_lya']
target_info['External\nFUV'] = roster_hosts['external_fuv']

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


#%% TBR: tabulate predicted lya fluxes

params = dict(default_rv="ism", show_progress=True)
wgrid = lya.wgrid_std
sets = (('nominal', 0),
        ('optimistic', 34))
lya_fluxes_earth = []
for lbl, pcntl in sets:
    n_H = ism.ism_n_H_percentile(50 - pcntl)
    lya_factor = lya.lya_factor_percentile(50 + pcntl)
    observed = lya.lya_at_earth_auto(roster_hosts, n_H, lya_factor=lya_factor, **params)
    _fluxes = np.trapz(observed, wgrid[None, :], axis=1)
    lya_fluxes_earth.append(_fluxes)
target_info['Nominal\nLya Flux'] = lya_fluxes_earth[0]
target_info['Optimistic\nLya Flux'] = lya_fluxes_earth[1]


#%% tabulate measured lya fluxes, save diagnostic plots

measured = []
msmt_error = []
predicted_nom = []
predicted_opt = []
predicted_nom_same_range = []
predicted_opt_same_range = []

insts=['hst-stis-g140m', 'hst-stis-e140m']
vstd = (lya.wgrid_std/1215.67/u.AA - 1)*const.c.to('km s-1')

for tic_id in target_info['TIC ID']:
    target_filename = preloads.stela_names.loc['tic_id', tic_id]['hostname_file']
    data_dir = paths.target_hst_data(target_filename)
    if target_filename not in targets:
        continue

    # find the appropriate file
    files = dbutils.find_coadd_or_x1ds(target_filename, instruments=insts, directory=data_dir)
    if len(files) == 0:
        print(f'No x1d file found for {target_filename}. Moving on.')
    if len(files) > 1:
        warnings.warn(f'Multiple x1d files found for {target_filename}, but no coadds.')
    specfile = files[0]
    if 'stis-e140m' in specfile.name:
        assert 'coadd' in specfile.name
    x1dfiles = dbutils.find_data_files('x1d', instruments=insts, directory=data_dir)

    file_info = dbutils.parse_filename(specfile)
    config = file_info['config']
    _, _, grating = config.split('-')

    h = fits.open(specfile, ext=1)
    data = h[1].data_targets
    spec = {}
    for name in data.names:
        spec[name.lower()] = data[name][0]
    w, flux, error = spec['wavelength'], spec['flux'], spec['error']

    # shift to velocity frame
    rv = roster_hosts.loc[tic_id]['st_radv'] * u.km/u.s
    v = (spec['wavelength']/1215.67 - 1) * const.c - rv
    v = v.to_value('km s-1')

    # get information on airglow location and aperture from contributing x1ds
    v_helios = []
    apertures = []
    for xf in x1dfiles:
        h = fits.open(xf)
        v_helios.append(h[1].header['v_helio'])
        apertures.append(h[0].header['aperture'])
    v_helio_range = np.array([min(v_helios), max(v_helios)])
    v_helio_range_sys = v_helio_range - rv.to_value('km s-1')
    unique_apertures, aperture_counts = np.unique(apertures, return_counts=True)
    main_aperture = unique_apertures[np.argmax(aperture_counts)]

    # infer range where airglow is significant
    ag_width_scale_factor_dic = dict(g140m=0.26, e140m=0.07)
    ag_width_scale_factor = ag_width_scale_factor_dic[grating]
    ap_width = float(main_aperture.split('X')[1])
    dw = np.interp(1215.67, spec['wavelength'][:-1], np.diff(spec['wavelength']))
    dv = np.interp(1215.67, spec['wavelength'][:-1], np.diff(v))
    vbuffer = ag_width_scale_factor * dv / dw * ap_width / 0.2  # 0.07 value determined by eye from looking at E140M spec
    airglow_contam_range = v_helio_range_sys + np.array((-vbuffer, vbuffer))

    # plot spectrum
    fig = plt.figure()
    plt.title(specfile.name)
    plt.step(v, spec['flux'], color='C0', where='mid')
    plt.xlim(-500, 500)
    ylim = plt.ylim()

    # get ranges to use in fit
    if grating == 'g140m':
        above_bk = flux > bk_flux
        snr = flux / error
        neighbors_above_bk = (np.append(above_bk[1:], False)
                                   & np.insert(above_bk[:-1], 0, False))
        uncontaminated = ((bk_flux < 1e-15) | (above_bk & neighbors_above_bk) | (snr > 2))
    elif grating == 'e140m':
        uncontaminated = (v < airglow_contam_range[0]) | (v > airglow_contam_range[1])
    else:
        raise NotImplementedError
    fit_rng = (v > -400) & (v < 400) & uncontaminated
    assert sum(fit_rng) > 4

    def data_loglike(ymod):
        terms = -(ymod - flux)**2/error**2
        return np.sum(terms[fit_rng])

    # function for lya predictions
    optic = getattr(stis, grating)
    itarget = roster_hosts.loc_indices[tic_id]
    broaden_and_bin = optic.fast_observe_function(w, lya.wgrid_std.value, main_aperture)
    def predict_lya(lya_factor, n_H):
        profile, = lya.lya_at_earth_auto(
            roster_hosts[[itarget]],
            n_H*u.cm**-3,
            lya_factor=lya_factor,
            default_rv='ism')
        flux_lsf = broaden_and_bin(profile)
        return profile, flux_lsf

    # plot 16th, 50th, 84th profiles
    #todo

    # find best fitting lya prediction
    def neg_loglike(params):
        _, ymod = predict_lya(*params)
        loglike = data_loglike(ymod.value)
        return - loglike
    min_lya, max_lya = lya.lya_factor_percentile(1), lya.lya_factor_percentile(99)
    min_nH, max_nH = ism.ism_n_H_percentile(0).value, ism.ism_n_H_percentile(100).value
    result = minimize(neg_loglike, (1.0, 0.03),
                      bounds=((min_lya, max_lya),
                              (min_nH, max_nH)))
    profile_best, prof_lsf_best = predict_lya(*result.x)

    # use scale fac and delta chi2 to get a sigma
    # scale up and down as a simple means of sampling likelihood of different fluxes
    scale_facs = np.arange(0.9, 1.1, 0.0001)
    Fs, loglikes = [], []
    for fac in scale_facs:
        profile = profile_best * fac
        F = np.trapz(profile, lya.wgrid_std)
        Fs.append(F.value)

        prof_lsf = prof_lsf_best * fac
        loglike = data_loglike(prof_lsf.value)
        loglikes.append(loglike)
    Fs = np.asarray(Fs)
    loglikes = np.asarray(loglikes)
    loglikes = loglikes - np.nanmax(loglikes)

    # predicted line likelihoods and fluxes


    pct_grid = np.arange(1, 100, 5)
    pdct_fluxes = []
    pdct_loglikes = []
    for pct in pct_grid:
        n_H = ism.ism_n_H_percentile(100 - pct)
        lya_factor = lya.lya_factor_percentile(pct)


        if pct in [15, 50, 85]:
            plt.plot(vstd - rv, profile, color='0.5', lw=1, ls=':')
            plt.step(v, flux_lsf, color='0.5', lw=1, where='mid')

        F = np.trapz(profile, lya.wgrid_std)
        pdct_fluxes.append(F)
        loglike = data_loglike(flux_lsf)
        pdct_loglikes.append(loglike)
    pdct_fluxes = np.asarray(pdct_fluxes)
    pdct_loglikes = np.asarray(pdct_loglikes)
    pdct_loglikes = pdct_loglikes - np.nanmax(pdct_loglikes)

    # plot airglow
    if 'extrsize' in spec:
        size_scale = spec['extrsize'] / (spec['bk1size'] + spec['bk2size'])
        flux_factor = spec['flux'] / spec['net']
        z = spec['net'] == 0
        if np.any(z):
            raise NotImplementedError
        bk_flux = spec['background'] * flux_factor * size_scale
        plt.step(v, bk_flux, color='C2', where='mid')
    else:
        #fixme get v_helio from all xfs and plot, do this earlier so can use for e140m integral too? except about to change up integration scheme

    C = pdct_same_range[1]
    sigma_hi = max(pdct_same_range) - pdct_same_range[1]
    sigma_lo = pdct_same_range[1] - min(pdct_same_range)
    O_C_s = (O - C) / sigma_hi
    disposition = 'PASS' if snr >= 3 else 'DROP'

    predicted_nom.append(pdct_full_range[1])
    predicted_opt.append(max(pdct_full_range))
    predicted_nom_same_range.append(C)
    predicted_opt_same_range.append(max(pdct_same_range))
    measured.append(O)
    msmt_error.append(E)

    flux_lbl = (f'{disposition}\n'
                f'\n'
                f'measured flux:\n'
                f'  {O:.2e} ± {E:.1e} ({snr:.1f}σ)\n'
                f'predicted flux:\n'
                f'  {C:.1e} +{sigma_hi:.1e} / -{sigma_lo:.1e}\n' 
                f'(O-C)/sigma:\n'
                f'  {O_C_s:.1f}\n'
                )
    leg = plt.legend(('signal', 'background', '-1,0,+1 σ predictions'), loc='upper right')
    plt.annotate(flux_lbl, xy=(0.02, 0.98), xycoords='axes fraction', va='top')

    plt.ylim(ylim)
    plt.xlabel('Velocity in System Frame (km s-1)')
    plt.ylabel('Flux Density (cgs)')

    if saveplots:
        pngfile = str(specfile).replace('.fits', '.plot.png')
        fig.savefig(pngfile, dpi=300)

        htmlfile = str(specfile).replace('.fits', '.plot.html')
        utils.save_standard_mpld3(fig, htmlfile)

    # plt.close()

# target_info['Nominal\nLya Flux'] = predicted_nom
# target_info['Optimistic\nLya Flux'] = predicted_opt
# target_info["Nom Lya\nSame Range"] = predicted_nom_same_range
# target_info["Opt Lya\nSame Range"] = predicted_opt_same_range
# target_info["Integrated\nLya Flux"] = measured
# target_info["Integrated\nLya Flux Error"] = msmt_error

#%% join with the existing table

prog_update = table.join(prog_update, target_info, keys='TIC ID', join_type='outer')


#%% setup for plan date updates
pass

# parse the xml visit status export from STScI
latest_status_path = dbutils.pathname_max(paths.status_input, 'HST-17804-visit-status*.xml')
tree = ET.parse(latest_status_path)
root = tree.getroot()

# utility to extract earliest date from PlanWindow string
def parse_planwindow_date(text):
    """ Extracts the left-most date from a planWindow string. Expected input example: "Mar 31, 2025 - Apr 1, 2025 (2025.090 - 2025.091)" This function takes the first date (e.g. "Mar 31, 2025") and returns a datetime object. """
    try: # Split on the hyphen and take the first part
        date_part = text.split(" - ")[0].strip() # Parse the date; expected format e.g. "Mar 31, 2025"
        parsed_date = datetime.strptime(date_part, "%b %d, %Y")
        return parsed_date
    except Exception as e:
        return None

# setup new cols
cols_to_update = 'Lya Visit\nin Phase II, Planned\nLya Obs, Last Lya\nObs, FUV Visit\nin Phase II, Planned\nFUV Obs, Last FUV\nObs'.split(', ')
for name in cols_to_update:
    name1 = name + '_1'
    name2 = name + '_2'
    prog_update.rename_column(name, name1)
    prog_update[name2] = ''
    prog_update[name2] = prog_update[name2].astype('object')

#%% parse xml to update plan dates

"""Iterate over each visit element in the visit status, find the appropriate row, and update dates
this can't be done with a standard table join given a row can be associated with multiple visits 
if there are redos"""
records = []
prog_update[f'Lya Visit\nin Phase II_2'] = False
prog_update[f'FUV Visit\nin Phase II_2'] = False
for visit in root.findall('visit'):
    visit_label = visit.attrib.get('visit')
    if visit_label in ['BH', 'OH']: # kludge FIXME delete after HD 118 is back in
        continue
    lya_mask = np.char.count(prog_update['Lya Visit\nLabels_2'], visit_label)
    fuv_mask = np.char.count(prog_update['FUV Visit\nLabels_2'], visit_label)
    if sum(lya_mask) > 0:
        stage = 'Lya'
        i, = np.nonzero(lya_mask)
    elif sum(fuv_mask) > 0:
        stage = 'FUV'
        i, = np.nonzero(fuv_mask)
    else:
        raise ValueError('Visit label not found.')

    # mark observation as in the phase II
    prog_update[f'{stage} Visit\nin Phase II_2'][i] = True

    plancol = f'Planned\n{stage} Obs_2'
    obscol = f'Last {stage}\nObs_2'

    # Get the status text (if available)
    status_elem = visit.find('status')
    status = status_elem.text.strip() if status_elem is not None else ""

    # Get all planWindow elements (if any)
    plan_windows = visit.findall('planWindow')

    # Check if this visit is flagged as "not a candidate..."
    if status in ["Executed", "Failed"]:
        # For executed visits use the startTime as the actual observation date.
        start_time_elem = visit.find('startTime')
        if start_time_elem is not None and start_time_elem.text:
            actual_obs, = re.findall(r'\w{3} \d+, \d{4}', start_time_elem.text)
            prog_update[obscol][i] = actual_obs
    elif status != "Executed":
        # For visits that have not executed or that failed (Scheduled, Flight Ready, Implementation, etc.)
        # if planWindow elements exist, extract the earliest possible observation date.
        dates = []
        for pw in plan_windows:
            if pw.text:
                dt = parse_planwindow_date(pw.text)
                if dt:
                    dates.append(dt)
        if dates:
            earliest_date = min(dates)
            # Format the date as "Mon DD, YYYY" (e.g., "Mar 31, 2025")
            earliest_possible = earliest_date.strftime("%b %d, %Y")
            prog_update[plancol][i] = earliest_possible

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
prog_update.write(paths.status_input / f'progress_table_updates_{today}.csv', overwrite=True)


#%% --- PROGRESS PLOTS ---

for stage in ['Lya', 'FUV']:
    mask = prog_update[f'{stage} Visit\nin Phase II_2']
    mask = mask.astype(bool)
    n = np.sum(mask)
    ivec = np.arange(n) + 1
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
    if obsdates:
        plt.plot(obsdates.decimalyear, ivec[:nobs], 'k-', lw=2)
    if plandates:
        plt.plot(plandates.decimalyear, ivec[nobs:nobs+nplan], '--', lw=2, color='0.5')
    plt.axhline(n, color='C2', lw=2)
    plt.xlim(2025.1, 2026.5)
    plt.ylim(-5, 135)
    plt.xlabel('Date')
    plt.ylabel(f'{stage.upper()} Observations Executed')
    plt.tight_layout()
    plt.savefig(paths.stage1_processing / f'{stage} progress chart.pdf')
    plt.savefig(paths.stage1_processing / f'{stage} progress chart.png', dpi=300)



