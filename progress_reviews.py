from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpld3
from mpld3 import plugins

from astropy import table
from astropy import time
from astropy.io import fits
from astropy import constants as const
from astropy import units as u

import utilities as utils
import database_utilities as dbutils
import paths
import catalog_utilities as catutils
from target_selection_tools import apt
from lya_prediction_tools import lya, ism

#%% paths and preloads
data_folder = Path('/Users/parke/Google Drive/Research/STELa/data/uv_observations/hst-stis')

target_table = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
target_table = catutils.planets2hosts(target_table)
aptnames = apt.cat2apt_names(target_table['hostname'].tolist())
target_table['aptname'] = aptnames
target_table.add_index('aptname')

# load up the latest export of the obs progress table
path_main_table = dbutils.pathname_max(paths.status_snapshots, 'Observation Progress*.xlsx')
progress_table = catutils.read_excel(path_main_table)
progress_table.add_index('Target')

#%% settings
saveplots = False

# targets = ['hd17156', 'k2-9', 'toi-1434', 'toi-1696', 'wolf503', 'hd207496']
# targets = ['toi-2015', 'toi-2079']
targets = 'any'

obs_filters = dict(targets=targets, after='2025-01-01', directory=data_folder)


#%% --- PROGRESS TABLE UPDATES ---
pass

#%% copy progress table for revision
prog_update = progress_table.copy()


#%% convert datetime cols to strings of just the date without time
for col in prog_update.colnames:
    valid = prog_update[col] != ''
    if not np.any(valid):
        continue
    test_value = prog_update[col][valid][0]
    if isinstance(test_value, datetime):
        prog_update[col][valid] = [x.strftime("%b %d, %Y") for x in prog_update[col][valid]]

#%% load and filter latest target build

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
mask = (cat['stage1'].filled(False) # either selected for stage1
        | cat['stage1_backup'].filled(False) # backup for stage1
        | cat['external_lya'].filled(False)) # or archival
roster = cat[mask]
roster.sort('stage1_rank')

# pick the highest transit SNR of planets in system
roster_picked_transit = roster.copy()
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_nominal', np.max, 'transit_snr_nominal')
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_optimistic', np.max, 'transit_snr_optimistic')

# slim down to just the hosts
roster_hosts = catutils.planets2hosts(roster_picked_transit)
target_info = roster_hosts[['tic_id']].copy()
target_info.rename_column('tic_id', 'TIC ID')


#%% tabulate basic info on targets
pass

target_info['Global\nRank'] = roster_hosts['stage1_rank']
target_info['Target'] = roster_hosts['hostname']
target_info['No. of\nPlanets'] = roster_hosts['sy_pnum']
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


#%% tabulate predicted lya fluxes

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


#%% join with the existing table

prog_update = table.join(prog_update, target_info, keys='TIC ID', join_type='outer')


#%% parse STScI plan dates
pass

# parse the xml visit status export from STScI
latest_status_path = dbutils.pathname_max(paths.status_snapshots, 'HST-17804-visit-status*.xml')
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

# Iterate over each visit element in the visit status, find the appropriate row, and update dates
# this can't be done with a standard table join given a row can be associated with multiple visits if there are redos
records = []
prog_update[f'Lya Visit\nin Phase II_2'] = False
prog_update[f'FUV Visit\nin Phase II_2'] = False
for visit in root.findall('visit'):
    visit_label = visit.attrib.get('visit')
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
idx, _, _ = catutils.loc_indices_and_unmatched(prog_update, progress_table['TIC ID'])
mask_new = ~np.in1d(prog_update['TIC ID'], progress_table['TIC ID'])
i_new, = np.nonzero(mask_new)
isort = idx + i_new.tolist()
prog_update = prog_update[isort]

today = datetime.today().strftime("%Y-%m-%d")
prog_update.write(paths.status_snapshots / f'progress_table_updates_{today}.csv', overwrite=True)


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
    plt.savefig(paths.progress_reviews / f'{stage} progress chart.pdf')
    plt.savefig(paths.progress_reviews / f'{stage} progress chart.png', dpi=300)


#%% --- OBSERVATION CHECKS ---
pass

#%% check acquisitions

# use tagfiles to find raw files to be sure I didn't forget to download any
tagfiles = dbutils.find_data_files('tag', instruments='hst-stis', **obs_filters)
rawfiles = []
for tf in tagfiles:
    pieces = dbutils.parse_filename(tf)
    rsrch = f'{pieces['target']}.hst-stis-mirvis.*.{pieces['id'][:6]}*_raw.fits'
    rf, = data_folder.glob(rsrch)
    assert rf.exists()
    rawfiles.append(rf)

stages = ['coarse', 'fine', '0.2x0.2']
for file in rawfiles:
    fig, axs = plt.subplots(1, 3, figsize=[7,3])
    try:
        h = fits.open(file)
    except FileNotFoundError:
        raise(FileNotFoundError('There is a tag file but '))
    for i, ax in enumerate(axs):
        data = h['sci', i+1].data
        ax.imshow(data)
        ax.set_title('')
    fig.suptitle(file.name)
    fig.supxlabel('dispersion')
    fig.supylabel('spatial')
    fig.tight_layout()


#%% plot extraction locations

fltfiles = dbutils.find_data_files('flt', instruments='hst-stis', **obs_filters)

for ff in fltfiles:
    img = fits.getdata(ff, 1)
    f1 = dbutils.modify_file_label(ff, 'x1d')
    td = fits.getdata(f1, 1)
    fig = plt.figure()
    plt.imshow(np.cbrt(img), aspect='auto')
    plt.title(ff.name)

    x = np.arange(img.shape[1]) + 0.5
    i = 36 if 'e140m' in ff.name else 0
    y = td['extrlocy'][i]
    ysz = td['extrsize'][i]
    plt.plot(x, y.T, color='w', lw=0.5, alpha=0.5)
    plt.fill_between(x, y - ysz/2, y + ysz/2, color='w', lw=0, alpha=0.5)
    for ibk in (1,2):
        off, sz = td[f'bk{ibk}offst'][i], td[f'bk{ibk}size'][i]
        ym = y + off
        y1, y2 = ym - sz/2, ym + sz/2
        plt.fill_between(x, y1, y2, color='0.5', alpha=0.5, lw=0)

    if saveplots:
        dpi = fig.get_dpi()
        fig.set_dpi(150)
        plugins.connect(fig, plugins.MousePosition(fontsize=14))
        htmlfile = str(ff).replace('_flt.fits', '_plot-extraction.html')
        mpld3.save_html(fig, htmlfile)
        fig.set_dpi(dpi)


#%% plot spectra, piggyback Lya flux (O-C)/sigma

vstd = (lya.wgrid_std/1215.67/u.AA - 1)*const.c.to('km s-1')
x1dfiles = dbutils.find_data_files('x1d', instruments='hst-stis', **obs_filters)

for xf in x1dfiles:
    h = fits.open(xf, ext=1)
    data = h[1].data
    order = 36 if 'e140m' in xf.name else 0
    spec = {}
    for name in data.names:
        spec[name.lower()] = data[name][order]

    # shift to velocity frame
    name = dbutils.parse_filename(xf)['target']
    rv = target_table.loc[name.upper()]['st_radv'] * u.km/u.s
    v = (spec['wavelength']/1215.67 - 1) * const.c - rv
    v = v.to_value('km s-1')

    fig = plt.figure()
    plt.title(xf.name)
    plt.step(v, spec['flux'], color='C0', where='mid')
    plt.xlim(-500, 500)

    # for plotting airglow
    size_scale = spec['extrsize'] / (spec['bk1size'] + spec['bk2size'])
    flux_factor = spec['flux'] / spec['net']
    z = spec['net'] == 0
    if np.any(z):
        raise NotImplementedError
    # i = np.arange(z.shape[1])
    # fi = np.interp(i[z], i[~z], ff[~z])
    # flux_factor[z] = fi
    bk_flux = spec['background'] * flux_factor * size_scale
    plt.step(v, bk_flux, color='C2', where='mid')

    # line integral and (O-C)/sigma
    w, flux, error = spec['wavelength'], spec['flux'], spec['error']
    if 'g140m' in xf.name:
        above_bk = flux > bk_flux
        neighbors_above_bk = (np.append(above_bk[1:], False)
                                   & np.insert(above_bk[:-1], 0, False))
        uncontaminated = ((bk_flux < 1e-15) | (above_bk & neighbors_above_bk))
    elif 'e140m' in xf.name:
        v_helio = h[1].header['V_HELIO']
        w0 = 1215.67
        dw = v_helio/3e5 * w0
        wa, wb = w0 - dw + np.array((-0.07, 0.07))
        uncontaminated = ~((w > wa) & (w < wb))
    else:
        raise NotImplementedError
    flux[~uncontaminated], error[~uncontaminated] = 0, 0 # zero out high airglow range
    int_rng = (v > -400) & (v < 400)
    int_mask = int_rng & uncontaminated
    if sum(int_mask) > 4:
        pltflux = flux.copy()
        pltflux[~int_mask] = np.nan
        O, E = utils.flux_integral(w[int_rng], flux[int_rng], error[int_rng])
        snr = O/E
        plt.fill_between(v, pltflux, step='mid', color='C0', alpha=0.3)
    else:
        O, E = 0, 0
        snr = 0
        int_mask[:] = 0

    # kinda kloogy but simple way to get a mask that covers the same integration intervals as the data but for the
    # model profile grid
    int_mask_mod = np.interp(lya.wgrid_std.value, w, int_mask)
    int_mask_mod[int_mask_mod < 0.5] = 0
    int_mask_mod = int_mask_mod.astype(bool)

    # predicted lines
    ylim = plt.ylim()
    itarget = target_table.loc_indices[name.upper()]
    predicted_fluxes = []
    for pct in (-34, 0, +34):
        n_H = ism.ism_n_H_percentile(50 + pct)
        lya_factor = lya.lya_factor_percentile(50 - pct)
        profile, = lya.lya_at_earth_auto(target_table[[itarget]], n_H, lya_factor=lya_factor, default_rv='ism')
        plt.plot(vstd - rv, profile, color='0.5', lw=1)
        # compute flux over same interval as it is computed from the observations
        # this neglects instrument braodening plus the line profile will be wrong, but it's close enough
        int_profile = profile.copy()
        int_profile[~int_mask_mod] = 0
        predicted_flux = np.trapz(int_profile, lya.wgrid_std)
        predicted_fluxes.append(predicted_flux.to_value('erg s-1 cm-2'))

    C = predicted_fluxes[1]
    sigma_hi = max(predicted_fluxes) - predicted_fluxes[1]
    sigma_lo = predicted_fluxes[1] - min(predicted_fluxes)
    O_C_s = (O - C) / sigma_hi
    disposition = 'PASS' if snr >= 3 else 'DROP'

    print('')
    print(f'{name}')
    print(f'\tpredicted nominal: {C:.2e}')
    print(f'\tpredicted optimistic: {max(predicted_fluxes):.2e}')
    print(f'\tmeasured: {O:.2e} ± {E:.1e}')

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
        pngfile = str(xf).replace('_x1d.fits', '_plot-spec.png')
        fig.savefig(pngfile, dpi=300)

        dpi = fig.get_dpi()
        fig.set_dpi(150)
        plugins.connect(fig, plugins.MousePosition(fontsize=14))
        htmlfile = str(xf).replace('_x1d.fits', '_plot-spec.html')
        mpld3.save_html(fig, htmlfile)
        fig.set_dpi(dpi)



#%% --- INFREQUENT OR SINGLE USE CHECKS ---
pass

#%% table of target parameters combined with observation table values

obscols = 'Rank,Target,Peak\nLya Flux,Integrated\nLya Flux,(O-C)/sigma,Pass to\nStage 1b?'.split(',')
colmap = {'Rank': 'rank',
          'Target': 'target',
          'Peak\nLya Flux': 'peak lya flux',
          'Integrated\nLya Flux': 'integrated lya flux',
          '(O-C)/sigma': 'lya (O-C)/sigma',
          'Pass to\nStage 1b?': 'pass'}
oldcols, newcols = list(zip(*colmap.items()))

bigtable = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__add-flags-scores.ecsv')
bigtable = catutils.planets2hosts(bigtable)
hostcols = 'hostname ra dec sy_dist st_radv st_teff st_mass st_rad st_rotp st_age st_agelim'.split()

_temp = bigtable.copy()
_temp['hostname'] = _temp['hostname'].astype(str)

diagnostic_table = table.join(_temp[hostcols], progress_table[obscols],
                              keys_left='hostname', keys_right='Target', join_type='right')
assert len(diagnostic_table) == len(progress_table)
diagnostic_table.rename_columns(oldcols, newcols)
diagnostic_table.remove_column('target')

numerical_cols = 'peak lya flux,integrated lya flux,lya (O-C)/sigma'.split(',')
for name in numerical_cols:
    empty = diagnostic_table[name] == ''
    diagnostic_table[name][empty] = 'nan'
    diagnostic_table[name] = table.MaskedColumn(diagnostic_table[name], mask=empty, fill_value=np.nan, dtype=float)


today = datetime.today().isoformat()[:10]
diagnostic_table.write(paths.status_snapshots / f'target properties and lya fluxes {today}.ecsv')

#%% plot coordinates of observed targets

observed = ~diagnostic_table['integrated lya flux'].mask
ot = diagnostic_table[observed]

# plot positions on sky
from astropy.coordinates import SkyCoord
import astropy.units as u

# SkyCoord object
c = SkyCoord(ra=ot['ra'], dec=ot['dec'], frame='icrs')

# Convert to radians for plotting
ra_rad = c.ra.wrap_at(180 * u.deg).radian  # wrap RA to [-180, +180]
dec_rad = c.dec.radian

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True)
ax.scatter(ra_rad, dec_rad, marker='o', color='C0')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')


#%% plot other params vs Lya fluxes

observed = ~diagnostic_table['integrated lya flux'].mask
ot = diagnostic_table[observed]

oc = ot['lya (O-C)/sigma']

names = 'sy_dist st_radv st_mass st_rotp'.split()
for name in names:
    plt.figure()
    plt.plot(ot[name], ot['lya (O-C)/sigma'], 'o')
    plt.xlabel(name)
    plt.ylabel('Lya Flux (O-C)/sigma')


plt.figure()
nolim = ot['st_agelim'].filled(0) == 0
lolim = ot['st_agelim'].filled(0) == -1
uplim = ot['st_agelim'].filled(0) == 1
plt.errorbar(ot['st_age'][nolim], ot['lya (O-C)/sigma'][nolim], fmt='oC0')
plt.errorbar(ot['st_age'][lolim], ot['lya (O-C)/sigma'][lolim], fmt='oC0', xerr=0.2, xlolims=True)
plt.errorbar(ot['st_age'][uplim], ot['lya (O-C)/sigma'][uplim], fmt='oC0', xerr=0.2, xuplims=True)
plt.xlabel('age')
plt.ylabel('Lya Flux (O-C)/sigma')