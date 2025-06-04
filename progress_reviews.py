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

target_table = catutils.load_and_mask_ecsv(paths.selection_outputs / 'stage1_host_catalog.ecsv')
aptnames = apt.cat2apt_names(target_table['hostname'].tolist())
target_table['aptname'] = aptnames
target_table.add_index('aptname')

# load up the latest export of the obs progress table
path_main_table = dbutils.pathname_max(paths.status_snapshots, 'Observation Progress*.xlsx')
main_table = catutils.read_excel(path_main_table)
main_table.add_index('Target')

#%% settings
saveplots = False

# targets = ['hd17156', 'k2-9', 'toi-1434', 'toi-1696', 'wolf503', 'hd207496']
# targets = ['toi-2015', 'toi-2079']
targets = 'any'

obs_filters = dict(targets=targets, after='2025-05-10', directory=data_folder)


#%% PROGRESS TABLE UPDATES
pass

#%% copy and groom key progress table columns for revision
new_table = main_table.copy()
for col in datecols:
        new_table[col] = ''
        new_table[col] = new_table[col].astype('object')


#%% load and process latest target build

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
mask = (cat['stage1'].filled(False) # either selected for stage1
        | cat['stage1_backup'].filled(False) # backup for stage1
        | cat['external_lya'].filled(False)) # or archival
roster = cat[mask]
roster.sort('stage1_rank')

# pick highest transit SNR of planets in system
roster_picked_transit = roster.copy()
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_nominal', np.max, 'transit_snr_nominal')
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_optimistic', np.max, 'transit_snr_optimistic')

# slim down to just the hosts
roster_hosts = catutils.planets2hosts(roster_picked_transit)
export = roster_hosts[['tic_id']].copy()

#%% tabulate basic info on targets
pass

# set Status to selected, backup, or archival
n = len(roster_hosts)
lya_obs_mask = roster_hosts['lya_verified'].filled('') == 'pass'
fuv_obs_mask = roster_hosts['fuv_verified'].filled('') == 'pass'
selected_mask = roster_hosts['stage1'].filled(False)
backup_mask = roster_hosts['stage1_backup'].filled(False)
export['Status'] = table.MaskedColumn(length=n, dtype='object', mask=True)
export['Status'][lya_obs_mask & fuv_obs_mask] = '2 candidate'
export['Status'][lya_obs_mask & ~fuv_obs_mask] = '1b candidate'
export['Status'][selected_mask & ~lya_obs_mask] = '1a target'
export['Status'][backup_mask & ~lya_obs_mask] = '1a backup'
export['1a External Data'] = roster_hosts['external_lya']
export['1b External Data'] = roster_hosts['external_fuv']

# 1a labels
labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')
labeltbl['target'] = labeltbl['target'].astype('object')
labeltbl = table.join(export, labeltbl[['target', 'base', 'pair']], keys_left='Target', keys_right='target', join_type='left')
labeltbl.sort('Rank')
export['1a Visit Label'] = labeltbl['base']
export['1b Visit Label'] = labeltbl['pair']

# region predicted lya fluxes
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
export['Nominal Lya Flux'] = lya_fluxes_earth[0]
export['Optimistic Lya Flux'] = lya_fluxes_earth[1]
# endregion

#%% region match into existing table


cols_in_order = ('Rank', 'Target', 'No. of Planets', 'Status', '1a Visit Label', '1a External Data', 'Nominal Lya Flux',
                 'Optimistic Lya Flux', 'Nominal Transit SNR', 'Optimistic Transit SNR', '1b Visit Label',
                 '1b External Data')
export = export[cols_in_order]

new_table =


#%% update plan dates

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

# Iterate over each visit element in the visit status, find the appropriate row, and update dates
records = []
for visit in root.findall('visit'):
    visit_label = visit.attrib.get('visit')
    lya_mask = np.char.count(main_table['lya visit'], visit_label)
    fuv_mask = np.char.count(main_table['fuv visit'], visit_label)
    if sum(lya_mask) > 0:
        stage = 'lya'
        i, = np.nonzero(lya_mask)
    elif sum(fuv_mask) > 0:
        stage = 'fuv'
        i, = np.nonzero(fuv_mask)
    else:
        raise ValueError('Visit label not found.')

    # mark observation as in the phase II
    new_table[f'{stage} planned'][i] = True

    plancol = f'{stage} plandate'
    obscol = f'{stage} obsdate'

    # Get the status text (if available)
    status_elem = visit.find('status')
    status = status_elem.text.strip() if status_elem is not None else ""

    # Get all planWindow elements (if any)
    plan_windows = visit.findall('planWindow')

    # Check if this visit is flagged as "not a candidate..."
    if status == "Executed":
        # For executed visits use the startTime as the actual observation date.
        start_time_elem = visit.find('startTime')
        if start_time_elem is not None and start_time_elem.text:
            actual_obs, = re.findall(r'\w{3} \d+, \d{4}', start_time_elem.text)
            new_table[obscol][i] = actual_obs
    else:
        # For visits that have not executed (Scheduled, Flight Ready, Implementation, etc.)
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
            new_table[plancol][i] = earliest_possible

#%% mark plan updates
sorted_cols = list(newcols[:1])
for col in newcols[1:]:
    if 'date' in col:
        strcol = []
        for item in main_table[col]:
            stritem = '' if item == '' else item.strftime("%b %d, %Y")
            strcol.append(stritem)
        strcol = table.Column(strcol)
        new_table[col + ' old'] = strcol
    else:
        new_table[col + ' old'] = main_table[col]
    new_table[col + ' updated'] = new_table[col] != new_table[col + ' old']
    sorted_cols.extend((col, col + ' old', col + ' updated'))
new_table = new_table[sorted_cols]

today = datetime.today().strftime("%Y-%m-%d")
new_table.write(paths.status_snapshots / f'visit_dates_for_copy-paste_{today}.csv', overwrite=True)


#%% make cumulative progress plots

for stage in ['lya', 'fuv']:
    mask = new_table[f'{stage} planned']
    n = np.sum(mask)
    ivec = np.arange(n) + 1
    datelists = []
    for datekey in ['obsdate', 'plandate']:
        date_strings = new_table[f'{stage} {datekey}'][mask]
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
    obsdates, plandates = datelists

    plt.figure()
    nobs = len(obsdates)
    nplan = len(plandates)
    if obsdates:
        plt.plot(obsdates.decimalyear, ivec[:nobs], 'k-', lw=2)
    plt.plot(plandates.decimalyear, ivec[nobs:nobs+nplan], '--', lw=2, color='0.5')
    plt.axhline(n, color='C2', lw=2)
    plt.xlim(2025.1, 2026.5)
    plt.ylim(-5, 135)
    plt.xlabel('Date')
    plt.ylabel(f'{stage.upper()} Observations Executed')
    plt.tight_layout()
    plt.savefig(progress_folder / f'{stage} progress chart.pdf')
    plt.savefig(progress_folder / f'{stage} progress chart.png', dpi=300)


#%% check acquisitions

rawfiles = dbutils.find_data_files('raw', instruments='hst-stis-mirvis', **obs_filters)

stages = ['coarse', 'fine', '0.2x0.2']
for file in rawfiles:
    fig, axs = plt.subplots(1, 3, figsize=[7,3])
    h = fits.open(file)
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
    y = td['extrlocy']
    ysz = td['extrsize']
    plt.plot(x, y.T, color='w', lw=0.5, alpha=0.5)
    for _y, _ysz in zip(y, ysz):
        plt.fill_between(x, _y - _ysz/2, _y + _ysz/2, color='w', lw=0, alpha=0.5)
    for ibk in (1,2):
        off, sz = td[f'bk{ibk}offst'], td[f'bk{ibk}size']
        ym = y + off[:,None]
        y1, y2 = ym - sz[:,None]/2, ym + sz[:,None]/2
        for yy1, yy2 in zip(y1, y2):
            plt.fill_between(x, yy1, yy2, color='0.5', alpha=0.5, lw=0)

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

    # predicted lines
    ylim = plt.ylim()
    itarget = target_table.loc_indices[name.upper()]
    predicted_fluxes = []
    for pct in (-34, 0, +34):
        n_H = ism.ism_n_H_percentile(50 + pct)
        lya_factor = lya.lya_factor_percentile(50 - pct)
        profile, = lya.lya_at_earth_auto(target_table[[itarget]], n_H, lya_factor=lya_factor, default_rv='ism')
        plt.plot(vstd - rv, profile, color='0.5', lw=1)
        predicted_flux = np.trapz(profile, lya.wgrid_std)
        predicted_fluxes.append(predicted_flux.to_value('erg s-1 cm-2'))

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
    C = predicted_fluxes[1]
    sigma_hi = max(predicted_fluxes) - predicted_fluxes[1]
    sigma_lo = predicted_fluxes[1] - min(predicted_fluxes)
    O_C_s = (O - C) / sigma_hi
    disposition = 'PASS' if snr >= 3 else 'DROP'

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

diagnostic_table = table.join(_temp[hostcols], main_table[obscols],
                              keys_left='hostname', keys_right='Target', join_type='right')
assert len(diagnostic_table) == len(main_table)
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