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
from processing import preloads
from processing import target_lists
from processing.observation_table import ObsTable


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

    plt.figure(figsize=[4,3])
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
