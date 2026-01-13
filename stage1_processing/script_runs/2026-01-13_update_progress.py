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


#%% join with the existing table

prog_update = table.join(prog_update, target_info, keys='TIC ID', join_type='outer')


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