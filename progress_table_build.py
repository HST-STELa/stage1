#%% imports

import numpy as np
from astropy import table

import paths
from lya_prediction_tools import lya, ism
from target_selection_tools import catalog_utilities as catutils

# region initial population of the progress table
#%% compile stage1 selections and anything with archival data we want to consider for stage 2

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
mask = (cat['stage1'].filled(False) # either selected for stage1
        | cat['stage1_backup'].filled(False) # backup for stage1
        | cat['lya_observed'].filled(False)) # or archival
roster = cat[mask]
roster.sort('stage1_rank')


#%% basic information for input to observation progress sheet

# pick highest transit SNR of planets in system
roster_picked_transit = roster.copy()
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_nominal', np.max, 'transit_snr_nominal')
catutils.pick_planet_parameters(roster_picked_transit, 'transit_snr_optimistic', np.max, 'transit_snr_optimistic')

# slim down to just the hosts
roster_hosts = catutils.planets2hosts(roster_picked_transit)
old_new_col_pairs = (('stage1_rank', 'Rank'),
                     ('hostname', 'Target'),
                     ('sy_pnum', 'No. of Planets'),
                     ('transit_snr_nominal', 'Nominal Transit SNR'),
                     ('transit_snr_optimistic', 'Optimistic Transit SNR'))
oldnames, newnames = zip(*old_new_col_pairs)
export = roster_hosts[oldnames]
export.rename_columns(oldnames, newnames)

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
export['1a External Data'] = roster_hosts['lya_observed']
export['1b External Data'] = roster_hosts['fuv_observed']

# 1a labels
labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')
labeltbl['target'] = labeltbl['target'].astype('object')
labeltbl = table.join(export, labeltbl[['target', 'base', 'pair']], keys_left='Target', keys_right='target', join_type='left')
labeltbl.sort('Rank')
export['1a Visit Label'] = labeltbl['base']
export['1b Visit Label'] = labeltbl['pair']


#%% predicted lya fluxes

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


#%% save (careful, will overwrite existing file!)

cols_in_order = ('Rank', 'Target', 'No. of Planets', 'Status', '1a Visit Label', '1a External Data', 'Nominal Lya Flux',
                 'Optimistic Lya Flux', 'Nominal Transit SNR', 'Optimistic Transit SNR', '1b Visit Label',
                 '1b External Data')
export = export[cols_in_order]
export.write('progress_reviews/progress_table_columns_from_selection_process.csv', overwrite=True)
