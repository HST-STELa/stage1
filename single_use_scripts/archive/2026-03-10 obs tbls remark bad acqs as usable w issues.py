import numpy as np
from astropy import table

import paths
import catalog_utilities as catutils

from processing.observation_table import ObsTable

#%% load obs tables, remove rawtags from cos sci files, save

obstbl_paths = sorted(paths.data_targets.rglob('*observation-table*.ecsv'))


#%%

ask_to_save = True
for obspath in obstbl_paths:
    obs_tbl = catutils.load_and_mask_ecsv(obspath)
    badacq = (
            ~obs_tbl['usable'].filled(True) &
            (np.char.count(obs_tbl['reason unusable'].filled('').astype(str), 'acq') > 0)
    )
    if 'usability status' not in obs_tbl.colnames:
        obs_tbl['usability status'] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')

    obs_tbl['usable'].mask[badacq] = True
    obs_tbl['reason unusable'].mask[badacq] = True
    obs_tbl['usability status'][badacq] = 'has issues'
    obs_tbl['flags'][badacq] = ['Acquisition untrustworthy.']

    good = obs_tbl['usable'].filled(False)
    obs_tbl['usability status'][good] = 'all clear'

    obs_tbl.pprint(-1,-1)
    if ask_to_save:
        save = input('Save? (enter/n/stop asking) ')
    if save == 'stop asking':
        ask_to_save = False
        save = ''
    if save == '':
        obs_tbl.write(obspath, overwrite=True)


#%% now do it again but make sure flags are a list and unusable exposures have correct usability status

ask_to_save = True
for obspath in obstbl_paths:
    obs_tbl = catutils.load_and_mask_ecsv(obspath)
    chng = False

    for i, row in enumerate(obs_tbl):
        flags = row['flags']
        if type(flags) is str:
            obs_tbl['flags'][i] = [flags]
            chng = True

    redo_status = ~obs_tbl['usable'].filled(True) & (obs_tbl['usability status'].filled('') != 'unusable')
    if any(redo_status):
        chng = True
        obs_tbl['usability status'][redo_status] = 'unusable'

    if chng:
        obs_tbl.pprint(-1, -1)
        if ask_to_save:
            save = input('Save? (enter/n/stop asking) ')
        if save == 'stop asking':
            ask_to_save = False
            save = ''
        if save == '':
            obs_tbl.write(obspath, overwrite=True)

#%%

ask_to_save = True
for obspath in obstbl_paths:
    obs_tbl = catutils.load_and_mask_ecsv(obspath)
    chng = False

    redo_status = obs_tbl['usability status'].filled('') == 0
    if any(redo_status):
        chng = True
        obs_tbl['usability status'].mask[redo_status] = True

    if chng:
        obs_tbl.pprint(-1, -1)
        if ask_to_save:
            save = input('Save? (enter/n/stop asking) ')
        if save == 'stop asking':
            ask_to_save = False
            save = ''
        if save == '':
            obs_tbl.write(obspath, overwrite=True)


#%%

ask_to_save = True
for obspath in obstbl_paths:
    obs_tbl = catutils.load_and_mask_ecsv(obspath)
    chng = False

    redo_status = [val == [] for val in obs_tbl['usability status'].filled('')]
    redo_status = np.array(redo_status, bool)
    if any(redo_status):
        chng = True
        obs_tbl['usability status'].mask[redo_status] = True

    if chng:
        obs_tbl.pprint(-1, -1)
        if ask_to_save:
            save = input('Save? (enter/n/stop asking) ')
        if save == 'stop asking':
            ask_to_save = False
            save = ''
        if save == '':
            obs_tbl.write(obspath, overwrite=True)

#%% get rid of lists in reason unusable, usability status, usable

ask_to_save = True
for obspath in obstbl_paths:
    obs_tbl = ObsTable.read(obspath)
    chng = False

    for key in ['usability status', 'reason unusable', 'usable', 'flags', 'notes']:
        col = obs_tbl[key]
        redo = [isinstance(val, np.ma.MaskedArray) for val in col]
        redo = np.array(redo, bool)
        if any(redo):
            chng = True
            if sum(redo) == len(obs_tbl):
                obs_tbl[key] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype=col.dtype)
            else:
                obs_tbl[key].mask[redo] = True

    if chng:
        obs_tbl.pprint(-1, -1)
        if ask_to_save:
            save = input('Save? (enter/n/stop asking) ')
        if save == 'stop asking':
            ask_to_save = False
            save = ''
        if save == '':
            obs_tbl.write(obspath, overwrite=True)