
import numpy as np
from astropy import table

import paths
from stage1_processing import observation_table as obt

#%% def acq rule

def enforce_acq_issue_status(obs_tbl):
    old_tbl = obs_tbl.copy()

    fixed = []
    for i in range(len(obs_tbl)):
        flags = obs_tbl['flags'][i]
        status = obs_tbl['usability status'][i]
        usable = obs_tbl['usable'][i]
        if np.ma.is_masked(flags):
            acq_issues = False
        else:
            acq_issues = any('acquisition untrustworthy' in flag.lower() for flag in flags if not np.ma.is_masked(flag))
        if not acq_issues:
            continue

        if not np.ma.is_masked(status) and status == 'all clear':
            obs_tbl['usability status'][i] = 'has issues'
            fixed.append(i)
        if not np.ma.is_masked(usable) and usable:
            obs_tbl['usable'].mask[i] = True
            fixed.append(i)

    fixed = list(set(fixed))
    if fixed:
        cmp_new = obs_tbl[['usable', 'usability status']].copy()
        cmp_new.rename_columns(['usable', 'usability status'], ['usable2', 'usability status2'])
        merged = table.Table((
            old_tbl['archive id'],
            old_tbl['usable'],
            cmp_new['usable2'],
            old_tbl['usability status'],
            cmp_new['usability status2'],
            old_tbl['flags'],
        ))
        merged[fixed].pprint(-1,-1)
    else:
        print("no updates")

    return obs_tbl


#%% tool to merge tastis flags into a single warning

tastis_substrs = (
    'Telemetry indicates that',
    'The fine slew',
    'The flux in the third image',
    'fluxes in the maximum checkbox',
    'Saturation of pixels',
    'flux in the confirmation image is',
    'pixels in the confirmation image were saturated',
    'ACQ/PEAK flux test failed',
    'maximum flux in the sequence occurred at one end',
)

def merge_tastis_flags(obs_tbl):
    old_tbl = obs_tbl.copy()
    for i in range(len(obs_tbl)):
        if obs_tbl['flags'].mask[i]:
            continue
        flags = obs_tbl['flags'][i]
        new_flags = []
        ta_wngns = False
        for flag in flags:
            contains_tastis_substr = any(sub in flag for sub in tastis_substrs)
            if contains_tastis_substr:
                ta_wngns = True
            else:
                new_flags.append(flag)

        if ta_wngns:
            new_flags.append('stistools.tastis logged acquisition warnings')

        obs_tbl['flags'][i] = new_flags

    old_tbl.rename_column('flags', 'old flags')
    inspctn_tbl = table.Table((
        old_tbl['archive id'],
        obs_tbl['flags'],
        old_tbl['old flags']
    ))
    inspctn_tbl.pprint(-1,-1)

    return obs_tbl


#%% run for first few targets I reviewed

viewcols = 'archive id,usable,usability status,flags'.split(',')
targets = ['55cnc', 'au-mic', 'ds-tuc-a']
for target in targets:
    obs_tbl = obt.ObsTable.load_from_targname(target)
    # obs_tbl[viewcols].pprint(-1,-1)
    new_tbl = merge_tastis_flags(obs_tbl)
    new_tbl = enforce_acq_issue_status(new_tbl)
    new_tbl.write(new_tbl.get_path(target), overwrite=True)


#%% run updated column cleaning

viewcols = 'archive id,usable,usability status,reason unusable,flags,notes'.split(',')
targets = ['55cnc', 'au-mic', 'ds-tuc-a', 'gj1132']
for target in targets:
    obs_tbl = obt.ObsTable.load_from_targname(target)
    obs_tbl[viewcols].pprint(-1,-1)
    for name in ('flags', 'notes', 'reason unusable'):
        obs_tbl.clean_nulls_col_of_lists(name)
    obs_tbl[viewcols].pprint(-1,-1)
    a = input('save?')
    if a == 'y':
        obs_tbl.write(obs_tbl.get_path(target), overwrite=True)

#%% better

viewcols = 'archive id,usable,usability status,reason unusable,flags,notes'.split(',')
targets = ['55cnc', 'au-mic', 'ds-tuc-a', 'gj1132']
keep_checking = True
for target in targets:
    obs_tbl = obt.ObsTable.load_from_targname(target)
    obs_tbl[viewcols].pprint(-1,-1)
    for name in ('flags', 'notes', 'reason unusable'):
        obs_tbl.clean_nulls_col_of_lists(name)
    i_good, = np.nonzero(~obs_tbl['reason unusable'].mask)
    for i in i_good:
        rsn = obs_tbl['reason unusable'][i]
        obs_tbl['reason unusable'][i] = rsn[0]
    obs_tbl[viewcols].pprint(-1,-1)
    if keep_checking:
        a = input('save?')
    else:
        a = 'y'

    if a in ['a', 'y']:
        obs_tbl.write(obs_tbl.get_path(target), overwrite=True)
    if a == 'a':
        keep_checking = False
    else:
        break


#%% hack to fix annoying np masked array objects appearing when tables are loaded

# after this hack, the problem should be permanently fixed by overriding the write function in my ObsTbl class

import paths
obs_tbl_paths = sorted(paths.data_targets.rglob('*observation-table.ecsv'))

for otpth in obs_tbl_paths:
    with open(otpth, 'r') as f:
        contents = f.read()
    new_contents = contents.replace("'float64[null]'", "json")
    with open(otpth, 'w') as f:
        f.write(new_contents)



#%% check that all masked array elements are truly gone

for otpth in obs_tbl_paths:
    ot = obt.ObsTable.read(otpth)
    for key in ot.colnames:
        col = ot[key]
        if col.dtype == 'O':
            for elmnt in col:
                assert type(elmnt) is not np.ma.MaskedArray


#%% move wordy flags to notes

