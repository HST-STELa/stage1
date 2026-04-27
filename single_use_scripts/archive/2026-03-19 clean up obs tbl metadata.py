import os

from tqdm import tqdm

import paths

from processing import target_lists
from processing.observation_table import ObsTable


#%%

targets = target_lists.everything_in_database()


#%%

no_obs_tbl = []
all_keys = set()
for target in tqdm(targets):
    hstdatadir = paths.target_data(target) / 'hst'

    try:
        obstbl = ObsTable.load_from_targname(target)
    except FileNotFoundError:
        no_obs_tbl.append(target)
        continue

    all_keys |= set(obstbl.meta.keys())

    chng = False

    last_acq = obstbl.meta.get('last acq check')
    last_chk = obstbl.meta.get('last data review')

    if last_acq and last_chk:
        last_chk = max(last_acq, last_chk)
        obstbl.meta['last data review'] = last_chk
        del obstbl.meta['last acq check']
        chng = True
    elif last_acq and last_chk is None:
        obstbl.meta['last data review'] = last_acq
        del obstbl.meta['last acq check']
        chng = True
    elif last_chk and last_acq is None:
        pass # all good
    else:
        pass # both absent, but that's okay

    if chng:
        obstbl.write(obstbl.get_path(target), overwrite=True)

odd_keys = all_keys - {'last archive query', 'last data review', 'last stis extraction', 'last acq check'}
print(f"Odd keys (if any): {odd_keys}")

if no_obs_tbl:
    print('These targets had no observation-table file:')
    for tgt in no_obs_tbl:
        print(f'\t{tgt}')
