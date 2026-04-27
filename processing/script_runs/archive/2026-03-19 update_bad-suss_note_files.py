import os

from tqdm import tqdm

import paths

from processing import target_lists
from processing.observation_table import ObsTable
import database_utilities as dbutils


#%%

targets = target_lists.everything_in_database()

maxids = 5 # max number that will be listed in a file name before "and more" is listed
statuses = ('unusable', 'has issues', 'unchecked', 'all clear')
more_message = 'and-more-see-observation-table'

helpful_message = (
    "This file is just mean to alert users at a glance that certain "
    "files might be unusable or suspect. Files are kept because time "
    "has shown that we often want to revist bad or suspect data to "
    "verify that it is indeed bad or suspect.\n"
    "\n"
    "This file was generated from the 'observation-table' that is in "
    "the same directory, which has additional information. If the "
    "observations the table lists as unusable or having issues "
    "disagree with those listed in the name of this file, "
    "the observation-table should be taken as the authority."
)


#%%

no_obs_tbl = []
for target in tqdm(targets):
    hstdatadir = paths.target_data(target) / 'hst'

    try:
        obstbl = ObsTable.load_from_targname(target)
    except FileNotFoundError:
        no_obs_tbl.append(target)

    for status in statuses:
        mask = obstbl['usability status'].filled('unchecked') == status
        ids = obstbl['archive id'][mask]

        if len(ids) == 0:
            idstr = 'none'
        elif len(ids) <= maxids:
            idstr = '-'.join(ids)
        else:
            idstr = f'{'-'.join(ids[:maxids-2])}-{more_message}'

        file = dbutils.one_glob(hstdatadir, f'_.{status}*.txt')
        newfile = hstdatadir / f'_.{status}.{idstr}.txt'
        if file:
            os.rename(file, newfile)
        newfile.write_text(helpful_message)

if no_obs_tbl:
    print('These targets had no observation-table file:')
    for tgt in no_obs_tbl:
        print(f'\t{tgt}')
