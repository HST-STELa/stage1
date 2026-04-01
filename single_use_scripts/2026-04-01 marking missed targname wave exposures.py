
import numpy as np
from astropy.io import fits

import paths
import database_utilities as dbutils

from stage1_processing import observation_table as obt
from stage1_processing import target_lists

#%%

targets = target_lists.everything_in_database()

#%%

modified_targets = []
for target in targets:
    modified = False
    obs_tbl = obt.ObsTable.read(obt.get_path(target))
    data_dir = paths.target_hst_data(target)

    for row in obs_tbl:
        path = path = dbutils.find_stela_files_from_hst_filenames(row['key science files'], data_dir)[0]

        pieces = dbutils.parse_filename(path)
        i = row.index

        file_target = fits.getval(path, 'targname')
        if file_target.lower() == 'wave':
            obs_tbl['usable'][i] = False
            obs_tbl['reason unusable'][i] = 'wave exposure'
            obs_tbl.add_flags(i, 'header targname=wave')
            obs_tbl.add_notes(i, 'Unclear why exposure with "wave" as the target shows up.')
            data = fits.getdata(path, 1)
            if np.all(data['flux'] == 0):
                obs_tbl.add_flags(i, 'Spectrum is all zeros.')
            modified = True

    if modified:
        obs_tbl.clean_duplicates_col_of_lists('flags')
        obs_tbl.clean_duplicates_col_of_lists('notes')
        obs_tbl.write(obt.get_path(target), overwrite=True)
        modified_targets.append(target)