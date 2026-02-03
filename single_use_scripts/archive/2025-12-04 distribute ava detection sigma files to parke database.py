import os
import shutil
import re

import paths
import database_utilities as dbutils

#%%

folder = paths.inbox / '2025-12-04 ava detection sigmas'
files = list(folder.glob('*sigmas.ecsv'))

for file in files:
    pl_name = file.name.split('.')[0]
    hostname = re.sub(r'-([a-z]|(0[1-9]))$', '', pl_name)
    targfolder = paths.target_data(hostname)
    assert targfolder.exists()

    dest_folder = targfolder / 'transit predictions'
    if not dest_folder.exists():
        os.mkdir(dest_folder)

    dest_path = dest_folder / file.name
    print(f"{dbutils.path_string_last_n(file, 1)} -> {dbutils.path_string_last_n(dest_path, 3)}")
    shutil.copy(file, dest_path)