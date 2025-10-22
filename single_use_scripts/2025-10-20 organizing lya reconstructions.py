import os
import shutil

import paths
import database_utilities as dbutils


#%%

lya_reconstruction_folder = paths.data / 'packages' / 'inbox' / '2025-10-20 lya reconstructions'

files = list(lya_reconstruction_folder.glob('*lya_recon.csv'))

#%% dry run
dry_run = True

for file in files:
    pieces = dbutils.parse_filename(file)
    target = pieces['target']
    targfolder = paths.target_data(target)
    assert targfolder.exists()
    reconfolder = targfolder / 'reconstructions'

    lastbit = '/'.join(reconfolder.parts[-3:])
    if not reconfolder.exists():
        print(f"Directory to be created: {lastbit}")
        if not dry_run:
            os.mkdir(reconfolder)

    print(f'Moving {file.name} into {lastbit}.')
    if not dry_run:
        shutil.copy(file, reconfolder / file.name)


#%% repairing mistake where I made a reconstructions folder inside the hst directories

for file in files:
    pieces = dbutils.parse_filename(file)
    target = pieces['target']
    targfolder = paths.target_hst_data(target)
    oldreconfolder = targfolder / 'reconstructions'
    newreconfolder = paths.target_data(target) / 'reconstructions'
    oldlastbit = '/'.join(oldreconfolder.parts[-3:])
    newlastbit = '/'.join(newreconfolder.parts[-3:])
    print(f'Moving {oldlastbit} to {newlastbit}')
    shutil.move(oldreconfolder, newreconfolder)