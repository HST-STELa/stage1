import os
import shutil
from pathlib import Path

import numpy as np

import catalog_utilities as catutils
import database_utilities as dbutils
import paths

#%% settings

eval_no = 1.1

temp_dir = Path(f'/Users/parke/Downloads/stela_stage2_eval_pkg{eval_no}')
if not temp_dir.exists():
    os.mkdir(temp_dir)
temp_dir_lya = Path(f'/Users/parke/Downloads/stela_stage2_eval_pkg{eval_no}/lya_data')
if not temp_dir_lya.exists():
    os.mkdir(temp_dir_lya)


#%% load up the latest export of the obs progress table
path_main_table = dbutils.pathname_max(paths.status_snapshots, 'Observation Progress*.xlsx')
progress_table = catutils.read_excel(path_main_table)


#%% assemble x1dfiles

lya_folder = 'lya_data'
eval_mask = progress_table["Stage 2 Eval\nBatch"] == 1
targets = progress_table["Target"][eval_mask]
targets_fnames = dbutils.target_names_stela2file(targets)
problem_targets = []
for name in targets_fnames:
    targdir = paths.data / name
    anyx1d = list(targdir.glob(f'*stis-?140m*x1d.fits'))[0]
    coadd = list(targdir.glob('*stis-?140m*coadd.fits'))
    if len(coadd) == 1:
        coadd, = coadd
        shutil.copy(coadd, temp_dir / lya_folder / coadd.name)
    elif len(coadd) > 1:
        raise NotImplementedError
    else:
        for suffix in ('x1dtrace', 'x1dbk1', 'x1dbk2', 'x1d'):
            file = list(targdir.glob(f'*stis-?140m*{suffix}.fits'))
            if len(file) != 1:
                if (suffix == 'x1dtrace') or ('g140m' in anyx1d.name):
                    problem_targets.append(name)
            else:
                file, = file
                shutil.copy(file, temp_dir / lya_folder / file.name)

problem_targets = np.unique(problem_targets)
print('Problem targets:')
for name in problem_targets:
    print(f'\t{name}')
if len(problem_targets):
    print('\tnone')


#%% make catalog of the targets

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt6__add-lya_transit_snr.ecsv')
cat.add_index('tic_id')
tic_ids = dbutils.target_names_stela2tic(targets)
eval_cat = cat.loc[tic_ids]
eval_cat['stela_name'] = dbutils.target_names_tic2stela(eval_cat['tic_id'])
eval_cat_hosts = catutils.planets2hosts(eval_cat)
eval_cat.write(temp_dir / 'planet_catalog.ecsv', overwrite=True)
eval_cat_hosts.write(temp_dir / 'host_catalog.ecsv', overwrite=True)


#%% check that there are files for all targets
files = list(temp_dir.glob('*.fits'))
targets_from_files = [dbutils.parse_filename(file)['target'] for file in files]
targets_from_files = np.unique(targets_from_files)
mask = ~np.isin(targets_fnames, targets_from_files)
print('No files copied over for:')
print(targets[mask])