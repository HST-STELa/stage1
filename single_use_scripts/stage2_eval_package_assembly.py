import os
import shutil
from pathlib import Path
import copy

import numpy as np
from astropy import table

import catalog_utilities as catutils
import database_utilities as dbutils
import paths

#%% settings

eval_no = 1

temp_dir = Path(f'/Users/parke/Downloads/stela_stage2_eval_pkg{eval_no}')
if not temp_dir.exists():
    os.mkdir(temp_dir)
temp_dir_lya = Path(f'/Users/parke/Downloads/stela_stage2_eval_pkg{eval_no}/lya_data')
if not temp_dir_lya.exists():
    os.mkdir(temp_dir_lya)


#%% load up the latest export of the obs progress table
path_main_table = dbutils.pathname_max(paths.status_input, 'Observation Progress*.xlsx')
progress_table = catutils.read_excel(path_main_table)


#%% assemble x1dfiles

lya_folder = 'lya_data'
eval_mask = progress_table["Stage 2 Eval\nBatch"] == 1
targets = progress_table["Target"][eval_mask]
targets_fnames = dbutils.target_names_stela2file(targets)
problem_targets = []
for name in targets_fnames:
    targdir = paths.data_targets / name
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


#%% --------------------------------- AFTER LYA & XRAY DELIVERY ---------------------------------
"""
    --------------------------------- AFTER LYA & XRAY DELIVERY ---------------------------------
"""


#%% distribute lya reconstructions

lya_inbox_folder = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2025-07-01 Lya reconstructions')
lya_recon_files = list(lya_inbox_folder.rglob('*lya-recon.csv'))
for file in lya_recon_files:
    pieces = dbutils.parse_filename(file)
    tgt_folder = paths.data_targets / pieces['target'] / 'reconstructions'
    if not tgt_folder.exists():
        os.mkdir(tgt_folder)
    tgt_path = tgt_folder / file.name
    shutil.copy(file, tgt_path)


#%% distribute x-ray spectra

xray_inbox_folder = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2025-06-30 X-ray')
xray_files = list(xray_inbox_folder.glob('*.fits'))
nodir = []
for file in xray_files:
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    if targname not in dbutils.stela_name_tbl['hostname']:
        targname, = dbutils.resolve_stela_name_w_simbad([targname])
    targname_file, = dbutils.target_names_stela2file([targname])

    recon_folder = paths.data_targets / targname_file / 'reconstructions'
    newname = f'{targname_file}.apec.na.na.na.xray_recon.fits'
    tgt_path = recon_folder / newname
    shutil.copy(file, tgt_path)



#%% add x-ray spectra to package

xray_files = list(paths.data_targets.rglob('*xray_recon.fits'))
tgt_folder = paths.data_targets / '../packages/2025-06-16.stela_stage2_eval_pkg1/xray_reconstructions'
for file in xray_files:
    newpath = tgt_folder / file.name
    shutil.copy(file, newpath)


#%% add line fluxes to package

line_files = list(paths.data_targets.rglob('*line-flux-table.ecsv'))
tgt_folder = paths.data_targets / '../packages/2025-06-16.stela_stage2_eval_pkg1/fuv_line_fluxes'
for file in line_files:
    tbl = table.Table.read(file)
    tbl.sort('wave')
    tbl.write(file, overwrite=True)
    newpath = tgt_folder / file.name
    shutil.copy(file, newpath)