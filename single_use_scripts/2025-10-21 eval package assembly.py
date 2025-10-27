import shutil
import os
from pathlib import Path

import paths
import database_utilities as dbutils

from stage1_processing import target_lists


#%% move line files

for eval_no in (1,2):
    targets = target_lists.eval_no(eval_no)
    try:
        targets.remove('v1298tau')
    except:
        pass
    destination = paths.data / 'packages/2025-09-26.stag2.eval2.staging_area/fuv_line_fluxes'
    destination /= f'eval_{eval_no}'
    if not destination.exists():
        os.mkdir(destination)

    for target in targets:
        folder = paths.target_data(target)
        line_file, = folder.glob('*line-flux-table.ecsv')
        shutil.copy(line_file, destination / line_file.name)


#%% distribute x-ray spectra

xray_inbox_folder = paths.inbox / '2025-10-26 x-ray'
xray_files = list(xray_inbox_folder.glob('*.fits'))
nodir = []
for file in xray_files:
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    if targname not in dbutils.stela_name_tbl['hostname']:
        targname, = dbutils.groom_hst_names_for_simbad([targname])
        targname, = dbutils.resolve_stela_name_w_simbad([targname])
    targname_file, = dbutils.target_names_stela2file([targname])

    recon_folder = paths.data_targets / targname_file / 'reconstructions'
    if not recon_folder.exists():
        os.mkdir(recon_folder)
    newname = f'{targname_file}.apec.na.na.na.xray-recon.fits'
    tgt_path = recon_folder / newname
    shutil.copy(file, tgt_path)


#%% oopsie fix _recon to -recon

files = list(paths.data_targets.rglob('*xray_recon.fits'))
for file in files:
    newname = file.name.replace('xray_recon', 'xray-recon')
    os.rename(file, file.parent / newname)


#%% add x-ray spectra to package


for eval_no in (1, 2):
    targets = target_lists.eval_no(eval_no)
    try:
        targets.remove('v1298tau')
    except:
        pass
    destination = paths.data / 'packages/2025-09-26.stag2.eval2.staging_area/xray_reconstructions'
    destination /= f'eval_{eval_no}'
    if not destination.exists():
        os.mkdir(destination)

    for target in targets:
        folder = paths.target_data(target) / 'reconstructions'
        xray_file, = folder.glob('*xray-recon.fits')
        newfile = destination / xray_file.name
        shutil.copy(xray_file, newfile)
        print(f'{Path(*xray_file.parts[-3:])} -> {Path(*newfile.parts[-3:])}')


