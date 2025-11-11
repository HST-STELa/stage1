import shutil
import os
import warnings
from pathlib import Path
import re

from astropy import table
from astropy import units as u
import numpy as np
from tqdm import tqdm

import paths
import database_utilities as dbutils
import utilities as utils
import catalog_utilities as catutils

from stage1_processing import target_lists

staging_area = paths.packages / '2025-09-26.stag2.eval2.staging_area'

#%% make catalogs of hosts and planets

targets = target_lists.eval_no(1) + target_lists.eval_no(2)
cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt6__add-lya_transit_snr.ecsv')
cat.add_index('tic_id')
tic_ids = dbutils.stela_name_tbl.loc['hostname_file', targets]['tic_id']
eval_cat = cat.loc[tic_ids]
eval_cat['stela_planet_suffix'] = dbutils.planet_suffixes(eval_cat)
eval_cat['stela_name'] = dbutils.target_names_tic2stela(eval_cat['tic_id'])
eval_cat_hosts = catutils.planets2hosts(eval_cat)

eval_cat_hosts.write(staging_area / 'host_catalog.ecsv', overwrite=True)

#%% add mass range to use for sims


Mlo = np.zeros(len(eval_cat))
Mhi = np.zeros(len(eval_cat))
M = eval_cat['pl_bmasse']
e1 = eval_cat['pl_bmasseerr1']
e2 = eval_cat['pl_bmasseerr2']
lim = eval_cat['pl_bmasselim']

# no error means a calculated mass from chen and kipping, use 1 dex range msrd from their plot (5-95%)
calcd = e1.filled(0) == 0
Mlo[calcd] = M[calcd]/10**0.5
Mhi[calcd] = M[calcd]*10**0.5

msrd = ~calcd
Mlo[msrd] = M[msrd] + 2*e2[msrd]
Mhi[msrd] = M[msrd] + 2*e1[msrd]

uplim = lim.filled(0) == 1
Mlo[uplim] = 1
Mhi[uplim] = M[uplim]
assert not np.any(lim.filled(0) == -1)

young = eval_cat['flag_young'].filled(False)
Mlo[young] = 1
Mhi[young] = M[young]

Mlo = np.clip(Mlo, 0.1, np.inf)

eval_cat['pl_massgrid_lolim'] = Mlo
eval_cat['pl_massgrid_hilim'] = Mhi

assert np.all(Mlo > 0)
assert np.all(Mhi > 0)
assert np.all(Mhi > Mlo)

eval_cat.write(staging_area / 'planet_catalog.ecsv', overwrite=True)


#%% move line files

for eval_no in (1,2):
    targets = target_lists.eval_no(eval_no)
    try:
        targets.remove('v1298tau')
    except:
        pass
    destination = staging_area / 'fuv_line_fluxes'
    destination /= f'eval_{eval_no}'
    if not destination.exists():
        os.mkdir(destination)

    for target in targets:
        folder = paths.target_data(target)
        line_file, = folder.glob('*line-flux-table.ecsv')
        shutil.copy(line_file, destination / line_file.name)


#%% standardize and distribute x-ray spectra

xray_inbox_folder = paths.inbox / '2025-10-26 x-ray'
xray_files = list(xray_inbox_folder.glob('*.fits'))
for file in tqdm(xray_files):
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    if targname not in dbutils.stela_name_tbl['hostname']:
        targname, = dbutils.groom_hst_names_for_simbad([targname])
        targname, = dbutils.resolve_stela_name_w_simbad([targname])
    targname_file, = dbutils.target_names_stela2file([targname])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*did not parse as fits unit.*')
        xspec = table.Table.read(file)
    for name in xspec.colnames:
        unit = xspec[name].unit
        unitstr = str(unit)
        unitstr = unitstr.replace('photons', 'photon')
        unitstr = unitstr.replace('cts', 'ct')
        unit = u.Unit(unitstr)
        xspec[name].unit = unit
    try:
        xw, xdw, xf = xspec['wavelength'].quantity, xspec['bin_width'].quantity, xspec['flux'].quantity
    except KeyError:
        xw, xdw, xf = xspec['Wave'].quantity, xspec['bin_width'].quantity*2, xspec['Flux'].quantity

    isort = np.argsort(xw)
    xw, xdw, xf = [ary[isort] for ary in [xw, xdw, xf]]

    # xspec christian used produces rounding errors that can result in duplicate points in xw
    # so reconstruct a more accurate array
    if not np.all(np.diff(xw) > 0):
        xr = 1/xw
        xi = np.arange(len(xr))
        xp = np.polyfit(xi[[0,-1]], xr[[0,-1]], 1)
        xr_new = np.polyval(xp, xi)
        xw = 1/xr_new * xw.unit

    xwbins_test = utils.mids2bins(xw, xdw)
    assert np.all(np.diff(xwbins_test) > 0)
    xw_test = utils.midpts(xwbins_test)
    assert np.allclose(xw, xw_test, rtol=1e-5)

    recon_folder = paths.data_targets / targname_file / 'reconstructions'
    if not recon_folder.exists():
        os.mkdir(recon_folder)
    newname = f'{targname_file}.apec.na.na.na.xray-recon.fits'
    tgt_path = recon_folder / newname
    newtbl = table.Table((xw, xdw, xf), names='wavelength bin_width flux'.split())
    newtbl.meta = xspec.meta
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*HIERARCH.*')
        newtbl.write(tgt_path, overwrite=True)


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
    destination = staging_area / 'xray_reconstructions'
    destination /= f'eval_{eval_no}'
    if not destination.exists():
        os.mkdir(destination)

    for target in targets:
        folder = paths.target_data(target) / 'reconstructions'
        xray_file, = folder.glob('*xray-recon.fits')
        newfile = destination / xray_file.name
        shutil.copy(xray_file, newfile)
        print(f'{Path(*xray_file.parts[-3:])} -> {Path(*newfile.parts[-3:])}')


#%% check that all targets have fuv and x-ray files

folders = (staging_area / 'fuv_line_fluxes',
           staging_area / 'xray_reconstructions')

targets = target_lists.eval_no(1) + target_lists.eval_no(2)

for folder in folders:
    print(f'Missing in {folder.name}')
    files = list(folder.rglob('*'))
    filetargets = [f.name.split('.')[0] for f in files if '.' in f.name]
    unmatched = set(targets) - set(filetargets)
    for name in unmatched:
        print(f'\t{name}')


#%% delete phoenix v2 files

files = list(paths.data_targets.rglob('*phoenix*v2*'))
for file in files:
    os.remove(file)

#%% delete phoenix v1 files

files = list(paths.data_targets.rglob('*phoenix*'))
files = [file for file in files if '-v3.' not in file.name]
for file in files:
    os.remove(file)

#%% delete "p-17" dem files (duplicates since I took that out of the name)

files = list(paths.data_targets.rglob('*dem*p17*'))
for file in files:
    os.remove(file)