import os
import shutil
import warnings
from random import choices
from pathlib import Path

import numpy as np
from tqdm import tqdm
from astropy import table
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt

import paths
import utilities as utils
import catalog_utilities as catutils
import database_utilities as dbutils

from stage1_processing import target_lists


#%%

staging_area = paths.packages / '2026-03-10.stage2.eval3.staging_area'
eval_no = 3

obscat = catutils.read_excel(paths.observation_progress_google_sheet_xlsx_export)

planetcat = catutils.load_and_mask_ecsv(
    paths.selection_intermediates / 'chkpt4__fill-basic_properties.ecsv')


#%% id batch 3 hosts

eval_targets = target_lists.eval_no(3)
tics = dbutils.stela_name_tbl.loc['hostname_file', eval_targets]['tic_id']


#%% find targets that need coadds

needs_coadd = []
for host in eval_targets:
    x1dpath = dbutils.find_coadd_or_x1ds(
        host,
        out_of_transit_coadd=True,
        instruments=('stis-g140m', 'stis-e140m'),
        directory=paths.target_data(host)
    )
    if len(x1dpath) > 1:
        needs_coadd.append(host)


#%% copy lya data

lya_folder = staging_area / 'lya_data'
if not lya_folder.exists():
    os.mkdir(lya_folder)

for host in eval_targets:
    x1dpath, = dbutils.find_coadd_or_x1ds(
        host,
        out_of_transit_coadd=True,
        instruments=('stis-g140m', 'stis-e140m'),
        directory=paths.target_data(host)
    )
    newpath = lya_folder / x1dpath.name
    if not newpath.exists():
        shutil.copy(x1dpath, newpath)


#%% planets missing from catalog

missing = ~np.isin(tics, planetcat['tic_id'])
missing_names = eval_targets[missing]


#%% make planet and host tables

planetcat.add_index('tic_id')
eval_cat = planetcat.loc[tics]
eval_cat['stela_planet_suffix'] = dbutils.planet_suffixes(eval_cat)
eval_cat['stela_name'] = dbutils.target_names_tic2stela(eval_cat['tic_id'])
eval_cat_hosts = catutils.planets2hosts(eval_cat)

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


#%% save planet and host tables

eval_cat.write(staging_area / 'planet_catalog.ecsv', overwrite=True)
eval_cat_hosts.write(staging_area / 'host_catalog.ecsv', overwrite=True)


#%% standardize and distribute x-ray spectra

xray_inbox_folder = paths.inbox / '2026-03-22 x-ray'
xray_files = sorted(xray_inbox_folder.glob('*.fits'))
for file in tqdm(xray_files):
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    if targname not in dbutils.stela_name_tbl['hostname']:
        targname, = dbutils.groom_hst_names_for_simbad([targname])
        targname, = dbutils.resolve_stela_name_w_simbad([targname])
    targname_file, = dbutils.target_names_stela2file([targname])
    if targname_file not in eval_targets:
        warnings.warn(f"{targname_file} in provided x-ray spectra but not in eval batch. Skipping.")
        continue

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


#%% inspect a random assortment

inspct_files = choices(xray_files, k=5)
for file in inspct_files:
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    targname_file, = dbutils.target_names_stela2file([targname])
    recon_fd = paths.target_data(targname_file) / 'reconstructions'
    xfile = dbutils.one_glob(recon_fd, '*xray-recon.fits', error_on_multiple=True)
    assert xfile is not None, 'No xray file found.'
    h = fits.open(xfile)
    data = h[1].data

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(8,4))
    fig.suptitle(targname)
    fig.supxlabel('wavelength')
    ax0.step(data['wavelength'], data['flux'])
    ax0.set_ylabel('flux')
    ax0.set_ylim(1e-21, None)
    ax0.set_yscale('log')
    ax1.step(data['wavelength'], data['bin_width'])
    ax1.set_ylabel('bin width')



#%% close plots

plt.close('all')


#%% figure out mimatch in x-ray targets

targets_xray = []
for file in xray_files:
    targname = file.name[:-10]
    targname = targname.replace('_', ' ')
    targname_file, = dbutils.target_names_stela2file([targname])
    targets_xray.append(targname_file)

eval_targets = set(eval_targets)
targets_xray = set(targets_xray)

xray_not_eval = targets_xray - eval_targets
eval_not_xray = eval_targets - targets_xray


#%% load host catalog to see if that's the problem

hostcat = catutils.load_and_mask_ecsv(staging_area / 'host_catalog.ecsv')
hostnames = dbutils.stela_name_tbl.loc['tic_id', hostcat['tic_id']]['hostname_file']
cat_not_xray = set(hostnames) - targets_xray

#%% notes

"""
TOI-1730 is a rename. Christian used lhs 1903. Wasp-84 is a mystery.

Update: it was a mix up on his end. wasp-84 was swapped for l22-69. 
"""

#%% add x-ray spectra to package

destination = staging_area / 'xray_reconstructions'
if not destination.exists():
    os.mkdir(destination)

missed = []
for target in eval_targets:
    folder = paths.target_data(target) / 'reconstructions'
    xray_file = dbutils.one_glob(folder, '*xray-recon.fits', error_on_multiple=True)
    if xray_file:
        newfile = destination / xray_file.name
        shutil.copy(xray_file, newfile)
        print(f'{Path(*xray_file.parts[-3:])} -> {Path(*newfile.parts[-3:])}')
    else:
        missed.append(target)

print(missed)


#%% lya: check that all targets have files

lya_inbox = paths.inbox / '2026-03-24 lya reconstructions'
lya_files = sorted(lya_inbox.glob('*lya_recon.csv'))
lya_targets = [f.name.split('.')[0] for f in lya_files]
assert set(lya_targets) == set(eval_targets)

#%% lya: plot a random smattering

lya_files_random = choices(lya_files, k=5)
for f in lya_files_random:
    tbl = table.Table.read(f)

    plt.figure()
    plt.title(f.name.split('.')[0])
    plt.plot(tbl['wave_lya'], tbl['lya_model unconvolved_median'])
    plt.plot(tbl['wave_lya'], tbl['lya_intrinsic unconvolved_median'])


#%% distribute and stage lya-recon files

dry_run = False
fd_lya_stage = staging_area / 'lya_reconstructions'
for f in lya_files:
    target = f.name.split('.')[0]
    fd_targrecon = paths.target_data(target) / 'reconstructions'
    newname = f.name.replace('lya_recon', 'lya-recon')
    tgpath = fd_targrecon / newname
    stgpath = fd_lya_stage / newname
    if dry_run:
        print(f"{dbutils.path_string_last_n(f, 2)} --> {dbutils.path_string_last_n(tgpath, 3)}")
        print(f"{dbutils.path_string_last_n(f, 2)} --> {dbutils.path_string_last_n(stgpath, 3)}")
        print()
    else:
        shutil.copy(f, tgpath)
        shutil.copy(f, stgpath)


