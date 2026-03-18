import os
import shutil

import numpy as np

import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from stage1_processing import target_lists


#%%

staging_area = paths.packages / '2026-03-10.stage2.eval3.staging_area'

obscat = catutils.read_excel(paths.observation_progress_google_sheet_xlsx_export)

planetcat = catutils.load_and_mask_ecsv(
    paths.selection_intermediates / 'chkpt4__fill-basic_properties.ecsv')


#%% id batch 3 hosts

hostnames_file = target_lists.eval_no(3)
tics = dbutils.stela_name_tbl.loc['hostname_file', hostnames_file]['tic_id']


#%% find targets that need coadds

needs_coadd = []
for host in hostnames_file:
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

for host in hostnames_file:
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
missing_names = hostnames_file[missing]


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