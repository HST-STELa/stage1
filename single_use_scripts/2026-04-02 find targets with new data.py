
from tqdm import tqdm
from astropy import table

import paths
import database_utilities as dbutils

from processing import target_lists
from processing import observation_table as obt

#%%

targets3 = target_lists.eval_no(3)
targets = target_lists.eval_no(1) + target_lists.eval_no(2) + targets3

sets = []
coadds_needed = []
for target in tqdm(targets):
    newconfigs = []
    tbl = obt.load_obs_tbl(target)
    configs = list(set(tbl['science config'].tolist()))
    for config in configs:
        fd = paths.target_data(target)
        files = dbutils.find_coadd_or_x1ds(target, out_of_transit_coadd=False, instruments=config, directory=fd)
        if len(files) > 1 and all(f.name.endswith('_x1d.fits') for f in files):
            coadds_needed.append([target, config])
        for file in files:
            ll = dbutils.one_glob(file.parent, f'{target}.{config}.*.line_fluxes.ecsv', error_on_multiple=False)
            if ll is None:
                if len(newconfigs) == 0 or newconfigs[-1] != config:
                    newconfigs.append(config)

    # cull unnecssary ones
    nice_fuv = {'hst-stis-g140l', 'hst-cos-g130m', 'hst-cos-g160m', 'hst-cos-g140l'}
    if 'hst-stis-g140m' in newconfigs:
        if target not in targets3:
            newconfigs.remove('hst-stis-g140m')
        elif nice_fuv & set(newconfigs):
            newconfigs.remove('hst-stis-g140m')
    if 'hst-stis-e140m' in newconfigs:
        if nice_fuv & set(newconfigs):
            newconfigs.remove('hst-stis-e140m')

    pairs = [[target, config] for config in newconfigs]
    sets.extend(pairs)

print(len(sets))

#%%

targets, configs = list(zip(*sets))
tbl = table.Table((targets, configs), names='target config'.split())
tbl.sort('target')
tbl.add_index('target')

#%% cull

tbl.write(paths.scratch / '2026-04-02 targets with new data.ecsv', overwrite=True)

#%% coadds

targets, configs = list(zip(*coadds_needed))
cd = table.Table((targets, configs), names='target config'.split())
keep = []
for i in range(len(cd)):
    target, config = tbl[i]
    if target in tbl['target']:
        slctd = tbl.loc[target]
        if config in slctd['config']:
            keep.append(i)
cd = cd[keep]
cd.pprint(-1,-1)
