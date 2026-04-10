import numpy as np
from astropy import table

import paths


#%% make sure no actual rawtag files in database

rawtag_files = sorted(paths.data_targets.rglob('*rawtag*.fits'))

print(len(rawtag_files))


#%% load obs tables, remove rawtags from cos sci files, save

obstbl_paths = sorted(paths.data_targets.rglob('*observation-table*.ecsv'))

#%% test on one

obstbl_path, = sorted(paths.data_targets.rglob('k2-3*observation-table*.ecsv'))
obstbl = table.Table.read(obstbl_path)
obstbl.pprint(-1,-1)

i, row = 0, obstbl[0]
key = 'key science files'
scifiles = row[key]
if type(scifiles) is list:
    scifiles, = [f for f in scifiles if '_x1d.fits' in f]
    obstbl[key][i] = scifiles

obstbl.pprint(-1, -1)

#%%

key = 'key science files'
obstbl_paths = sorted(paths.data_targets.rglob('*observation-table*.ecsv'))

for obstbl_path in obstbl_paths:
    obstbl = table.Table.read(obstbl_path)
    print()
    print(obstbl_path.name.split('.')[0])
    print()
    diff = obstbl[['science config', 'archive id', key]].copy()

    for i, row in enumerate(obstbl):
        scifiles = row[key]
        config = row['science config']
        if type(scifiles) is list:
            if 'hst-stis' in config:
                scifiles, = scifiles
            elif 'hst-cos' in config:
                scifiles = [f for f in scifiles if '_x1d.fits' in f]
                if len(scifiles) == 0:
                    scifiles = row['archive id'] + '_x1d.fits'
                else:
                    scifiles, = scifiles
            else:
                raise ValueError
            obstbl[key][i] = scifiles

    if np.all(diff[key] == obstbl[key]):
        print('no modifications')
        print()
        print('=' * 40)
        continue

    diff[key + ' new'] = obstbl[key].copy()
    diff.pprint(-1,-1)

    save = input('Save? enter/n ')

    if save == '':
        obstbl.write(obstbl_path, overwrite=True)

    print('='*40)