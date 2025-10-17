import os
import re
from pathlib import Path

import paths

#%% dry run

target_dirs = [f for f in paths.data_targets.glob('*') if f.is_dir()]
for dir in target_dirs:
    hst_dir = dir / 'hst'
    obsfiles = list(dir.glob('*.fits')) + list(dir.glob('*.html'))
    for file in obsfiles:
        hst_path = hst_dir / file.name
        if hst_path.exists():
            print(f'Deleting: {file.name}')
        else:
            print(f'Not present in hst dir: {file.name}')
    print('\n\n')


#%% do it

target_dirs = [f for f in paths.data_targets.glob('*') if f.is_dir()]
for dir in target_dirs:
    obsfiles = list(dir.glob('*.fits')) + list(dir.glob('*.html'))
    for file in obsfiles:
        os.remove(file)


#%% observation tables and png plots

target_dirs = [f for f in paths.data_targets.glob('*') if f.is_dir()]
for dir in target_dirs:
    hst_dir = dir / 'hst'
    obsfiles = list(dir.glob('*.png')) + list(dir.glob('*observation-table*'))
    if len(obsfiles) == 0:
        continue
    for file in obsfiles:
        hst_path = hst_dir / file.name
        if hst_path.exists():
            print(f'Deleting: {file.name}')
        else:
            print(f'Not present in hst dir: {file.name}')
    print('\n\n')


#%% do it

target_dirs = [f for f in paths.data_targets.glob('*') if f.is_dir()]
for dir in target_dirs:
    obsfiles = list(dir.glob('*.png')) + list(dir.glob('*observation-table*'))
    if len(obsfiles) == 0:
        continue
    for file in obsfiles:
        os.remove(file)