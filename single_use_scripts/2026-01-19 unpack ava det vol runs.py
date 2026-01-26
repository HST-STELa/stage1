import os
import shutil
from pathlib import Path
from tqdm import tqdm

import paths


#%%

dnlds = Path('/Users/parke/Downloads')
folders = [f for f in dnlds.glob('share_transit_predictions*') if '.zip' not in f.name]
folders = sorted(folders)

dest_pkg = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2026-01-19 ava detection sigmas')
for folder in tqdm(folders):
    targs = [f for f in folder.glob('*') if f.is_dir()]
    for targ in tqdm(targs):
        src = targ / 'transit predictions'
        dest = dest_pkg / targ.name / 'transit predictions'
        if not dest.exists():
            os.makedirs(dest)
        files = list(src.glob('*'))
        for f in files:
            shutil.copy(f, dest / f.name)


#%%


pkg = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2026-01-19 ava detection sigmas')
targs = [f for f in pkg.glob('*') if f.is_dir()]
for targ in tqdm(targs):
    src = targ / 'transit predictions'
    dest = paths.target_data(targ.name) / 'transit predictions'
    files = list(src.glob('*'))
    for f in files:
        if 'sigmas-max.ecsv' not in f.name:
            shutil.copy(f, dest / f.name)


#%%

files = list(paths.data_targets.rglob('*detection-sigmas-max.ecsv*'))
for f in files:
    os.remove(f)