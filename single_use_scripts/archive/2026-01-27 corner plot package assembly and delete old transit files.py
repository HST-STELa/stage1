import shutil
import os
from datetime import datetime

import paths

package_folder = paths.packages / '2026-01-27.detection_sigma_corner_plots'
os.makedirs(package_folder, exist_ok=True)


#%% delete old transit files dry run

getem = lambda sfx: list(paths.data_targets.rglob(f'*/transit predictions/*.{sfx}'))
files = []
sfxs = 'png pdf h5 ecsv'.split()
for sfx in sfxs:
    files += getem(sfx)

threshold_date = datetime(2026, 1, 1)

oldfiles = []
for f in files:
    modtime = f.stat().st_mtime
    modtime = datetime.fromtimestamp(modtime)

    if modtime < threshold_date:
        oldfiles.append(f)

for f in oldfiles:
    os.remove(f)


#%%

files = list(paths.data_targets.rglob('*corner.png'))

for file in files:
    shutil.copy(file, package_folder / file.name)

#%%

files = list(paths.data_targets.rglob('*/transit predictions/*sigmas-max*.png'))

for file in files:
    shutil.copy(file, package_folder / file.name)