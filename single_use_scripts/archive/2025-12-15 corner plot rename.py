import paths
import os

#%%

files = list(paths.data_targets.rglob('*plot-snr_corner.*'))
for f in files:
    os.remove(f)

#%%

files = list(paths.data_targets.rglob('*plot-snr-corner.*'))
for f in files:
    newname = f.name.replace('plot-snr-corner' , 'plot-det-vol-corner')
    print(f"{f.name} --> {newname}")
    os.rename(f, f.parent / newname)