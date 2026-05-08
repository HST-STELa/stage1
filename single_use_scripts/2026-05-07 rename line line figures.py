import shutil
from pathlib import Path


fd_in = Path('/Users/parke/Google Drive/Research/STELa/scratch/lya_fuv-line_correlations_2025-07')
fd_out = Path('/Users/parke/Google Drive/Research/STELa/papers/1 overview/figures/lineline')


#%%

files = sorted(fd_in.glob('*.pdf'))

for f in files:
    name = f.name
    newname = name.replace('Flya', 'Lya')
    newname = newname.replace(' ', '')
    fnew = fd_out / newname
    shutil.copy(f, fnew)

#%%

files = sorted(fd_out.glob('*.pdf'))
for f in files:
    print(f.name)