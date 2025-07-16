import os
import re

import paths

#%%  get all plot files, look at some example names
files = list(paths.data.rglob('*plot*.png'))
names = [f.name for f in files]


#%% rename _plot-spec, dry run

batch = [f for f in files if '_plot-spec.png' in f.name]
for f in batch[:10]:
    newname = f.name.replace('_plot-spec.png', '_x1d.plot.png')
    print(f"{f.name} --> {newname}")


#%% do rename

for f in batch:
    newname = f.name.replace('_plot-spec.png', '_x1d.plot.png')
    os.rename(f, f.parent / newname)


#%% rename tool

def rename_replace(batch, old, new, dry_run):
    for f in batch:
        newname = f.name.replace(old, new)
        if dry_run:
            print(f"{f.name} --> {newname}")
        else:
            os.rename(f, f.parent / newname)


#%% dry run

batch = list(paths.data.rglob('*x1dplot*'))
rename_replace(batch, 'x1dplot', 'x1d.plot', dry_run=True)

#%% do rename
rename_replace(batch, 'x1dplot', 'x1d.plot', dry_run=False)

#%% coadd_plot

batch = list(paths.data.rglob('*coadd_plot*'))
rename_replace(batch, 'coadd_plot', 'coadd.plot', dry_run=True)
rename_replace(batch, 'coadd_plot', 'coadd.plot', dry_run=False)


#%% x1d_plot

old = 'x1d_plot'
new = 'x1d.plot'
batch = list(paths.data.rglob(f'*{old}*'))
rename_replace(batch, old, new, dry_run=True)
rename_replace(batch, old, new, dry_run=False)


#%% add pgm id into names that don't have it

files = list(paths.data.rglob('*.hst-*'))
files = [f for f in files if 'pgm' not in f.name]
names = [f.name for f in files]

#%% dry run

for f in files:
    pieces = f.name.split('.')
    root = '.'.join(pieces[:3])
    targpath = paths.target_hst_data(pieces[0])
    otherfile = list(targpath.glob(f'{root}*pgm*'))[0]
    pgmstr, = re.findall(r'pgm\d{5}', otherfile.name)
    pieces.insert(3, pgmstr)
    newname = '.'.join(pieces)
    print(f"{f.name} --> {newname}")

#%% for reals

for f in files:
    pieces = f.name.split('.')
    root = '.'.join(pieces[:3])
    targpath = paths.target_hst_data(pieces[0])
    otherfile = list(targpath.glob(f'{root}*pgm*'))[0]
    pgmstr, = re.findall(r'pgm\d{5}', otherfile.name)
    pieces.insert(3, pgmstr)
    newname = '.'.join(pieces)
    os.rename(f, f.parent / newname)

#%% _plot-extraction.html

old = '_plot-extraction'
new = '.plot-extraction'
batch = list(paths.data.rglob(f'*{old}*'))
rename_replace(batch, old, new, dry_run=True)
rename_replace(batch, old, new, dry_run=False)


#%% _plot_spec.html

old = '_plot-spec'
new = '.plot'
batch = list(paths.data.rglob(f'*{old}*'))
rename_replace(batch, old, new, dry_run=True)
rename_replace(batch, old, new, dry_run=False)


#%%  get all plot files, look at some example names
files = list(paths.data.rglob('*plot*'))
names = [f.name for f in files]


#%%  missing the x1d

files = list(paths.data.rglob('*plot.*'))
batch = [f for f in files if ('_x1d.' not in f.name) and ('_coadd' not in f.name)]
names = [f.name for f in batch]

rename_replace(batch, '.plot', '_x1d.plot', dry_run=True)
rename_replace(batch, '.plot', '_x1d.plot', dry_run=False)


#%% move obs tables into hst dirs

orphan_obs_files = list(paths.data.glob('*/*observation-table.ecsv'))
for f in orphan_obs_files:
    os.rename(f, f.parent / 'hst' / f.name)