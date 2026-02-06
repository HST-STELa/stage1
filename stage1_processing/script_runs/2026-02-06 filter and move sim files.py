import re
import os
import shutil as sh
from pathlib import Path
from datetime import datetime

import numpy as np

import paths
import database_utilities as dbutils


# %% paths

models_dnld = Path('/Users/parke/Downloads/batch2_high_sw')
models_inbox = paths.inbox / '2026-02-06 transit predictions high velocity wind'


# %% naming

odd_names = {
    'aumic': 'au-mic',
    'dstuca': 'ds-tuc-a',
}
def parse_ethan_targname(file):
    name = file.name
    name = name.replace('.h5', '')
    if re.findall(r'0\d$', name): # name ends in a number, so planet must be a 2-digit number
        targname = name[:-2]
        suffix = name[-2:]
    else:
        targname = name[:-1]
        suffix = name[-1:]
    targname = odd_names.get(targname, targname)
    return targname, suffix


# %% delete tmp files

print('To be deleted:  ')
tmpfiles = list(models_dnld.glob('*.tmp'))
for f in tmpfiles:
    print(f'\t{dbutils.path_string_last_n(f, 3)}')
ans = input('Proceed with deletion (enter/n)?')
if ans == '':
    for f in tmpfiles:
        os.remove(f)


# %% identify and delete old files

files = list(models_dnld.glob('*.h5'))

files = snr_files = list(paths.data_targets.rglob('*tail-model*sigmas*.ecsv'))


threshold_date = datetime(2026, 1, 28, 17)

oldfiles = []
for f in files:
    modtime = f.stat().st_mtime
    modtime = datetime.fromtimestamp(modtime)

    if modtime < threshold_date:
        oldfiles.append(f)
targets_all = [f.name.split('.')[0] for f in oldfiles]
targets_all = [dbutils.split_hostname_planet_letter(tgt, '-')[0] for tgt in targets_all]
targets_all = set(targets_all)

targets_all = sorted(list(targets_all))
target_sets = np.array_split(list(targets_all), 3)
targets = target_sets[i_set]



names_planets = [parse_ethan_targname(file) for file in files]
targnames_ethan, planet_suffixes = zip(*names_planets)
targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
targnames_file = dbutils.target_names_stela2file(targnames_stela.astype(str))

def move_files(dry_run=True):
    for targname, planet, file in zip(targnames_file, planet_suffixes, files):
        newname = f'{targname}-{planet}.outflow-tail-model.transmission-grid.h5'
        newfolder = paths.target_data(targname) / 'transit predictions'

        if dry_run:
            print(f'{file.name} --> {'/'.join(newfolder.parts[-2:])}/{newname}')
        else:
            if not newfolder.exists():
                os.mkdir(newfolder)
            sh.copy(file, newfolder / newname)

move_files(dry_run=True)
for_reals = input('\nProceed with copying the files? (enter/n)')
if for_reals == '':
    move_files(dry_run=False)



#%% get targets that haven't been run yet
files = snr_files = list(paths.data_targets.rglob('*tail-model*sigmas*.ecsv'))
from datetime import datetime
threshold_date = datetime(2026, 1, 28, 17)

oldfiles = []
for f in files:
    modtime = f.stat().st_mtime
    modtime = datetime.fromtimestamp(modtime)

    if modtime < threshold_date:
        oldfiles.append(f)
targets_all = [f.name.split('.')[0] for f in oldfiles]
targets_all = [dbutils.split_hostname_planet_letter(tgt, '-')[0] for tgt in targets_all]
targets_all = set(targets_all)

targets_all = sorted(list(targets_all))
target_sets = np.array_split(list(targets_all), 3)
targets = target_sets[i_set]