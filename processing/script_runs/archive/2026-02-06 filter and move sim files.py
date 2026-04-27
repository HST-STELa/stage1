import re
import os
import shutil as sh
from pathlib import Path
from datetime import datetime

import paths
import database_utilities as dbutils


# %% paths

models_dnld = Path('/Users/parke/Downloads/batch2_high_sw')
models_inbox = paths.inbox / '2026-02-06 transit predictions high velocity wind'
os.makedirs(models_inbox, exist_ok=True)


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


# %% copy tools

files = list(models_dnld.glob('*.h5'))

threshold_date = datetime(2026, 2, 3, 0)

newfiles = []
for f in files:
    modtime = f.stat().st_mtime
    modtime = datetime.fromtimestamp(modtime)

    if modtime > threshold_date:
        newfiles.append(f)

def check_and_move(fn):
    print('To be copied: ')
    fn(dry_run=True)
    for_reals = input('\nProceed? (enter/n)')
    if for_reals == '':
        fn(dry_run=False)

def move_to_inbox(dry_run=True):
    for file in newfiles:
        if dry_run:
            print(f'{file.name} --> {'/'.join(models_inbox.parts[-2:])}/{file.name}')
        else:
            sh.copy(file, models_inbox / file.name)

names_planets = [parse_ethan_targname(file) for file in newfiles]
targnames_ethan, planet_suffixes = zip(*names_planets)
targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
targnames_file = dbutils.target_names_stela2file(targnames_stela.astype(str))

def move_to_target_folders(dry_run=True):
    raise ValueError("Somehow this failed and put the wrong names on files. Probs bc I didn't rename files to newfiles below. Careful on resuse.")
    for targname, planet, file in zip(targnames_file, planet_suffixes, files):
        newname = f'{targname}-{planet}.outflow-tail-model.transmission-grid.h5'
        newfolder = paths.target_data(targname) / 'transit predictions'

        if dry_run:
            print(f'{file.name} --> {'/'.join(newfolder.parts[-2:])}/{newname}')
        else:
            if not newfolder.exists():
                os.mkdir(newfolder)
            sh.copy(file, newfolder / newname)


# %% copy to inbox

check_and_move(move_to_inbox)


# %% copy to target folders

check_and_move(move_to_target_folders)

# %% try again this time using inbox files bc I deleted the dnlds already

files = list(models_inbox.glob('*.h5'))

names_planets = [parse_ethan_targname(file) for file in files]
targnames_ethan, planet_suffixes = zip(*names_planets)
targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
targnames_file = dbutils.target_names_stela2file(targnames_stela.astype(str))

def move_to_target_folders(dry_run=True):
    for targname, planet, file in zip(targnames_file, planet_suffixes, files):
        newname = f'{targname}-{planet}.outflow-tail-model.transmission-grid.h5'
        newfolder = paths.target_data(targname) / 'transit predictions'

        if dry_run:
            print(f'{file.name} --> {'/'.join(newfolder.parts[-2:])}/{newname}')
        else:
            if not newfolder.exists():
                os.mkdir(newfolder)
            sh.copy(file, newfolder / newname)

check_and_move(move_to_target_folders)