import os
from pathlib import Path
from datetime import datetime

from astropy import time
from astropy import units as u

import paths
import database_utilities as dbutils


#%%


fds = [fd for fd in paths.data_targets.glob('*') if fd.is_dir()]
fds = sorted(fds)

def move_ava_files(dry_run=True):
    for fd in fds:
        files = fd.glob('*detection-sigmas*')
        fd_new = fd / 'transit predictions'

        if files and not fd_new.exists():
            os.mkdir(fd_new)

        for f in files:
            name = f.name
            name_new = name.replace('_ava_test', '')
            f_new = fd_new / name_new
            if dry_run:
                old = dbutils.path_string_last_n(f, 3)
                new = dbutils.path_string_last_n(f_new, 4)
                print(f'{old} --> {new}')
            else:
                os.rename(f, f_new)

#%%

move_ava_files(dry_run=True)

#%%

move_ava_files(dry_run=False)


#%%

fds = [fd for fd in paths.data_targets.glob('**/transit predictions') if fd.is_dir()]
fds = sorted(fds)

def delete_opaque_tail_max_files(dry_run=True):
    if dry_run:
        print('Will delete:')
    for fd in fds:
        files = sorted(fd.glob('*simple-opaque-tail.detection-sigmas-max*'))
        for f in files:
            tmod = datetime.fromtimestamp(f.stat().st_mtime)
            if time.Time(tmod) > time.Time(datetime.now()) - 10*u.d:
                raise ValueError
            if dry_run:
                print(f'\t{dbutils.path_string_last_n(f,3 )}')
            else:
                os.remove(f)

#%%

delete_opaque_tail_max_files(True)

#%%

delete_opaque_tail_max_files(False)


