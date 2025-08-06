from pathlib import Path
import shutil as sh
import os

from astropy.table import Table
from astropy import units as u
import h5py
import numpy as np

import paths
import utilities as utils
import database_utilities as dbutils

from stage1_processing import target_lists
from stage1_processing import preloads

#%% settings, including option to loop (batch mode)

rotation_amplitude_saturation = 0.1
rotation_amplitude_Ro1 = 0.01
jitter_saturation = 0.1
jitter_Ro1 = 0.01

obstimes_temp = [-1.5, 0, 18, 19.5, 21, 22.5, 24] * u.h
obstimes = obstimes_temp - 21 * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s

batch_mode = True
care_level = 2 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step


#%% tables

stela_name_tbl = preloads.stela_names.copy()
planet_catalog = preloads.planets.copy()
planet_catalog.add_index('tic_id')


#%% move model transit spectra into target folders

delivery_folder = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2025-07-30 transit predictions')
files = list(delivery_folder.glob('*.h5'))

targnames_ethan = [file.name[:-4] for file in files]
targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
targnames_file = dbutils.target_names_stela2file(targnames_stela)

for targname, file in zip(targnames_file, files):
    planet = file.name[-4]
    newname = f'{targname}.outflow-tail-model.na.na.transit-{planet}.h5'
    newfolder = paths.target_data(targname) / 'transit predictions'
    if not newfolder.exists():
        os.mkdir(newfolder)
    sh.copy(file, newfolder / newname)


#%% targets

targets = target_lists.eval_no(1)


#%% SKIP? set up loop for batch processing (skip if not in batch mode)

if batch_mode:
    print("When 'Continue?' prompts appear, hit enter to continue, anything else to break out of the loop.")

itertargets = iter(targets)
while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:

#%% move to next target

    target = next(itertargets)
    tic_id = stela_name_tbl.loc['hostname_file', target]['tic_id']
    planets = planet_catalog.loc[tic_id]
    targfolder = paths.target_data(target)


#%% set stellar variability parameters



#%% loop through planets based on outflow sims

    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    for planet in planets:
        letter = planet['pl_letter']
        transit_simulation_file, = targfolder.rglob(f'*outflow-tail-model*transit-{letter}.h5')

        with h5py.File('gj341b.h5', 'r') as f:
            print("Top-level keys:", list(f.keys()))

            # Example: Check shapes
            print("intensity shape:", f['intensity'].shape)
            print("tgrid:", f['tgrid'][:5])  # Preview first 5 time steps
            print("wavgrid:", len(f['wavgrid']))

            # System parameters (stored as attributes)
            for key, val in f['system_parameters'].attrs.items():
                print(f"{key} = {val}")

            # print(np.shape(f['intensity'][0, :, 100]))
            print(f['tgrid'][:])

            j = 75
            for i in range(27):
                plt.plot(f['tgrid'][:], f['intensity'][:][i, :, j], 'b')

            v = (f['wavgrid'][:][j] / 1215.67e-8 - 1) * 3e5
            plt.xlabel('Time (h)')
            plt.ylabel(f'Transmission at {v:.0f} km s-1')
        transits =

        utils.query_next_step(batch_mode, care_level, 1)

#%% loop close

  except StopIteration:
    break