from pathlib import Path
import shutil as sh
import os

import paths
import utilities as utils
import database_utilities as dbutils

from stage1_processing import target_lists


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

#%% option to loop (batch mode)

batch_mode = True
care_level = 2 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step


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

#%% estimate SNR across phase space



    utils.query_next_step(batch_mode, care_level, 1)

#%% loop close

  except StopIteration:
    break