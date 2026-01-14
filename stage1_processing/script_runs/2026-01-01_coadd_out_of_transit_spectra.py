import os
import warnings
from pathlib import Path
import re

import matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import table
import numpy as np

import database_utilities as dbutils
import utilities as utils
import paths

from stage1_processing import target_lists
from stage1_processing import observation_table as obs_tbl_tools
from stage1_processing import preloads




#%% catalog setup

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    planets = preloads.planets.copy()
planets.add_index('tic_id')
planets['pl_id'] = dbutils.planet_suffixes(planets)
planets.add_index('pl_id')


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

transit_avoidance_start = -2 # in optical transit half-durations
transit_avoidance_end = 1/8 # in fractions of planetary orbit

batch_mode = True
care_level = 0 # 0 = just loop with no stopping, 1 = pause before each loop

targets = target_lists.new_data(4)
instruments = ['hst-cos'] # currently this does nothing

if care_level == 1:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('agg')

#%% target iterator

itertargets = iter(targets)


#%% SKIP? set up batch processing (skip if not in batch mode)

if batch_mode:
    print("When 'Continue?' prompts appear, hit enter to continue, anything else to break out of the loop.")

counter = 0
while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:


#%% move to next target

    target = next(itertargets)


#%% change to data directory and load file list

    banner = f'{counter}/{len(targets)} {target}'
    counter += 1
    print(
        f"""
    {'=' * len(banner)}
    {banner.upper()}
    {'=' * len(banner)}
    """
    )

    data_dir = paths.target_hst_data(target)
    if not data_dir.exists():
        os.makedirs(data_dir)
    os.chdir(data_dir)

    obs_tbl = obs_tbl_tools.load_obs_tbl(target)

    obs_tbl_usbl = obs_tbl[obs_tbl['usable'].filled(True)]
    print(f'\n{target} observation table:\n')
    obs_tbl.pprint(-1,-1)


#%% mark out of transit exposures

    tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
    targplanets = planets.loc['tic_id', tic_id]
    if isinstance(targplanets, table.Row):
        targplanets = planets[[targplanets.index]]
    phase_cols = [col for col in obs_tbl.colnames if col.startswith('phase_') and not col.endswith('_err')]
    assert len(phase_cols) > 0
    planets_oot = []
    for col in phase_cols:
        _, id = col.split('_')
        planet = targplanets.loc['pl_id', id]
        P = planet['pl_orbper']
        T = planet['pl_trandur']
        trange = [transit_avoidance_start * T/2,
                  transit_avoidance_end * P]
        planet_oot = ~utils.is_in_range(obs_tbl[col].filled(0), *trange)
        planets_oot.append(planet_oot)
    all_oot = np.all(planets_oot, axis=0)
    obs_tbl['out of transit'] = all_oot


#%% coadd multiple exposures or orders, both all available and just out of transit

    use_tbl = dbutils.filter_observations(
        obs_tbl,
        config_substrings=['g140m', 'e140m', 'g130m', 'g160m', 'g140l'],
        usable=True,
        usability_fill_value=True,
        exclude_flags=['flare'])
    configs = np.unique(use_tbl['science config'])

    for config in configs:
        config_tbl = dbutils.filter_observations(use_tbl, config_substrings=[config])
        if (len(config_tbl) == 1) and (len(re.findall('e140m|g130m|g160m', config)) == 0):
            print(f'{target} {config} only one file and it does not have multiple orders/segments. skipping.')
            continue

        shortnames = np.char.add(config_tbl['archive id'], '_x1d.fits')
        x1dfiles = dbutils.find_stela_files_from_hst_filenames(shortnames)
        x1dfiles = np.array(x1dfiles)

        all = np.ones(len(config_tbl), bool)
        oot = config_tbl['out of transit'].data
        labels = ['coadd', 'oot_coadd']

        for mask, label in zip((all, oot), labels):
            if sum(mask) == 0:
                if mask is all:
                    raise ValueError
                else:
                    print(f'{target} {config} no out of transit exposures. '
                          f'Only a coadd from all exposures will be made.')
                    continue
            slctd_x1dfiles = x1dfiles[mask]

            # copy just the files we want into a /temp folder
            path_temp = Path('./temp')
            if not path_temp.exists():
                os.mkdir(path_temp)
            [os.rename(f, path_temp / f.name) for f in slctd_x1dfiles]

            !swrapper -i ./temp -o ./temp -x

            # move files back into the main directory
            [os.rename(path_temp / f.name, f) for f in slctd_x1dfiles]

            # keep only the main coadd, move it into the main directory
            main_coadd_file, = list(path_temp.glob('*aspec.fits'))
            intermediates = list(path_temp.glob('*cspec.fits'))
            for intermediate in intermediates:
                os.remove(intermediate)

            pieces_list = [dbutils.parse_filename(f) for f in slctd_x1dfiles]
            pieces_tbl = table.Table(rows=pieces_list)
            target, = np.unique(pieces_tbl['target'])
            assert len(np.unique(pieces_tbl['config'])) == 1
            config =  pieces_tbl['config'][0]
            date = f'{min(pieces_tbl['datetime'])}--{max(pieces_tbl['datetime'])}'
            program = 'pgm' + '+'.join(np.unique(np.char.replace(pieces_tbl['program'], 'pgm', '')))
            id = f'{len(slctd_x1dfiles)}exposure'
            coadd_name = f'{target}.{config}.{date}.{program}.{id}_{label}.fits'
            os.rename(main_coadd_file, f'{coadd_name}')

            data = fits.getdata(coadd_name, 1)
            plt.figure(figsize=(15,5))
            plt.title(coadd_name)
            plt.step(data['wavelength'].T, data['flux'].T, where='mid')

    if care_level >= 1:
        print('Pausing before next target. Click off axes on plot to continue.')
        xy = utils.click_coords()
    plt.close('all')

  except StopIteration:
    break