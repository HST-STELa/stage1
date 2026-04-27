import re
import os
from datetime import datetime
import warnings

from astropy.io import fits
from astropy import table
import numpy as np
from astroquery.mast import MastMissions
from astropy import units as u

import utilities as utils
import database_utilities as dbutils
import hst_utilities as hstutils
import catalog_utilities as catutils
import paths

from processing import target_lists
from processing import preloads
from processing import observation_table as obt


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = target_lists.everything_in_database()
i = targets.index('lhs1140')
targets = targets[i+1:]
missing_acq_action = 'raise' # warn or raise
batch_mode = True
care_level = 0 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step
confirm_file_moves = False
dnld_from_insts = 'COS,STIS'
dnld_from_specs = "G140M,G140L,E140M,G130M,G160M"
dnld_availability = 'PUBLIC,PROPRIETARY'


#%% setup for MAST query

hst_database = MastMissions(mission='hst')
# note that you need to have created and stored a token for this, see
# https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.MastClass.html
# you can specify the exact toke you want to use with token=...
hst_database.login()

#%% target iterator

itertargets = iter(targets)


#%% SKIP? set up batch processing (skip if not in batch mode)

if batch_mode:
    print("When 'Continue?' prompts appear, hit enter to continue, anything else to break out of the loop.")

while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:


#%% move to next target

    target = next(itertargets)


#%% prep for target processing

    print(
    f"""
    {'='*len(target)}
    {target.upper()}
    {'='*len(target)}
    """
    )

    tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
    data_dir = paths.target_hst_data(target)
    if not data_dir.exists():
        os.makedirs(data_dir)


#%% find key science files already downloaded

    """
    for it to be a science file, 
    - the target must be the target
    - it must be tag or raw
    - if raw, mode must be accum
    """
    files_tag_or_raw = (list(data_dir.glob('*tag.fits'))
                        + list(data_dir.glob('*rawtag.fits'))
                        + list(data_dir.glob('*rawtag_[ab].fits'))
                        + list(data_dir.glob('*raw.fits'))
                        + list(data_dir.glob('*raw_[ab].fits')))
    files_acqs_removed = [f for f in files_tag_or_raw if 'ACQ' not in fits.getval(f, 'obsmode')]
    files_raw_only_if_accum = []
    for f in files_acqs_removed:
        pieces = dbutils.parse_filename(f.name)
        if pieces['type'] in ['raw', 'raw_a', 'raw_b']:
            if 'ACCUM' in fits.getval(f, 'obsmode'):
                files_raw_only_if_accum.append(f)
        else:
            files_raw_only_if_accum.append(f)

    files_science = files_raw_only_if_accum


#%% create or load table of observation information

    try:
        obs_tbl = obt.load_obs_tbl(target)
        print(f'\nExisting observation table loaded for {target}:\n')
    except FileNotFoundError:
        obs_tbl = obt.initialize(files_science)
        print(f'\nObservation table initialized for {target}:\n')
    obs_tbl.pprint(-1,-1)

    care_level = utils.query_next_step(batch_mode, care_level, 2)


#%% set download directory

    dnld_dir = data_dir / 'downloads'
    os.makedirs(dnld_dir, exist_ok=True)


#%% make sure info on all files are in the obs tbl

    fits_files = list(data_dir.glob('*.fits'))
    sci_files = [file for file in fits_files if hstutils.is_key_science_file(file)]
    catutils.set_index(obs_tbl, 'archive id')
    for file in sci_files:
        pieces = dbutils.parse_filename(file)
        id = pieces['id']
        filename = f'{pieces['id']}_{pieces['type']}.fits'
        if id in obs_tbl['archive id']:
            i = obs_tbl.loc_indices[id]
            cur = obs_tbl['key science files'][i]
            if np.ma.is_masked(cur) or cur is None:
                obs_tbl['key science files'][i] = [filename]
            else:
                items = list(cur) if isinstance(cur, (list, tuple)) else [cur]
                if filename not in items:
                    items.append(filename)
                obs_tbl['key science files'][i] = items
        else:
            pi = fits.getval(file, 'PR_INV_L')
            row = {
                'pi': pi,
                'observatory': 'hst',
                'archive id': pieces['id'],
                'science config': pieces['config'],
                'start': re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime']),
                'program': int(pieces['program'].replace('pgm', '')),
                'key science files': [filename],
                'supporting files': {},
                'usable': False,
                'usability status': '',
                'reason unusable': '',
                'flags': [],
                'notes': [],
            }
            obs_tbl.add_row(row)
            ni = len(obs_tbl) - 1
            obs_tbl.update_usability(ni, 'mask')
            for colname in ('supporting files', 'flags', 'notes'):
                obs_tbl[colname].mask[ni] = True

    cleaned_sci_files = [np.unique(sfs).tolist() for sfs in obs_tbl['key science files']]
    obs_tbl['key science files'] = cleaned_sci_files


#%% clear supporting file column

    for i in range(len(obs_tbl)):
        obs_tbl['supporting files'][i] = None
        obs_tbl['supporting files'].mask[i] = True


#%% download supporting acquisitions and wavecals

    print(f'Searching for supporting files for {target} observations.')
    new_supporting_files = False
    for obs_row in obs_tbl:
        scifiles = dbutils.find_stela_files_from_hst_filenames(obs_row['key science files'], data_dir)
        path = scifiles[0]
        pieces = dbutils.parse_filename(path)
        i = obs_row.index

        dq = hstutils.assess_key_science_files_data_quality(scifiles)
        if dq.reject:
            obs_tbl.update_usability(i, 'unusable', dq.reason)
            continue

        if obs_tbl['supporting files'].mask[i]:
            supporting_files = {}
        else:
            supporting_files = obs_tbl['supporting files'][i]

        # look for acquisitions
        search_radius = 10*u.arcmin / 2**7
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            acq_tbl_w_spts = []
            while len(acq_tbl_w_spts) == 0 and search_radius < 10*u.arcmin:
                acq_tbl_w_spts = hstutils.locate_nearby_acquisitions(path, search_radius, additional_files=('SPT',))
                search_radius *= 2
        if len(acq_tbl_w_spts) == 0:
            obs_tbl.update_usability(i, 'has issues')
            obs_tbl.add_flag(i, obt.flag_menu['bad acq'])
            note = obt.notes_menu['acq not found'].format(search_radius=search_radius)
            obs_tbl.add_note(i, note)
            missing_acq_msg = f'No acquisition found for {path.name} within a {search_radius} search radius.'
            if missing_acq_action == 'warn':
                warnings.warn(missing_acq_msg)
            else:
                raise ValueError(missing_acq_msg)
        else:
            acq_tbl = hstutils.infer_associated_acquisitions(path, acq_tbl_w_spts)

            # if stis and there are two peaks, number them to differentiate
            # (one will be a peakd and the other a peakxd,
            # but we won't know for sure until looking at the files after downloading)
            if 'stis' in obs_row['science config']:
                acq_tbl['obsmode'] = acq_tbl['obsmode'].astype('object')
                modes = acq_tbl['obsmode']
                peaks = np.char.count(modes.astype('str'), 'PEAK') > 0
                if sum(peaks) == 2:
                    modes[peaks] = np.char.add(modes[peaks].astype('str'), ['1', '2'])
                    acq_tbl['obsmode'][peaks] = modes[peaks]

            # list acquisitions in the obs table
            for acq_row in acq_tbl:
                aqt = acq_row['obsmode'].lower()
                file = acq_row['filename']
                supporting_files[aqt] = file

            # download missing ones
            not_present = [len(list(data_dir.glob(f'*{name}'))) == 0 for name in acq_tbl_w_spts['filename']]
            if any(not_present):
                new_supporting_files = True
                manifest = hst_database.download_products(acq_tbl_w_spts[not_present], download_dir=dnld_dir, flat=True)

        # record wavecal
        if 'hst-stis' in pieces['config']:
            wav_filename = pieces['id'] + '_wav.fits'
            supporting_files['wavecal'] = wav_filename
            wavfiles = list(data_dir.rglob(f'*{wav_filename}'))
            if len(wavfiles) > 1:
                raise ValueError('Uh oh.')
            if len(wavfiles) == 0:
                results = hst_database.query_criteria(sci_data_set_name=pieces['id'])
                datasets = hst_database.get_unique_product_list(results)
                filtered = hst_database.filter_products(datasets, file_suffix=['WAV'])
                if len(filtered) == 0:
                    print(f'!! No WAV found in the HST database for {path.name}.')
                if len(filtered) > 1:
                    raise ValueError('Uh oh.')
                manifest = hst_database.download_products(filtered, download_dir=dnld_dir, flat=True)

        if supporting_files:
            obs_tbl['supporting files'][i] = supporting_files

        # assert not obs_tbl['supporting files'].mask[i] and supporting_files != {}

    if not new_supporting_files:
        print('All supporting files present, nothing downloaded.')

    care_level = utils.query_next_step(batch_mode, care_level, 2)


#%% move and rename downloaded files

    if new_supporting_files:
        dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=target, into_target_folders=False, confirm=confirm_file_moves)
        # dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True, into_target_folders=False, confirm=confirm_file_moves)

        care_level = utils.query_next_step(batch_mode, care_level, 2)


#%% verify that acq files are present for every observation

    for row in obs_tbl:
        if not row.usable(True) or any('no acquisition found' in n for n in row.get('notes', [])):
                continue
        sfs = row['supporting files']
        config = row['science config']
        if obs_tbl._is_null_like(sfs):
            raise ValueError
        sfkeys = list(sfs.keys())
        if 'cos' in config:
            if not ('acq/image' in sfkeys or 'acq/peakxd' in sfkeys):
                raise ValueError
        elif 'stis' in config:
            if not 'acq' in sfkeys:
                raise ValueError


#%% check obs_tbl

    print(
          f"""
          {target} observing table following updates.
          """
    )
    obs_tbl.pprint(-1,-1)

    care_level = utils.query_next_step(batch_mode, care_level, 2)


#%% save obs_tbl

    print(f'\nSaving obs_tbl for {target}.\n')
    obs_tbl.write(obt.get_path(target), overwrite=True)

    care_level = utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    break