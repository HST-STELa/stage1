import re
import os
from pathlib import Path
from datetime import datetime

from astropy.io import fits
from astropy import table
import numpy as np
from astroquery.mast import MastMissions

import utilities as utils
import database_utilities as dbutils
import hst_utilities as hstutils
import catalog_utilities as catutils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import observation_table as obs_tbl_tools


#%% batch mode or single runs?

batch_mode = True
care_level = 1 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step
confirm_file_moves = True


#%% get targets

targets = target_lists.observed_since('2025-07-14')
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
    data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{target}/hst')
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
        obs_tbl = obs_tbl_tools.load_obs_tbl(target)
        print(f'\nExisting observation table loaded for {target}:\n')
    except FileNotFoundError:
        obs_tbl = obs_tbl_tools.initialize(files_science)
        print(f'\nObservation table initialized for {target}:\n')
    obs_tbl.pprint(-1,-1)

    utils.query_next_step(batch_mode, care_level, 2)


#%% setup for MAST query

    hst_database = MastMissions(mission='hst')
    dnld_dir = data_dir / 'downloads'
    os.makedirs(dnld_dir, exist_ok=True)


#%% find new science data

    print(f'\nSearching for new science files in the archive for {target}.')

    results = hst_database.query_object(f'TIC {tic_id}',
                                        radius=3,
                                        sci_instrume="COS,STIS",
                                        sci_spec_1234="G140M,G140L,E140M,G130M,G160M",
                                        sci_status='PUBLIC',
                                        sci_aec='S', # science not calibration
                                        select_cols="sci_operating_mode sci_instrume".split())
    mode = results['sci_operating_mode'].filled('').tolist()
    inst = results['sci_instrume'].filled('').tolist()
    mask_accum = np.char.count(mode, 'ACCUM') > 0
    mask_stis = np.char.count(inst, 'STIS') > 0
    mask_cos = np.char.count(inst, 'COS') > 0
    mask_suffix_sets = ((mask_stis & mask_accum, 'RAW'),
                        (mask_stis & ~mask_accum, 'TAG'),
                        (mask_cos, ['X1D', 'RAWTAG', 'RAWTAG_A', 'RAWTAG_B']))
    file_tbls = []
    for mask, sfxs in mask_suffix_sets:
        slctd_results = results[mask]
        if len(slctd_results):
            datasets = hst_database.get_unique_product_list(slctd_results)
            filtered = hst_database.filter_products(datasets, file_suffix=sfxs, extension='fits')
            file_tbls.append(filtered)
    files_in_archive = table.vstack(file_tbls)

    # list observations already deemed unusable
    unusable_ids = obs_tbl['archive id'][~obs_tbl['usable'].filled(True)]

    new_files_mask = []
    for file_info in files_in_archive:
        if file_info['authz_primary_identifier'].lower() in unusable_ids:
            new_files_mask.append(False)
        else:
            files = list(data_dir.glob(f'*{file_info['filename']}'))
            new_files_mask.append(len(files) == 0)
    new_files = files_in_archive[new_files_mask]

    if new_files:
        print()
        print('Attempting download of:')
        new_files['instrument_name filters filename access'.split()].pprint(-1)
        print()
    else:
        print('\nNo new science files found in the archive.\n')

    utils.query_next_step(batch_mode, care_level, 2)


#%% download and rename new files

    if new_files:
        manifest = hst_database.download_products(new_files, download_dir=dnld_dir, flat=True)
        # dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True,
        #                                       into_target_folders=False, confirm=confirm_file_moves)
        dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=target, into_target_folders=False, confirm=confirm_file_moves)

        utils.query_next_step(batch_mode, care_level, 2)


#%% make sure info on all files are in the obs tbl

    fits_files = list(data_dir.glob('*.fits'))
    sci_files = [file for file in fits_files if hstutils.is_raw_science(file)]
    catutils.set_index(obs_tbl, 'archive id')
    for file in sci_files:
        pieces = dbutils.parse_filename(file)
        id = pieces['id']
        filename = f'{pieces['id']}_{pieces['type']}.fits'
        if id in obs_tbl['archive id']:
            i = obs_tbl.loc_indices[id]
            assert not hasattr(i, '__iter__')
            ksf = obs_tbl['key science files'][i]
            if isinstance(ksf, np.ndarray):
                ksf = ksf.tolist()
            if np.ma.is_masked(ksf):
                ksf = []
            if filename in ksf:
                continue
            ksf.append(filename)
            obs_tbl['key science files'][i] = ksf
        else:
            row = {}
            pi = fits.getval(file, 'PR_INV_L')
            row['pi'] = pi
            row['observatory'] = 'hst'
            row['archive id'] = pieces['id']
            row['science config'] = pieces['config']
            row['start'] = re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime'])
            row['program'] = int(pieces['program'].replace('pgm', ''))
            row['key science files'] = [filename]
            row['supporting files'] = []
            row['usable'] = True
            row['reason unusable'] = ''
            row['flags'] = []
            row['notes'] = ''
            obs_tbl.add_row(row)
            for name in 'supporting files,usable,reason unusable,notes,flags'.split(','):
                obs_tbl[name].mask[-1] = True

    cleaned_sci_files = [np.unique(sfs).tolist() for sfs in obs_tbl['key science files']]
    obs_tbl['key science files'] = cleaned_sci_files


#%% download supporting acquisitions and wavecals

    print(f'Searching for supporting files for {target} observations.')
    new_supporting_files = False
    for row in obs_tbl:
        usable = row['usable']
        if not np.ma.is_masked(usable) and not usable:
            continue
        path = dbutils.find_stela_files_from_hst_filenames(row['key science files'], data_dir)[0]
        pieces = dbutils.parse_filename(path)
        id = pieces['id']
        i = obs_tbl.loc_indices[id]
        if obs_tbl['supporting files'].mask[i]:
            supporting_files = {}
        else:
            supporting_files = obs_tbl['supporting files'][i]

        # look for acquisitions
        acq_tbl_w_spts = hstutils.locate_associated_acquisitions(path, additional_files=('SPT',))
        if len(acq_tbl_w_spts) == 0:
            continue
        acq_tbl = hst_database.filter_products(acq_tbl_w_spts, file_suffix=['RAW', 'RAWACQ'])
        acq_tbl.add_index('obsmode')

        # record these
        for acq_row in acq_tbl:
            file = str(acq_row['filename'])
            acq_type = acq_row['obsmode']
            inst = acq_row['inst']
            if 'COS' in inst:
                aqt = acq_type.lower()
            elif 'STIS' in inst:
                if acq_type == 'ACQ/PEAK':
                    files = acq_tbl.loc[acq_type]['filename']
                    assert len(files) <= 2
                    aqt = 'acq/peakd' if min(files) == acq_row['filename'] else 'acq/peakxd'
                else:
                    aqt = acq_type.lower()
            else:
                NotImplementedError

            if aqt in supporting_files:
                if file > supporting_files[aqt]:
                    supporting_files[aqt] = file
            else:
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

        obs_tbl['supporting files'][i] = supporting_files

    if not new_supporting_files:
        print('All supporting files present, nothing downloaded.')

    utils.query_next_step(batch_mode, care_level, 2)


#%% move and rename downloaded files

    if new_supporting_files:
        dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=target, into_target_folders=False, confirm=confirm_file_moves)
        # dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True, into_target_folders=False, confirm=confirm_file_moves)

        utils.query_next_step(batch_mode, care_level, 2)


#%% identify files missing data

    print(f'\nChecking for empty science files for {target}.')
    reasons = dict(nodata='No data taken.',
                   no_gs_lock='Guide star tracking not locked.')
    for i, row in enumerate(obs_tbl):
        usable = row['usable']
        if not np.ma.is_masked(usable) and not usable:
            continue
        reject = False
        shortnames = row['key science files'][:] # [:] to copy, otherwise may be modified in the table
        scifiles = dbutils.find_stela_files_from_hst_filenames(shortnames, data_dir)
        pieces = dbutils.parse_filename(scifiles[0])
        if 'tag' in pieces['type']:
            counts = 0
            for file_info in scifiles:
                h = fits.open(file_info)
                counts += len(h[1].data['time'])
                if counts <= 100:
                    reject = True
                    reason = reasons['nodata']
        elif 'raw' in pieces['type']:
            exptimes = [fits.getval(f, 'exptime', 1) for f in scifiles]
            if np.all(np.array(exptimes) == 0):
                reject = True
                reason = reasons['nodata']

        # alert if there are odd header flags
        h = fits.open(scifiles[0])
        hdr = h[0].header + h[1].header
        if hdr['FGSLOCK'] != 'FINE':
            reject = True
            reason = reasons['no_gs_lock']
        if hdr['expflag'] != 'NORMAL':
            print(f'!! Odd exposure flag value of {hdr['expflag']} for {shortnames[0]}.')
        if hdr['exptime'] == 0 and not reject:
            print(f'!! Exposure time of zero for {shortnames[0]} despite apparently having data.')
            note = ''
            for shortname, file_info in zip(shortnames, scifiles):
                with fits.open(file_info, mode='update') as h:
                    if len(h[2].data['start']):
                        raise NotImplementedError
                    if h[1].header['exptime'] == 0:
                        start, stop = h[1].data['time'][[0, -1]]
                        data = np.recarray((1,), dtype=[('START', 'f8'), ('STOP', 'f8')])
                        data['START'] = start
                        data['STOP'] = stop
                        h[2].data = data
                        h[1].header['EXPTIME'] = stop - start
                        h[0].header['TEXPTIME'] = stop - start
                        h.flush()
                        note += f'{shortname} had data but header set to zero exposure time. Manually replaced GTIs based on first and last photon count.'
                    else:
                        raise ValueError('Weird. Look into this.')
            if len(note):
                obs_tbl['notes'][i] = note


        if reject:
            obs_tbl['usable'][i] = False
            obs_tbl['reason unusable'][i] = reason

    any_bad_files = any(~obs_tbl['usable'].filled(True))
    if any_bad_files:
        print('\nSome files found to be missing data:\n')
        obs_tbl.pprint(-1,-1)

    utils.query_next_step(batch_mode, care_level, 2)


#%% note to self
    """It is still worth checking the acquisitions, so don't delete the files yet."""


#%% check obs_tbl

    print(
          f"""
          {target} observing table following updates.
          """
    )
    obs_tbl.pprint(-1,-1)

    utils.query_next_step(batch_mode, care_level, 2)


#%% save obs_tbl

    print(f'\nSaving obs_tbl for {target}.\n')
    obs_tbl.sort('start')
    obs_tbl.meta['last archive query'] = datetime.now().isoformat()
    obs_tbl.write(obs_tbl_tools.get_path(target), overwrite=True)

    utils.query_next_step(batch_mode, care_level, 1)


#%% dummy cell
    """just needed so my code collapses prettily :)"""
#%% loop close

  except StopIteration:
    break