import re
import warnings

import stistools as stis

import os
from pathlib import Path
import glob
from copy import copy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.ion()
from astropy.io import fits
from astropy import table
from astropy import constants as const
from astropy import units as u
import numpy as np
from astroquery.mast import MastMissions
import mpld3
from mpld3 import plugins

import database_utilities as dbutils
import utilities as utils
from data_reduction_tools import stis_extraction as stx
from target_selection_tools import query
import paths
import hst_utilities as hstutils
from lya_prediction_tools import lya, ism
import catalog_utilities as catutils

main_dir = '/Users/parke/Google Drive/Research/STELa/public github repo/stage1'
stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')

target_table = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
target_table = catutils.planets2hosts(target_table)
target_table.add_index('tic_id')

#%% target

"""
pick a target from the obs progress table for stage 2 eval 1 -- drop those keighley flagged tho
"""
target = 'GJ 357'
tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
targname_file = str(targname_file)
data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{targname_file}')


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

# # check that they are targeting the science target
# hdr_targets = [fits.getval(f, 'targname') for f in files_raw_only_if_accum]
# simbad_names = dbutils.groom_hst_names_for_simbad(hdr_targets)
# tids = query.query_simbad_for_tic_ids(simbad_names)
# tids = list(map(int, tids))
# stela_names = np.zeros(len(tids), 'object')
# in_stela = np.isin(tids, stela_name_tbl['tic_id'])
# stela_names[~in_stela] = 'not in stela'
# stela_names[in_stela] = stela_name_tbl.loc[tids]['hostname']
# science_target = stela_names == target
#
# files_science = np.array(files_raw_only_if_accum)[science_target]
files_science = files_raw_only_if_accum


#%% create or load table of observation information

newcols = 'flags notes'.split()
path_obs_tbl = data_dir / f'{targname_file}.observation-table.ecsv'
if path_obs_tbl.exists():
    obs_tbl = catutils.load_and_mask_ecsv(path_obs_tbl)
    for col in newcols:
        if col not in obs_tbl.colnames:
            obs_tbl[col] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')
else:
    key_science_files = []
    files_science_copy = files_science.tolist()
    while files_science_copy:
        file = files_science_copy.pop(0)
        pieces = dbutils.parse_filename(file)
        associated_files = [file.name]
        ftp = pieces['type']
        pairs = (('_a', '_b'), ('_b', '_a'))
        for s1, s2 in pairs:
            if ftp.endswith(s1):
                file2 = file.parent / file.name.replace(f'{s1}.fits', f'{s2}.fits')
                if file2 in files_science_copy:
                    files_science_copy.remove(file2)
                    associated_files.append(file.name)
        key_science_files.append(associated_files)

    n = len(key_science_files)
    columns = [
        table.MaskedColumn(data=['hst']*n, name='observatory', dtype='object'),
        table.MaskedColumn(length=n, name='science config', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='start', dtype='S20', mask=True),
        table.MaskedColumn(length=n, name='program', dtype='int', mask=True),
        table.MaskedColumn(length=n, name='pi', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='archive id', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='key science files', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='supporting files', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='usable', dtype='bool', mask=True),
        table.MaskedColumn(length=n, name='reason unusable', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='flags', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='notes', dtype='object', mask=True)
    ]
    obs_tbl = table.Table(columns)
    for i, asc_files in enumerate(key_science_files):
        pi = fits.getval(data_dir / asc_files[0], 'PR_INV_L')
        pieces = dbutils.parse_filename(asc_files[0])
        obs_tbl['archive id'][i] = pieces['id']
        obs_tbl['science config'][i] = pieces['config']
        obs_tbl['start'][i] = re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime'])
        obs_tbl['program'][i] = pieces['program'].replace('pgm', '')
        obs_tbl['pi'][i] = pi
        obs_tbl['key science files'][i] = ['.'.join(name.split('.')[-2:]) for name in asc_files]


#%% setup for MAST query

hst_database = MastMissions(mission='hst')
dnld_dir = data_dir / 'downloads'
if not dnld_dir.exists():
    os.mkdir(dnld_dir)


#%% find new science data

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


#%% download and rename new files

manifest = hst_database.download_products(new_files, download_dir=dnld_dir, flat=True)
# dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True)
dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=targname_file)


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
        if filename in obs_tbl['key science files'][i]:
            continue
        obs_tbl['key science files'][i].append(filename)
    else:
        row = {}
        pi = fits.getval(file, 'PR_INV_L')
        row['pi'] = pi
        row['observatory'] = 'hst'
        row['archive id'] = pieces['id']
        row['science config'] = pieces['config']
        row['start'] = re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime'])
        row['program'] = pieces['program'].replace('pgm', '')
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


#%% move and rename downloaded files

pass
dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=targname_file)
# dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True)
pass


#%% identify, record, and delete files missing data

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


#%% review ACQs. record and delete failures.

acq_filenames = []
usbl_tbl = dbutils.filter_observations(obs_tbl, usable=True)
for supfiles in usbl_tbl['supporting files']:
    if np.ma.is_masked(supfiles):
        continue
    for type, file in supfiles.items():
        if 'acq' in type.lower():
            acq_filenames.append(file)
acq_filenames = np.unique(acq_filenames)

for acq_name in acq_filenames:
    bad_acq = False
    def associated(sfs):
        return not np.ma.is_masked(sfs) and acq_name in list(sfs.values())
    assoc_obs_mask = [associated(sfs) for sfs in obs_tbl['supporting files']]

    acq_file, = dbutils.find_stela_files_from_hst_filenames(acq_name, data_dir)
    print(f'\n\nAcquistion file {acq_file.name} associated with:')
    obs_tbl[assoc_obs_mask]['start,science config,program,key science files'.split(',')].pprint(-1,-1)
    print('\n\n')
    if 'hst-stis' in acq_file.name:
        stages = ['coarse', 'fine', '0.2x0.2']
        stis.tastis.tastis(str(acq_file))
        h = fits.open(acq_file)

        if 'mirvis' in acq_file.name and 'PEAK' not in h[0].header['obsmode']:
            fig, axs = plt.subplots(1, 3, figsize=[7,3])
            for j, ax in enumerate(axs):
                data = h['sci', j+1].data
                ax.imshow(data)
                ax.set_title(stages[j])
            fig.suptitle(acq_file.name)
            fig.supxlabel('dispersion')
            fig.supylabel('spatial')
            fig.tight_layout()

            print('Click outside the plots to continue.')
            xy = utils.click_coords(fig)
    else:
        print('\nCOS data, no automatic eval routine\n')
        stages = ['initial', 'confirmation']
        h = fits.open(acq_file)
        if h[0].header['exptype'] == 'ACQ/SEARCH':
            raise NotImplementedError
        if h[0].header['exptype'] == 'ACQ/PEAKXD':
            print('PEAKXD acq')
            print(f'\txdisp offsets: {h[1].data['XDISP_OFFSET']}')
            print(f'\tcounts: {h[1].data['counts']}')
            print(f'\tslew: {h[0].header['ACQSLEWY']}')
        if h[0].header['exptype'] == 'ACQ/PEAKD':
            print('PEAKD acq')
            print(f'\tdisp offsets: {h[1].data['DISP_OFFSET']}')
            print(f'\tcounts: {h[1].data['counts']}')
            print(f'\tslew: {h[0].header['ACQSLEWX']}')
        if h[0].header['exptype'] == 'ACQ/IMAGE':
            fig, axs = plt.subplots(1, 2, figsize=[5,3])
            for j, ax in enumerate(axs):
                data = h['sci', j+1].data
                ax.imshow(np.cbrt(data))
                ax.set_title(stages[j])
            fig.suptitle(acq_file.name)
            fig.tight_layout()

            print('Click outside the plots to continue.')
            xy = utils.click_coords(fig)

    answer = input('Mark acq as good? (y/n)')
    if answer == 'n':
        bad_acq = True
    plt.close('all')

    if bad_acq:
        obs_tbl['usable'][assoc_obs_mask] = False
        obs_tbl['reason unusable'][assoc_obs_mask] = 'Target not acquired or other acquisition issue.'


#%% clean up unusable files

dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=True, verbose=True, directory=data_dir)
answer = input('Proceed with file deletion? y/n')
if answer == 'y':
    dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=False, directory=data_dir)


#%% change to data directory and renew file list

os.chdir(data_dir)
instruments = ['hst-stis-g140l', 'hst-stis-e140m']
# instruments = ['hst-stis-g140l', 'hst-stis-e140m', 'hst-stis-g140m']
stis_tag_files = dbutils.find_data_files('tag', instruments=instruments)
stis_tag_files = list(map(str, stis_tag_files))
if not stis_tag_files:
    print('No G140L or E140M files. Might want to look in folder to double check.')

#%% point to calibration files

setenvs = """
setenv CRDS_PATH /Users/parke/crds_cache/
setenv CRDS_SERVER_URL https://hst-crds.stsci.edu
setenv iref  ${CRDS_PATH}/references/hst/iref/
setenv jref  ${CRDS_PATH}/references/hst/jref/
setenv oref  ${CRDS_PATH}/references/hst/oref/
setenv lref  ${CRDS_PATH}/references/hst/lref/
setenv nref  ${CRDS_PATH}/references/hst/nref/
setenv uref  ${CRDS_PATH}/references/hst/uref/
"""
setenvs = setenvs.split('\n')
for line in setenvs:
    if line != '':
        _, key, path = line.split()
        path = path.replace('${CRDS_PATH}', '/Users/parke/crds_cache')
        os.environ[key] = path


#%% update calibration files

import crds
if stis_tag_files:
    crds.bestrefs.assign_bestrefs(stis_tag_files, sync_references=True, verbosity=10)


#%% set extraction parameters
pass

# note that the G140L traces get as low as 100 and G140M as low as 150, so I can't go hog wild with the backgroun regions
# x1d_params = dict()
def get_x1dparams(file):
    min_params = dict(maxsrch=0.01,
                      bksmode='off')  # no background smoothing
    if 'g140m' in file.name:
        newparams = dict(
            extrsize=19,
            bk1offst=-30, bk2offst=30,
            bk1size=20, bk2size=20
        )
    elif 'g140l' in file.name:
        newparams = dict(
            extrsize=13,
            bk1offst=-30, bk2offst=30,
            bk1size=20, bk2size=20
        )
    elif 'e140m' in file.name:
        newparams = dict(
            extrsize=7,
            bk1size=5, bk2size=5,
        )
    else:
        raise ValueError
    if 'bk1offst' in newparams:
        assert (abs(newparams['bk1offst']) - newparams['bk1size'] / 2) > (newparams['extrsize'] / 2) + 5
        assert (abs(newparams['bk2offst']) - newparams['bk2size'] / 2) > (newparams['extrsize'] / 2) + 5
    allparams = {**min_params, **newparams}
    return allparams


#%% initial extraction

overwrite_consent = False
for stis_tf in stis_tag_files:
    stis_tf = str(stis_tf)
    root = dbutils.modify_file_label(stis_tf, '')
    root = str(root).replace('_', '')
    if '_tag.fits' in stis_tf:
        rawfile = stis_tf.replace('_tag', '_raw')
        asn_id = fits.getval(stis_tf, 'asn_id').lower()
        wavfile, = glob.glob(f'*{asn_id}_wav.fits')
        # exposure
        stis.inttag.inttag(stis_tf, rawfile)
        status = stis.calstis.calstis(rawfile, wavecal=wavfile, outroot=root) # exposure
    else:
        rawfile = stis_tf.replace('_tag', '_raw')
        asn_id = fits.getval(stis_tf, 'asn_id').lower()
        status = stis.calstis.calstis(rawfile, outroot=root) # exposure
    assert status == 0


#%% identify trace location

fltfiles = dbutils.find_data_files('flt', instruments=instruments, targets=[targname_file])

print(
    f'As plots appear click the trace at either the Lya red wing (g140m) or at the C II line (g140l), then click off axes.\n'
    '\n'
    'You can zoom and pan, but coordinates will be registered for each click. '
    'Just make sure your last click is where you want it to be before clikcing off axes.\n'
    '\n'
    'If you cannot find the trace, click at x < 100 to indicate this and the default y location '
    'will be used.\n'
    '\n'
    '(Sorry that this is cloodgy :)')

ids, locs = [], []
for ff in fltfiles:
    h = fits.open(ff)
    id = h[0].header['asn_id'].lower()
    ids.append(id)

    img = h[1].data
    plt.figure()
    plt.title(Path(ff).name)
    plt.imshow(np.cbrt(img), aspect='auto')

    fx = dbutils.modify_file_label(ff, 'x1d')
    if fx.exists():
        hx = fits.open(fx)
        y = hx[1].data['extrlocy']
        x = np.arange(img.shape[1]) + 0.5
        iln = plt.plot(x, y.T, color='r', lw=0.5, alpha=0.5, label='intial pipeline extraction')[0]
    else:
        plt.annotate('calstis cross correlation to locate spectrum failed', xy=(0.05,0.99),
                     xycoords='axes fraction', va='top', color='w')
        print('Calstis did not create an x1d for this file. Maybe run this cell again to refine trace location.')

    y_predicted, offsets = stx.predicted_trace_location(h, return_pieces=True)
    pln, = plt.plot(512, y_predicted, 'y+', ms=10, label='predicted location')

    aperture = h[0].header['propaper']
    offset_lbls = [f'{key}={value:.1f}' for key, value in offsets.items()]
    offset_lbl = f'aperture: {aperture}\nY position of + from: ' + ', '.join(offset_lbls)
    plt.annotate(offset_lbl, xy=(0.05,0.05), xycoords='axes fraction', color='w', fontsize='small')

    xy = utils.click_coords()
    xclick, yclick = xy[-1]

    if xclick < 100:
        xclick, yclick = hx[1].data['a2center'], y_predicted
        plt.annotate('predicted location used', xy=(0.05, 0.95), xycoords='axes fraction', color='r', va='top')
    if fx.exists():
        # find offset to nearest trace
        yt = np.array([np.interp(xclick, x, yy) for yy in y])
        dist = np.abs(yt - yclick)
        imin = np.argmin(dist)
        dy = yclick - yt[imin]
        a2 = hx[1].data['a2center'] + dy
    else:
        a2 = yclick

    if fx.exists():
        nln = plt.plot(x, y.T + dy, color='w', alpha=0.5, label='after manual correction')[0]
        plt.legend(handles=(iln, pln, nln))
    else:
        nln = plt.axhline(a2, color='w', alpha=0.5, label='manual selection')
        plt.legend(handles=(pln, nln))

    plt.savefig(Path('/Users/parke/Google Drive/Research/STELa/scratch/yloc prediction tests')
                / ff.name.replace('_flt.fits', '_extraction.png'), dpi=300)

    locs.append(a2)
    h.close()

tracelocs = dict(zip(ids, locs))

#%% extract at user-defined trace locations

fltfiles = dbutils.find_data_files('flt', instruments=instruments, targets=[targname_file])

# remove existing x1d files
print('Proceed with deleting and recreating x1ds associated with?')
print('\n'.join([f.name for f in fltfiles]))
_ = input('y/n? ')
if _ == 'y':
    for fltfile in fltfiles:
        x1dfile = dbutils.modify_file_label(fltfile, 'x1d')
        if x1dfile.exists():
            os.remove(x1dfile)

        x1d_params = get_x1dparams(x1dfile)
        id = fits.getval(fltfile, 'asn_id').lower()
        traceloc = tracelocs[id]
        stis.x1d.x1d(str(fltfile), str(x1dfile), a2center=traceloc, **x1d_params)


#%% flag anomalous spectra, if any

"""Data could be good but look like crap, so spectra should only be flagged unusable
if they clearly differ from the norm in a serious way, I think."""

usbl_tbl = dbutils.filter_observations(obs_tbl, usable=True)
configs = np.unique(usbl_tbl['science config'])
for config in configs:
    config_mask = usbl_tbl['science config'] == config
    ids = usbl_tbl['archive id'][config_mask]
    for id in ids:
        fig = plt.figure()
        file, = data_dir.glob(f'*{id}_x1d.fits')
        plt.title(file.name)
        data = fits.getdata(file, 1)
        plt.step(data['wavelength'].T, data['flux'].T, where='mid')

    print('Click outside the plots to continue.')
    xy = utils.click_coords(fig)

    while True:
        id_ending = input('Any spectra that should be flagged ? Give last few letters of the id.\n'
                          'Hit enter if none. Prompt will loop until an empty answer is given.')
        if id_ending == '':
            break
        mask = np.char.endswith(obs_tbl['archive id'].astype(str), id_ending)
        i_mask, = np.nonzero(mask)

        usable_ans = input(f'Should {id_ending} be flagged unusable and files deleted (y/n)?')
        usable = not (usable_ans == 'y')
        if not usable:
            obs_tbl['usable'][mask] = False
            reason = input(f'Enter reason for flagging {id_ending} as unusable.')
            obs_tbl['reason unusable'][mask] = reason
            continue

        flags_ans = input('What other flags should be recorded? Separate with commas, no spaces: ')
        flags = flags_ans.split(',')
        for i in i_mask:
            obs_tbl['flags'][i_mask] = flags

    plt.close('all')

obs_tbl['usable'][obs_tbl['usable'].mask] = True


#%% clean up unusable files

dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=True, verbose=True, directory=data_dir)
answer = input('Proceed with file deletion? y/n')
if answer == 'y':
    dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=False, directory=data_dir)


#%% coadd multiple exposures

use_tbl = dbutils.filter_observations(
    obs_tbl,
    config_substrings=['e140m', 'g130m', 'g160m', 'g140l'],
    usable=True,
    usability_fill_value=True,
    exclude_flags=['flare'])
configs = np.unique(use_tbl['science config'])

for config in configs:
    config_tbl = dbutils.filter_observations(use_tbl, config_substrings=[config])
    if (len(config_tbl) == 1) and (len(re.findall('e140m|g130m|g160m', config)) == 0):
        continue

    shortnames = np.char.add(config_tbl['archive id'], '_x1d.fits')
    x1dfiles = dbutils.find_stela_files_from_hst_filenames(shortnames)

    # copy just the files we want into a /temp folder
    path_temp = Path('./temp')
    if not path_temp.exists():
        os.mkdir(path_temp)
    [os.rename(f, path_temp / f.name) for f in x1dfiles]

    !swrapper -i ./temp -o ./temp -x

    # move files back into the main directory
    [os.rename(path_temp / f.name, f) for f in x1dfiles]

    # keep only the main coadd, move it into the main directory
    main_coadd_file, = list(path_temp.glob('*cspec.fits'))
    afile, = list(path_temp.glob('*aspec.fits'))
    os.remove(afile)

    pieces_list = [dbutils.parse_filename(f) for f in x1dfiles]
    pieces_tbl = table.Table(rows=pieces_list)
    target, = np.unique(pieces_tbl['target'])
    assert len(np.unique(pieces_tbl['config'])) == 1
    config =  pieces_tbl['config'][0]
    date = f'{min(pieces_tbl['datetime'])}..{max(pieces_tbl['datetime'])}'
    program = 'pgm' + '+'.join(np.unique(np.char.replace(pieces_tbl['program'], 'pgm', '')))
    id = f'{len(x1dfiles)}exposure'
    coadd_name = f'{target}.{config}.{date}.{program}.{id}_coadd.fits'
    os.rename(main_coadd_file, f'{coadd_name}')

    data = fits.getdata(coadd_name, 1)
    plt.figure()
    plt.title(coadd_name)
    plt.step(data['wavelength'].T, data['flux'].T, where='mid')


#%% back to regular path
os.chdir(main_dir)


#%% suppress some annoying warnings
warnings.filterwarnings("ignore", message="The converter")
warnings.filterwarnings("ignore", message="line style")


#%% plot extraction locations for stis

saveplots = True

usbl_tbl = dbutils.filter_observations(obs_tbl, usable=True)
for row in usbl_tbl:
    if 'stis' not in row['science config']:
        continue
    flt_name = row['archive id'] + '_flt.fits'
    ff, = dbutils.find_stela_files_from_hst_filenames([flt_name], data_dir)
    img = fits.getdata(ff, 1)
    f1 = dbutils.modify_file_label(ff, 'x1d')
    td = fits.getdata(f1, 1)
    fig = plt.figure()
    plt.imshow(np.cbrt(img), aspect='auto')
    plt.title(ff.name)

    x = np.arange(img.shape[1]) + 0.5
    y = td['extrlocy']
    ysz = td['extrsize']
    plt.plot(x, y.T, color='w', lw=0.5, alpha=0.5)
    for yy, yysz in zip(y, ysz):
        plt.fill_between(x, yy - yysz/2, yy + yysz/2, color='w', lw=0, alpha=0.3)
    for ibk in (1,2):
        off, sz = td[f'bk{ibk}offst'], td[f'bk{ibk}size']
        ym = y + off[:,None]
        y1, y2 = ym - sz[:,None]/2, ym + sz[:,None]/2
        for yy1, yy2 in zip(y1,y2):
            plt.fill_between(x, yy1, yy2, color='0.5', alpha=0.3, lw=0)

    if saveplots:
        dpi = fig.get_dpi()
        fig.set_dpi(150)
        plugins.connect(fig, plugins.MousePosition(fontsize=14))
        htmlfile = str(ff).replace('_flt.fits', '_plot-extraction.html')
        mpld3.save_html(fig, htmlfile)
        fig.set_dpi(dpi)


#%% check obs manifest

obs_tbl.pprint(-1,-1)


#%% save observation manifest

obs_tbl.sort('start')
obs_tbl.write(path_obs_tbl, overwrite=True)


