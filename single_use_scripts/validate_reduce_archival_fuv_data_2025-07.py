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
target = 'GJ 486'
tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
targname_file = str(targname_file)
data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{targname_file}')


#%% load table of observation information

path_obs_tbl = data_dir / f'{targname_file}.observation-table.ecsv'
obs_tbl = catutils.load_and_mask_ecsv(path_obs_tbl)
backup = obs_tbl.copy() # in case I screw something up and want to revert real quick


#%% setup for MAST query

hst_database = MastMissions(mission='hst')
dnld_dir = data_dir / 'downloads'
if not dnld_dir.exists():
    os.mkdir(dnld_dir)


#%% download science data

results = hst_database.query_object(f'TIC {tic_id}',
                                    radius=3,
                                    sci_instrume="COS,STIS",
                                    sci_spec_1234="G140L,E140M,G130M,G160M",
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

new_files_mask = []
for file_info in files_in_archive:
    files = list(data_dir.glob(f'*{file_info['filename']}'))
    new_files_mask.append(len(files) == 0)
new_files = files_in_archive[new_files_mask]


#%% download and rename new files

manifest = hst_database.download_products(new_files, download_dir=dnld_dir, flat=True)
# dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True)
dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=targname_file)


#%% make sure info on all files is in the obs tbl

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
        obs_tbl.add_row(row)
        for name in 'supporting files,usable,reason unusable'.split(','):
            obs_tbl[name].mask[-1] = True

cleaned_sci_files = [np.unique(sfs).tolist() for sfs in obs_tbl['key science files']]
obs_tbl['key science files'] = cleaned_sci_files

#%% filter for just the broadband files

bb_files = []
for file in sci_files:
    pieces = dbutils.parse_filename(file)
    grating = pieces['config'].split('-')[2]
    if grating in ['g140l', 'e140m', 'g130m', 'g160m']:
        bb_files.append(file)

#%% download supporting acquisitions and wavecals

for path in bb_files:
    pieces = dbutils.parse_filename(path)
    id = pieces['id']
    i = obs_tbl.loc_indices[id]
    if obs_tbl['supporting files'].mask[i]:
        supporting_files = {}
    else:
        continue

    # look for acquisitions
    acq_tbl_w_spts = hstutils.locate_associated_acquisitions(path, additional_files=('SPT',))
    if len(acq_tbl_w_spts) == 0:
        continue
    acq_tbl = hst_database.filter_products(acq_tbl_w_spts, file_suffix=['RAW', 'RAWACQ'])
    acq_tbl.add_index('obsmode')

    # record these
    acq_types = np.unique(acq_tbl['obsmode'].tolist())
    # keep all non-peak acqs, but only keep the most recent peakups
    for atp in acq_types:
        supporting_files[atp.lower()] = acq_tbl.loc[atp]['filename'].tolist()

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

dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, target_name=targname_file)
# dbutils.rename_and_organize_hst_files(dnld_dir, data_dir, resolve_stela_name=True)


#%% file deletion tool

def delete_all_files(obs_tbl_row):
    row = obs_tbl_row
    shortnames = row['key science files'][:]
    for name in row['supporting files'].values():
        if len(name[0]) > 1:
            shortnames.extend(name)
        else:
            shortnames.append(name)
    ids = [re.search(r'(\w{9})_\w+.\w+$', name).groups()[0] for name in shortnames]
    dbutils.delete_files_by_hst_id(ids, data_dir)


#%% identify, record, and delete files missing data

reasons = dict(nodata='No data taken.',
               no_gs_lock='Guide star tracking not locked.')
for i, row in enumerate(obs_tbl):
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
                    start, stop = h[1].data['time'][[0,-1]]
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

        # delete all files
        delete_all_files(row)


#%% identify rows with broadband data

gratings = [config.split('-')[-1] for config in obs_tbl['science config']]
bb_mask = [grating in ['g140l', 'e140m', 'g130m', 'g160m'] for grating in gratings]
i_bb, = np.nonzero(bb_mask)

#%% review ACQs. record and delete failures.

acq_filenames = []
for supfiles in obs_tbl[bb_mask]['supporting files']:
    for type, file in supfiles.items():
        if 'acq' in type.lower():
            acq_filenames.append(file)
acq_filenames = np.unique(acq_filenames)

delete = []
for acq_name in acq_filenames:
    bad_acq = False
    assoc_obs_mask = [acq_name in list(sfs.values()) for sfs in obs_tbl['supporting files']]

    acq_file, = dbutils.find_stela_files_from_hst_filenames(acq_name, data_dir)
    print(f'\n\nAcquistion file {acq_file.name} associated with:\n')
    obs_tbl[assoc_obs_mask]['start,science config,key science files'.split(',')].pprint(-1,-1)
    print('\n\n')
    if 'hst-stis' in acq_file.name:
        stages = ['coarse', 'fine', '0.2x0.2']
        stis.tastis.tastis(str(acq_file))
        h = fits.open(acq_file)

        if 'mirvis' in acq_file.name:
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
        if len(h) > 7:
            raise NotImplementedError
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
        i_delete, = np.nonzero(assoc_obs_mask)
        delete.extend(i_delete)

for i in delete:
    delete_all_files(obs_tbl[i])


#%% now reduce data for stis files


#%% change to data directory and renew file list

os.chdir(data_dir)
instruments = ['hst-stis-g140l', 'hst-stis-e140m']
scifiles = dbutils.find_data_files('tag', instruments=instruments)
scifiles = list(map(str, scifiles))

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
crds.bestrefs.assign_bestrefs(scifiles, sync_references=True, verbosity=10)


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
for scifile in scifiles:
    scifile = str(scifile)
    root = dbutils.modify_file_label(scifile, '')
    root = str(root).replace('_', '')
    if '_tag.fits' in scifile:
        rawfile = scifile.replace('_tag', '_raw')
        asn_id = fits.getval(scifile, 'asn_id').lower()
        wavfile, = glob.glob(f'*{asn_id}_wav.fits')
        # exposure
        stis.inttag.inttag(scifile, rawfile)
        status = stis.calstis.calstis(rawfile, wavecal=wavfile, outroot=root) # exposure
    else:
        rawfile = scifile.replace('_tag', '_raw')
        asn_id = fits.getval(scifile, 'asn_id').lower()
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
        iln, = plt.plot(x, y.T, color='r', lw=0.5, alpha=0.5, label='intial pipeline extraction')
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
        nln, = plt.plot(x, y.T + dy, color='w', alpha=0.5, label='after manual correction')
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

obs_tbl['flags'] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')

bb_tbl = obs_tbl[bb_mask]
configs = np.unique(bb_tbl['science config'])
pwd = Path('.')
for config in configs:
    config_mask = bb_tbl['science config'] == config
    ids = bb_tbl['archive id'][config_mask]
    for id in ids:
        fig = plt.figure()
        file, = pwd.glob(f'*{id}_x1d.fits')
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
            delete_all_files(i_delete)
            continue

        flags_ans = input('What other flags should be recorded? Separate with commas, no spaces: ')
        flags = flags_ans.split(',')
        for i in i_mask:
            obs_tbl['flags'][i_mask] = flags

    plt.close('all')

obs_tbl['usable'][obs_tbl['usable'].mask] = True


#%% coadd multiple exposures


x1dfiles = dbutils.find_data_files('x1d', instruments=instruments, directory=data_dir)
if len(x1dfiles) > 1:
    # if e140m and g140m, just keep the g140m
    insts = [dbutils.parse_filename(f)['config'] for f in x1dfiles]
    if any(np.char.count(insts, 'g140m') > 0) & any(np.char.count(insts, 'e140m') > 0):
        x1dfiles = dbutils.find_data_files('flt', instruments='hst-stis-g140m', targets=[targname_file])

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
    config = 'hst-stis-' + '+'.join(np.unique(np.char.replace(pieces_tbl['config'], 'hst-stis-', '')))
    date = f'{min(pieces_tbl['datetime'])}..{max(pieces_tbl['datetime'])}'
    program = 'pgm' + '+'.join(np.unique(np.char.replace(pieces_tbl['program'], 'pgm', '')))
    id = f'{len(x1dfiles)}exposure'
    coadd_name = f'{target}.{config}.{date}.{program}.{id}_coadd.fits'
    os.rename(main_coadd_file, f'{coadd_name}')



#%% back to regular path
os.chdir(main_dir)


#%% suppress some annoying warnings
warnings.filterwarnings("ignore", message="The converter")
warnings.filterwarnings("ignore", message="line style")


#%% plot extraction locations for stis

saveplots = True
fltfiles = dbutils.find_data_files('flt', instruments='hst-stis', directory=data_dir)

for ff in fltfiles:
    img = fits.getdata(ff, 1)
    f1 = dbutils.modify_file_label(ff, 'x1d')
    td = fits.getdata(f1, 1)
    fig = plt.figure()
    plt.imshow(np.cbrt(img), aspect='auto')
    plt.title(ff.name)

    x = np.arange(img.shape[1]) + 0.5
    i = 36 if 'e140m' in ff.name else 0
    y = td['extrlocy'][i]
    ysz = td['extrsize'][i]
    plt.plot(x, y.T, color='w', lw=0.5, alpha=0.5)
    plt.fill_between(x, y - ysz/2, y + ysz/2, color='w', lw=0, alpha=0.5)
    for ibk in (1,2):
        off, sz = td[f'bk{ibk}offst'][i], td[f'bk{ibk}size'][i]
        ym = y + off
        y1, y2 = ym - sz/2, ym + sz/2
        plt.fill_between(x, y1, y2, color='0.5', alpha=0.5, lw=0)

    if saveplots:
        dpi = fig.get_dpi()
        fig.set_dpi(150)
        plugins.connect(fig, plugins.MousePosition(fontsize=14))
        htmlfile = str(ff).replace('_flt.fits', '_plot-extraction.html')
        mpld3.save_html(fig, htmlfile)
        fig.set_dpi(dpi)


#%% save observation manifest

obs_tbl.sort('start')
obs_tbl.write(path_obs_tbl, overwrite=True)


