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
choose from:
GJ 357, GJ 367, GJ 486, HAT-P-26, HD 149026, HD 191939, HD 260655, HD 42813, HD 63935, 
HD 86226, K2-136, K2-233, L 168-9, L 98-59, LHS 1140, LHS 475, LTT 1445 A, LTT 9779, 
TOI-1201, TOI-1203, TOI-1685, TOI-1759, TOI-1774, TOI-178, TOI-260, TOI-270, TOI-421, TOI-431, 
TOI-561, TOI-741, TOI-836, WASP-107, WASP-127, WASP-38, WASP-52, WASP-77 A
"""
target = 'TOI-2134'
tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{targname_file}')

#%% find key science files

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

# check that they are targeting the science target
hdr_targets = [fits.getval(f, 'targname') for f in files_raw_only_if_accum]
simbad_names = dbutils.groom_hst_names_for_simbad(hdr_targets)
tids = query.query_simbad_for_tic_ids(simbad_names)
tids = list(map(int, tids))
stela_names = np.zeros(len(tids), 'object')
in_stela = np.isin(tids, stela_name_tbl['tic_id'])
stela_names[~in_stela] = 'not in stela'
stela_names[in_stela] = stela_name_tbl.loc[tids]['hostname']
science_target = stela_names == target

files_science = np.array(files_raw_only_if_accum)[science_target]


#%% setup for storing information on observations

path_obs_tbl = data_dir / f'{targname_file}.observation-table.ecsv'
if path_obs_tbl.exists():
    obs_tbl = table.Table.read(path_obs_tbl)
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


#%% download supporting acquisitions and wavecals

all_files = list(data_dir.glob('*.fits'))
all_ids = [dbutils.parse_filename(f)['id'].upper() for f in all_files]

for i, row in enumerate(obs_tbl):
    path, = data_dir.glob(f'*{row['key science files'][0]}')
    supporting_files = {}

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
    pieces = dbutils.parse_filename(path)
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

dbutils.rename_and_organize_hst_files(dnld_dir, data_dir / '..', resolve_stela_name=True)


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
        for file in scifiles:
            h = fits.open(file)
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
        for shortname, file in zip(shortnames, scifiles):
            with fits.open(file, mode='update') as h:
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

        # delete all files
        delete_all_files(row)


#%% from here on out, just handle stis files

stis_mask = (np.char.count(obs_tbl['science config'].tolist(), 'hst-stis') > 0) & obs_tbl['usable'].filled(True)
stis_indices, = np.nonzero(stis_mask)
stis_tbl = obs_tbl[stis_mask]
instruments = ['hst-stis-g140m','hst-stis-e140m']


#%% review ACQs. record and delete failures.

delete = []
for i in stis_indices:
    bad_acq = False
    row = obs_tbl[i]
    supporting_filenames = row['supporting files'].copy() # must copy, otherwise may be modified in the table
    for key, val in supporting_filenames.items():
        if 'acq' not in key:
            continue
        shortnames = val if len(val[0]) > 1 else [val]
        acq_files = dbutils.find_stela_files_from_hst_filenames(shortnames, data_dir)

        stages = ['coarse', 'fine', '0.2x0.2']
        for file in acq_files:
            stis.tastis.tastis(str(file))

            if 'mirvis' in file.name and 'peak' not in key:
                h = fits.open(file)
                fig, axs = plt.subplots(1, 3, figsize=[7,3])
                for j, ax in enumerate(axs):
                    data = h['sci', j+1].data
                    ax.imshow(data)
                    ax.set_title('')
                fig.suptitle(file.name)
                fig.supxlabel('dispersion')
                fig.supylabel('spatial')
                fig.tight_layout()

                print('Click outside the plots to continue.')
                xy = utils.click_coords(fig)

    answer = input('Mark acq as good? (y/n)')
    if answer == 'n':
        bad_acq = True
    plt.close('all')

    if bad_acq:
        obs_tbl['usable'][i] = False
        obs_tbl['reason unusable'][i] = 'Target not acquired or other acquisition issue.'
        delete.append(i)

for i in delete:
    delete_all_files(obs_tbl[i])


#%% change to data directory and renew file list

os.chdir(data_dir)
shortname_list = [obs_tbl[i]['key science files'] for i in stis_indices if obs_tbl['usable'].filled(True)[i]]
shortname_list = sum(shortname_list, [])
scifiles = dbutils.find_stela_files_from_hst_filenames(shortname_list)
scifiles = [str(f) for f in scifiles]

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
        if 'e140m' in fltfile.name:
            traceloc = traceloc[-8]
        stis.x1d.x1d(str(fltfile), str(x1dfile), a2center=traceloc, **x1d_params)



#%% if only one STIS exposure, extract background traces

if len(fltfiles) == 1:
    # remove existing files
    labels = "x1dbk1 x1dtrace x1dbk2".split()
    print(f'Proceed with deleting and recreating {labels[0]}, {labels[1]}, and  {labels[2]} associated with?')
    print('\n'.join([f.name for f in fltfiles]))
    _ = input('y/n? ')
    if _ == 'y':
        for fltfile in fltfiles:
            id = fits.getval(fltfile, 'asn_id').lower()
            x1d_params = get_x1dparams(fltfile)

            x1dfile = dbutils.modify_file_label(fltfile, 'x1d')
            tracelocs = fits.getdata(x1dfile, 1)['a2center']
            if 'e140m' in fltfile.name:
                traceloc = tracelocs[-8]

                mod_params = copy(x1d_params)
                mod_params['bk1size'] = mod_params['bk2size'] = 0
                mod_params['bk1offst'] = mod_params['bk2offst'] = 0
                del mod_params['extrsize']

                yt = traceloc
                szt = x1d_params['extrsize']
                sets = ((yt, szt, labels[1]),)
            else:
                traceloc, = tracelocs

                mod_params = copy(x1d_params)
                mod_params['bk1size'] = mod_params['bk2size'] = 0
                mod_params['bk1offst'] = mod_params['bk2offst'] = 0
                del mod_params['extrsize']

                y1 = traceloc + x1d_params['bk1offst']
                yt = traceloc
                y2 = traceloc + x1d_params['bk2offst']
                sz1 = x1d_params['bk1size']
                szt = x1d_params['extrsize']
                sz2 = x1d_params['bk2size']
                sets = ((y1, sz1, labels[0]),
                        (yt, szt, labels[1]),
                        (y2, sz2, labels[2]))

            for y, sz, lbl in sets:
                x1dfile = dbutils.modify_file_label(fltfile, lbl)
                if x1dfile.exists():
                    os.remove(x1dfile)

                stis.x1d.x1d(str(fltfile), str(x1dfile), a2center=y, extrsize=sz, **mod_params)






#%% if multiple STIS exposures coadd

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

#%% plot extraction locations

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


#%% plot spectra, piggyback Lya flux (O-C)/sigma

saveplots = True
vstd = (lya.wgrid_std/1215.67/u.AA - 1)*const.c.to('km s-1')
x1dfiles = dbutils.find_data_files('coadd', instruments='hst-stis', directory=data_dir)
if len(x1dfiles) == 0:
    x1dfiles = dbutils.find_data_files('x1d', instruments='hst-stis', directory=data_dir)

for xf in x1dfiles:
    h = fits.open(xf, ext=1)
    data = h[1].data
    order = 36 if 'e140m' in xf.name else 0
    spec = {}
    for name in data.names:
        spec[name.lower()] = data[name][order]

    # shift to velocity frame
    rv = target_table.loc[tic_id]['st_radv'] * u.km/u.s
    v = (spec['wavelength']/1215.67 - 1) * const.c - rv
    v = v.to_value('km s-1')

    fig = plt.figure()
    plt.title(xf.name)
    plt.step(v, spec['flux'], color='C0', where='mid')
    plt.xlim(-500, 500)

    # for plotting airglow
    if 'coadd' not in xf.name:
        size_scale = spec['extrsize'] / (spec['bk1size'] + spec['bk2size'])
        flux_factor = spec['flux'] / spec['net']
        z = spec['net'] == 0
        if np.any(z):
            raise NotImplementedError
        bk_flux = spec['background'] * flux_factor * size_scale
        plt.step(v, bk_flux, color='C2', where='mid')
    else:
        bk_flux = 0

    # line integral and (O-C)/sigma
    w, flux, error = spec['wavelength'], spec['flux'], spec['error']
    if 'g140m' in xf.name:
        above_bk = flux > bk_flux
        neighbors_above_bk = (np.append(above_bk[1:], False)
                                   & np.insert(above_bk[:-1], 0, False))
        uncontaminated = ((bk_flux < 1e-15) | (above_bk & neighbors_above_bk))
    elif 'e140m' in xf.name:
        v_helio = h[1].header['V_HELIO']
        w0 = 1215.67
        dw = v_helio/3e5 * w0
        wa, wb = w0 - dw + np.array((-0.07, 0.07))
        uncontaminated = ~((w > wa) & (w < wb))
    else:
        raise NotImplementedError
    flux[~uncontaminated], error[~uncontaminated] = 0, 0 # zero out high airglow range
    int_rng = (v > -400) & (v < 400)
    int_mask = int_rng & uncontaminated
    if sum(int_mask) > 4:
        pltflux = flux.copy()
        pltflux[~int_mask] = np.nan
        O, E = utils.flux_integral(w[int_rng], flux[int_rng], error[int_rng])
        if 'coadd' in xf.name:
            E *= np.sqrt(h[1].data['eff_exptime'].max() / 2000)
        snr = O/E
        plt.fill_between(v, pltflux, step='mid', color='C0', alpha=0.3)
    else:
        O, E = 0, 0
        snr = 0
        int_mask[:] = 0

    # kinda kloogy but simple way to get a mask that covers the same integration intervals as the data but for the
    # model profile grid
    int_mask_mod = np.interp(lya.wgrid_std.value, w, int_mask)
    int_mask_mod[int_mask_mod < 0.5] = 0
    int_mask_mod = int_mask_mod.astype(bool)

    # predicted lines
    ylim = plt.ylim()
    itarget = target_table.loc_indices[tic_id]
    predicted_fluxes = []
    for pct in (-34, 0, +34):
        n_H = ism.ism_n_H_percentile(50 + pct)
        lya_factor = lya.lya_factor_percentile(50 - pct)
        profile, = lya.lya_at_earth_auto(target_table[[itarget]], n_H, lya_factor=lya_factor, default_rv='ism')
        plt.plot(vstd - rv, profile, color='0.5', lw=1)
        # compute flux over same interval as it is computed from the observations
        # this neglects instrument braodening plus the line profile will be wrong, but it's close enough
        int_profile = profile.copy()
        int_profile[~int_mask_mod] = 0
        predicted_flux = np.trapezoid(int_profile, lya.wgrid_std)
        predicted_fluxes.append(predicted_flux.to_value('erg s-1 cm-2'))

    C = predicted_fluxes[1]
    sigma_hi = max(predicted_fluxes) - predicted_fluxes[1]
    sigma_lo = predicted_fluxes[1] - min(predicted_fluxes)
    O_C_s = (O - C) / sigma_hi
    disposition = 'PASS' if snr >= 3 else 'DROP'

    print('')
    print(f'{target}')
    print(f'\tpredicted nominal: {C:.2e}')
    print(f'\tpredicted optimistic: {max(predicted_fluxes):.2e}')
    print(f'\tmeasured: {O:.2e} ± {E:.1e}')

    flux_lbl = (f'{disposition}\n'
                f'\n'
                f'measured flux:\n'
                f'  {O:.2e} ± {E:.1e} ({snr:.1f}σ)\n'
                f'predicted flux:\n'
                f'  {C:.1e} +{sigma_hi:.1e} / -{sigma_lo:.1e}\n' 
                f'(O-C)/sigma:\n'
                f'  {O_C_s:.1f}\n'
                )
    leg = plt.legend(('signal', 'background', '-1,0,+1 σ predictions'), loc='upper right')
    plt.annotate(flux_lbl, xy=(0.02, 0.98), xycoords='axes fraction', va='top')

    plt.ylim(ylim)
    plt.xlabel('Velocity in System Frame (km s-1)')
    plt.ylabel('Flux Density (cgs)')

    if saveplots:
        pngfile = str(xf).replace('.fits', '_plot.png')
        fig.savefig(pngfile, dpi=300)

        dpi = fig.get_dpi()
        fig.set_dpi(150)
        plugins.connect(fig, plugins.MousePosition(fontsize=14))
        htmlfile = pngfile = str(xf).replace('.fits', '_plot.html')
        mpld3.save_html(fig, htmlfile)
        fig.set_dpi(dpi)


#%% save observation manifest

obs_tbl.sort('start')
obs_tbl.write(path_obs_tbl)


