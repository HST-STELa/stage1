import os
import warnings
from pathlib import Path
import glob
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

import stistools as stis

import database_utilities as dbutils
import utilities as utils
import paths

from stage1_processing import target_lists
from stage1_processing import observation_table as obs_tbl_tools
from data_reduction_tools import stis_extraction as stx

matplotlib.use('Qt5Agg')
plt.ion()


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = target_lists.observed_since('2025-09-04') + target_lists.new_data()
instruments = 'hst-stis'
batch_mode = True
care_level = 1 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step

redo_extractions = False
# note that the above will redo all extractions for the target. If you want to redo specific extractions, just go
# delete the files produced by the extraction (including intermediates)


#%% set extraction parameters

# note that traces can be as low as 100 pixels if using the D1 apertures, so don't go wild with the background offsets
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



#%% function to filter out e140ms from custom extractions

def no_e140ms(files):
    return [f for f in files if 'stis-e140m' not in f.name]


#%% point to where the calibration files are

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

import crds


#%% target iteration

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


#%% change to data directory and load file list

    print(
        f"""
    {'=' * len(target)}
    {target.upper()}
    {'=' * len(target)}
    """
    )

    data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{target}/hst')
    if not data_dir.exists():
        os.makedirs(data_dir)
    os.chdir(data_dir)

    obs_tbl = obs_tbl_tools.load_obs_tbl(target)

    obs_tbl_usbl = obs_tbl[obs_tbl['usable'].filled(True)]
    print(f'\n{target} observation table:\n')
    obs_tbl.pprint(-1,-1)

    stis_tag_files_in_tbl = []
    for row in obs_tbl_usbl:
        if 'hst-stis' in row['science config']:
            stis_tag_files_in_tbl.extend(row['key science files'])
    stis_tag_files_in_dir = dbutils.find_data_files('tag', instruments=instruments)
    stis_tag_files_in_dir = list(map(str, stis_tag_files_in_dir))

    n_tbl = len(stis_tag_files_in_tbl)
    n_obs = len(stis_tag_files_in_dir)
    if n_tbl != n_obs:
        warnings.warn(f"There are {n_tbl} in the observation manifest table but {n_tbl} in the directory for {target}."
                      f"\nOnly the files in the table will be extracted.")


#%% identify existing STIS extractions

    obs_tbl_usbl['skip_extraction'] = False
    obs_tbl_usbl['skip_extraction'].description = (
        "Temporary column telling the extraction scrip whether to extract the data or not. Can likely be deleted if"
        "still present later."
    )

    for row in obs_tbl_usbl:
        if 'hst-cos' in row['science config']:
            continue
        sci_names = row['key science files']
        for name in sci_names:
            sci_file, = dbutils.find_stela_files_from_hst_filenames(name, data_dir)
            flt_file = dbutils.modify_file_label(sci_file, 'flt')
            # need to use flt not x1d bc sometimes calstis can't find trace to make x1d
            if flt_file.exists():
                if redo_extractions:
                    delete_extensions = ['flt', 'x1d', 'x2d']
                    if '_raw.fits' not in sci_file.name:
                        delete_extensions.append('raw')
                    for ext in delete_extensions:
                        del_file = dbutils.modify_file_label(sci_file, ext)
                        if del_file.exists():
                            os.remove(del_file)
                else:
                    row['skip_extraction'] = True
    obs_tbl_usbl.add_index('archive id')

    def skip(file_name):
        id = dbutils.parse_filename(file_name)['id']
        return obs_tbl_usbl.loc['archive id', id]['skip_extraction']



#%% update calibration files and perform initial extraction

    overwrite_consent = False
    for sci_tf_name in stis_tag_files_in_tbl:
        stis_tf, = dbutils.find_stela_files_from_hst_filenames(sci_tf_name, '.')
        stis_tf = str(stis_tf)

        if not redo_extractions and skip(stis_tf):
            continue

        crds.bestrefs.assign_bestrefs([stis_tf], sync_references=True, verbosity=10)

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

    utils.query_next_step(batch_mode, care_level, 2)


#%% identify trace location

    """
    As plots appear click the trace at either the Lya red wing (g140m) or at the C II line (g140l), then click off axes.
    
    You can zoom and pan, but coordinates will be registered for each click. Just make sure your last click is where you want it to be before clicking off axes.
    
    If you cannot find the trace, click at x < 100 to indicate this and the default y location will be used.
    
    (Sorry that this is cloodgy :)
    """

    matplotlib.use('Qt5Agg')
    fltfiles = dbutils.find_data_files('flt', instruments=instruments, targets=[target])
    fltfiles = no_e140ms(fltfiles)

    ids, locs = [], []
    for ff in fltfiles:
        if not redo_extractions and skip(ff):
            continue

        if 'stis-e140m' in ff.name:
            continue
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

        print('Click trace. Click at x < 100 to use default location (yello cross). Click off axes when done.')
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

        plt.autoscale()
        plt.savefig(Path('/Users/parke/Google Drive/Research/STELa/scratch/yloc prediction tests')
                    / ff.name.replace('_flt.fits', '_extraction.png'), dpi=300)

        locs.append(a2)
        h.close()

    tracelocs = dict(zip(ids, locs))

    utils.query_next_step(batch_mode, care_level, 2)


#%% extract at user-defined trace locations

    fltfiles = dbutils.find_data_files('flt', instruments=instruments, targets=[target])
    fltfiles = no_e140ms(fltfiles)
    fltfiles = [_ff for _ff in fltfiles if not skip(_ff)]

    # remove existing x1d files
    if fltfiles:

        print('Proceed with deleting and recreating x1ds associated with?')
        print('\n'.join([f.name for f in fltfiles]))
        _ = input('enter/n? ')
        if _ == '':
            for fltfile in fltfiles:
                x1dfile = dbutils.modify_file_label(fltfile, 'x1d')
                if x1dfile.exists():
                    os.remove(x1dfile)

                x1d_params = get_x1dparams(x1dfile)
                id = fits.getval(fltfile, 'asn_id').lower()
                traceloc = tracelocs[id]
                stis.x1d.x1d(str(fltfile), str(x1dfile), a2center=traceloc, **x1d_params)


    utils.query_next_step(batch_mode, care_level, 2)

#%% flag anomalous spectra, if any

    """Data could be good but look like crap, so spectra should only be flagged unusable
    if they clearly differ from the norm in a serious way, I think."""

    plt.close('all')

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

            usable_ans = input(f'Should {id_ending} be flagged unusable and files deleted (y/enter)?')
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

    utils.query_next_step(batch_mode, care_level, 2)


#%% mark files not listed as unusable and without flags as usuable

    no_flags = [np.ma.is_masked(flags) or len(flags) == 0 for flags in obs_tbl['flags'] ]
    mark_usable = no_flags & obs_tbl['usable'].mask
    obs_tbl['usable'][mark_usable] = True


#%% clean up unusable files

    tbd = dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=True, verbose=True, directory=data_dir)
    if tbd:
        answer = input('Proceed with file deletion? enter/n')
        if answer == '':
            dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=False, directory=data_dir)


#%% back to main dir

    os.chdir(paths.stage1_code)



#%% delete skip column

    obs_tbl_usbl.remove_column('skip_extraction')

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
    obs_tbl.meta['last stis extraction'] = datetime.now().isoformat()
    obs_tbl.meta['last data review'] = datetime.now().isoformat()
    obs_tbl.write(obs_tbl_tools.get_path(target), overwrite=True)

    utils.query_next_step(batch_mode, care_level, 1)


#%% plot extraction locations

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

        htmlfile = str(ff).replace('.fits', '.plot-extraction.html')
        utils.save_standard_mpld3(fig, htmlfile)


#%% close plots

    plt.close('all')


#%% loop close

  except StopIteration:
    break

