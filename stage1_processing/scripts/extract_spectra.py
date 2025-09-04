import os
import warnings
from pathlib import Path
import glob
from copy import copy
import re
from datetime import datetime

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.ion()
from astropy.io import fits
from astropy import table
import numpy as np

import stistools as stis

import database_utilities as dbutils
import utilities as utils
import paths

from stage1_processing import target_lists
from stage1_processing import observation_table as obs_tbl_tools
from data_reduction_tools import stis_extraction as stx


#%% settings (incl. batch mode)

batch_mode = True
care_level = 2 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step

redo_extractions = False
# note that the above will redo all extractions for the target. If you want to redo specific extractions, just go
# delete the files produced by the extraction (including intermediates)


#%% define what you want to extract

targets = target_lists.observed_since('2025-07-14')
instruments = 'hst-stis'


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


#%% SKIP? set up batch processing (skip if not in batch mode)

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
    print(f'\n{target} observation table:\n')
    obs_tbl.pprint(-1,-1)

    stis_tag_files_in_tbl = []
    for row in obs_tbl:
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

    obs_tbl['skip_extraction'] = False
    obs_tbl['skip_extraction'].description = (
        "Temporary column telling the extraction scrip whether to extract the data or not. Can likely be deleted if"
        "still present later."
    )

    for row in obs_tbl:
        sci_names = row['key science files']
        for name in sci_names:
            sci_file, = dbutils.find_stela_files_from_hst_filenames(name, data_dir)
            x1d_file = dbutils.modify_file_label(sci_file, 'x1d')
            if x1d_file.exists():
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

#%% update calibration files and perform initial extraction

    overwrite_consent = False
    for row in obs_tbl:
        sci_tf_name, = row['key science files']
        stis_tf, = dbutils.find_stela_files_from_hst_filenames(sci_tf_name, '.')
        stis_tf = str(stis_tf)

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

    obs_tbl['usable'][obs_tbl['usable'].mask] = True

    utils.query_next_step(batch_mode, care_level, 2)


#%% clean up unusable files

    tbd = dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=True, verbose=True, directory=data_dir)
    if tbd:
        answer = input('Proceed with file deletion? enter/n')
        if answer == '':
            dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=False, directory=data_dir)


#%% if only one STIS G140M exposure, extract background traces

    g140mfiles = dbutils.find_data_files('flt', instruments='hst-stis-g140m', targets=[target])

    if len(g140mfiles) == 1:
        # remove existing files
        labels = "x1dbk1 x1dtrace x1dbk2".split()
        print(f'Proceed with deleting and recreating {labels[0]}, {labels[1]}, and  {labels[2]} associated with?')
        print('\n'.join([f.name for f in g140mfiles]))
        _ = input('enter/n? ')
        if _ == '':
            for fltfile in g140mfiles:
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

    utils.query_next_step(batch_mode, care_level, 2)


#%% revise uncertainties

    #todo utils.shift_floor_to_zero(spec.e, window_size=50) helpful


#%% coadd multiple exposures or orders

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
        date = f'{min(pieces_tbl['datetime'])}--{max(pieces_tbl['datetime'])}'
        program = 'pgm' + '+'.join(np.unique(np.char.replace(pieces_tbl['program'], 'pgm', '')))
        id = f'{len(x1dfiles)}exposure'
        coadd_name = f'{target}.{config}.{date}.{program}.{id}_coadd.fits'
        os.rename(main_coadd_file, f'{coadd_name}')

        data = fits.getdata(coadd_name, 1)
        plt.figure()
        plt.title(coadd_name)
        plt.step(data['wavelength'].T, data['flux'].T, where='mid')

    utils.query_next_step(batch_mode, care_level, 2)


#%% back to main dir

    os.chdir(paths.stage1_code)



#%% delete skip column

    obs_tbl.remove_column('skip_extraction')

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

