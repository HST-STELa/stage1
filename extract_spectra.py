#%% imports

try:
    import stistools as stis
except ImportError:
    print("Couldn't import stistools. You probably need to do `conda activate stenv' before starting python.\n"
          "Or, if using PyCharm, select the envrionment using the selector near the bottom right.\n"
          "Note, you're stistools environment might be named something other than stenv, or you might need to"
          "install one. See https://stistools.readthedocs.io/en/latest/gettingstarted.html")

import os
from pathlib import Path
import glob
from copy import copy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.ion()
from astropy.io import fits
import numpy as np

import database_utilities as dbutils
import utilities as utils
from data_reduction_tools import stis_extraction as stx
import paths


#%% set the targets you want to extract, locate tag files
# you will need to be in the directory for the files

os.chdir(paths.data)
# targets = ['hd17156', 'k2-9', 'toi-1434', 'toi-1696', 'wolf503', 'hd207496']
# targets = ['toi-1696']
targets = ('hd207496 hd5278 toi-1224 toi-1434 toi-2015 toi-2079 toi-2134 toi-2194 toi-2285 '
           'toi-4438 toi-4576 toi-6078 wolf503').split()
# targets = 'any'

obs_filters = dict(targets=targets)

tagfiles = dbutils.find_data_files('tag', **obs_filters)
print("Spectra to be extracted from:")
print('\n'.join([tf.name for tf in tagfiles]))


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


#%% update the calibration files

import crds
file_strings = [str(f) for f in tagfiles]
crds.bestrefs.assign_bestrefs(file_strings, sync_references=True, verbosity=10)


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


#%% run an initial extraction
pass

# check for 0-length exposures
files_with_no_data = []
for tagfile in tagfiles:
    gtis = fits.getdata(tagfile, extname='GTI')
    if len(gtis['start']) == 0:
        files_with_no_data.append(tagfile)
if files_with_no_data:
    raise ValueError(f"These files have no data:\n{'\n'.join(files_with_no_data)}")

overwrite_consent = False
for tagfile in tagfiles:
    tagfile = str(tagfile)
    rawfile = tagfile.replace('_tag', '_raw')
    asn_id = fits.getval(tagfile, 'asn_id').lower()
    wavfile, = glob.glob(f'*{asn_id}_wav.fits')
    # exposure
    stis.inttag.inttag(tagfile, rawfile)
    root = dbutils.modify_file_label(tagfile, '')
    root = str(root).replace('_', '')
    status = stis.calstis.calstis(rawfile, wavecal=wavfile, outroot=root) # exposure
    assert status == 0


#%% locate traces

fltfiles = dbutils.find_data_files('flt', instruments='hst-stis',  **obs_filters)

print(f'As plots appear click the trace at either the Lya red wing (g140m) or at the C II line (g140l), then click off axes.\n'
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
    
    y_predicted = stx.predicted_trace_location(h)
    pln, = plt.plot(512, y_predicted, 'y+', ms=10, label='predicted location')

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

    locs.append(a2)
    h.close()

tracelocs = dict(zip(ids,locs))


#%% now rerun the extraction at the appropriate locations

fltfiles = dbutils.find_data_files('flt', instruments='hst-stis', **obs_filters)

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


#%% extract traces without background

# fltfiles = dbutils.find_data_files('flt', instruments='hst-stis', **obs_filters)
fltfiles = dbutils.find_data_files('flt', instruments='hst-stis', targets=['hd5278'])

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




