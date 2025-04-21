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

from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

import database_utilities as dbutils
import utilities as utils


#%% set the targets you want to extract, locate tag files
# you will need to be in the directory for the files

os.chdir('/Users/parke/Google Drive/Research/STELa/data/uv_observations/hst-stis')
targets = ['hd17156', 'k2-9', 'toi-1434', 'toi-1696', 'wolf503', 'hd207496']

tagfiles = dbutils.find_data_files('tag', targets)
print("Spectra to be extracted from:")
print('\n'.join(tagfiles))


#%% point to where hte calibration files are

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

# note that the G140L traces get as low as 100 and G140M as low as 150, so I can't go hog wild with the backgroun regions
# x1d_params = dict()
def get_x1dparams(file):
    min_params = dict(maxsrch=0.01,
                      bksmode='off')  # no background smoothing
    if 'g140m' in file.name:
        extrsize = 19
        bkoffst = 30
        bksize = 20
    elif 'g140l' in file.name:
        extrsize = 13
        bkoffst = 30
        bksize = 20
    else:
        raise ValueError
    assert (bkoffst - bksize / 2) > (extrsize / 2) + 5
    newparams = dict(bk1offst=-bkoffst, bk2offst=bkoffst, bk1size=bksize, bk2size=bksize)
    allparams = {**min_params, **newparams}
    return allparams


#%% run an initial extraction

overwrite_consent = False
for tagfile in tagfiles:
    tagfile = str(tagfile)
    rawfile = tagfile.replace('_tag', '_raw')
    asn_id = fits.getval(tagfile, 'asn_id').lower()
    wavfile, = glob.glob(f'*{asn_id}_wav.fits')
    # exposure
    stis.inttag.inttag(tagfile, rawfile)
    root = dbutils.modify_file_label(tagfile, '')
    status = stis.calstis.calstis(rawfile, wavecal=wavfile, outroot=root) # exposure
    assert status == 0


#%% locate traces

ydefault = 387 # extraction location if no trace visible

fltfiles = dbutils.find_data_files('flt', targets)
print(f'As plots appear click the trace at either the Lya red wing (g140m) or at the C II line (g140l), then click off axes.\n'
      '\n'
      'You can zoom and pan, but coordinates will be registered for each click. '
      'Just make sure your last click is where you want it to be before clikcing off axes.\n'
      '\n'
      'If you cannot find the trace, click at x < 100 to indicate this and the default y location'
      'will be used.\n'
      '\n'
      '(Sorry that this is cloodgy :)')

ids, locs = [], []
for ff in fltfiles:
    h = fits.open(ff)
    id = h[0].header['asn_id'].lower()
    ids.append(id)

    grating = h[0].header['opt_elem'].lower()
    if grating not in ['g140l', 'g140m']:
        raise NotImplementedError

    img = h[1].data
    plt.figure()
    plt.title(Path(ff).name)
    plt.imshow(np.cbrt(img), aspect='auto')

    xy = utils.click_coords()
    xloc, yloc = xy[-1]
    if xloc < 100:
        yloc = ydefault
        plt.annotate('default used', xy=(0.05, 0.95), xycoords='axes fraction', color='r', va='top')

    plt.axhline(yloc, color='r', alpha=0.5)

    # adjust to the center of the flt based on how the pipeline tilts the trace
    if grating == 'g140m':
        yloc += 1.35
    if grating == 'g140l':
        yloc += 0.76

    locs.append(yloc)
    h.close()

tracelocs = dict(zip(ids,locs))


#%% now rerun the extraction at the appropriate locations

fltfiles = dbutils.find_data_files('flt', targets)

# remove existing x1d files
print('Proceed with deleting and recreating x1ds associated with?')
print('\n'.join([f.name for f in fltfiles]))
_ = input('y/n? ')
if _ == 'y':
    for fltfile in fltfiles:
        x1dfile = dbutils.modify_file_label(fltfile, 'x1d')
        if x1dfile.exists():
            os.remove(x1dfile)

        id = fits.getval(fltfile, 'asn_id').lower()
        traceloc = tracelocs[id]
        x1d_params = get_x1dparams(x1dfile)
        stis.x1d.x1d(str(fltfile), str(x1dfile), a2center=traceloc, **x1d_params)

