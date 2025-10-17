import warnings
from pathlib import Path
from datetime import datetime
from math import nan

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
from astropy.time import Time

import matplotlib
from matplotlib import pyplot as plt

import stistools as stis

import database_utilities as dbutils
import utilities as utils
import catalog_utilities as catutils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import observation_table as obs_tbl_tools


#%% batch mode or single runs?

batch_mode = True
care_level = 1 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step

matplotlib.use('Qt5Agg')
plt.ion()


#%% get targets

# targets = target_lists.observed_since('2025-07-14')
# targets = target_lists.eval_no(2)
# targets = ['hd63935', 'hd73583', 'toi-1898'] # external data to check from sept review
# targets = ['lp714-47']
targets = ['k2-25', 'wasp-80', 'wasp-29', 'hd219134', 'kepler-444', 'gliese12', 'gj1132', '55cnc']
targets = ['gj1214']


#%% rechecking flagged aquisitions
"""if you want to check aquisitions of a target that has already been flagged unusable, use the
database_utilities.clear_usability_values function to reset some of the table rows"""


#%% properties table

with catutils.catch_QTable_unit_warnings():
    targprops = preloads.hosts.copy()
targprops.add_index('tic_id')


#%% ra, dec plotting

def plot_acq_image(fits_handle, object_coords, figure, subplot_spec, zoom_region=None):
    h = fits_handle

    newobstime = Time(h.header['expstart'], format='mjd')
    coords_at_obs = object_coords.apply_space_motion(newobstime)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="'datfix'")
        wcs = WCS(h.header)
    ax = figure.add_subplot(*subplot_spec, projection=wcs)

    ax.imshow(hh.data, origin='lower')
    ax.scatter(coords_at_obs.ra, coords_at_obs.dec,
               transform=ax.get_transform('icrs'),
               marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
    ax.coords.grid(True, color='white', ls=':', lw=0.5)

    # ax.coords[0].set_ticklabel_visible(False)  # RA
    # ax.coords[1].set_ticklabel_visible(False)  # Dec
    # ax.coords[0].set_axislabel('')  # RA label
    # ax.coords[1].set_axislabel('')  # Dec label

    if zoom_region is not None:
        ra, dec = coords_at_obs.ra, coords_at_obs.dec
        coord1 = SkyCoord(ra - zoom_region, dec - zoom_region)
        coord2 = SkyCoord(ra + zoom_region, dec + zoom_region)

        # Convert to pixel coordinates
        (x1, y1) = wcs.world_to_pixel(coord1)
        (x2, y2) = wcs.world_to_pixel(coord2)

        # avoid reversed pixel coords
        xlo = min(x1, x2)
        xhi = max(x1, x2)
        ylo = min(y1, y2)
        yhi = max(y1, y2)

        # avoid skinny images
        dx = xhi - xlo
        dy = yhi - ylo
        dmx = max(dx, dy)
        if dx < dmx/2:
            xlo -= dmx/2
            xhi += dmx/2
        if dy < dmx/2:
            ylo -= dmx/2
            yhi += dmx/2

        # Set the limits using pixel coordinates
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    return ax


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
      {'=' * len(target)}
      {target.upper()}
      {'=' * len(target)}
      """
    )

    tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
    data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{target}/hst')

    obs_tbl = obs_tbl_tools.load_obs_tbl(target)
    print(f'\n{target} observation table:\n')
    obs_tbl.pprint(-1,-1)

#%% target coordinates

    """note the exoplanet catalog gives coordinates from gaia dr2, which uses a 2015.5 epoch
    I figured this out just by comparing coordinates for hd95338"""
    props = targprops.loc[tic_id]
    coords = SkyCoord(
        props['ra'].filled(nan),
        props['dec'].filled(nan),
        pm_ra_cosdec=props['sy_pmra'].filled(nan),
        pm_dec=props['sy_pmdec'].filled(nan),
        distance=props['sy_dist'].filled(nan),
        obstime=Time(2015.5, format='jyear')
    )



#%% actual checking

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

        # find associated science files, print info
        def associated(sfs):
            return not np.ma.is_masked(sfs) and acq_name in list(sfs.values())
        assoc_obs_mask = [associated(sfs) for sfs in obs_tbl['supporting files']]
        acq_file, = dbutils.find_stela_files_from_hst_filenames(acq_name, data_dir)
        print(f'\n\nAcquistion file {acq_file.name} associated with:')
        obs_tbl[assoc_obs_mask]['start,science config,program,key science files'.split(',')].pprint(-1,-1)
        print('\n\n')

        # check if the acquisition is an offset acquisition
        sci_name = obs_tbl[assoc_obs_mask]['key science files'][0][0]
        sci_file, = dbutils.find_stela_files_from_hst_filenames(sci_name, data_dir)
        targname_acq = fits.getval(acq_file, 'targname')
        targname_sci = fits.getval(sci_file, 'targname')
        if targname_sci != targname_acq:
            msg = ("Target names in the FITS headers for the science file and acq data differ. "
                   "This is likely an offset acquisition."
                   f"\n\tscience target:     {targname_sci}"
                   f"\n\tacquisition target: {targname_acq}")
            warnings.warn(msg)

        if 'hst-stis' in acq_file.name:
            # run builtin STIS tool for acq diagnosis
            stis.tastis.tastis(str(acq_file))

            # now plot the acq images
            stages = ['coarse', 'fine']
            h = fits.open(acq_file)
            if 'mirvis' in acq_file.name and 'PEAK' not in h[0].header['obsmode']:
                fig = plt.figure(figsize=[7,3])
                for j in range(2):
                    hh = h['sci', j+1]
                    ax = plot_acq_image(hh, coords, fig, (1, 2, j+1))
                    ax.set_title(stages[j])
                fig.suptitle(acq_file.name)
                fig.tight_layout()

                print('Click outside the plots to continue.')
                xy = utils.click_coords(fig)
        else:
            print('\nCOS data, no automatic eval routine\n')
            h = fits.open(acq_file)
            plate_scale_d = 0.023 # roughly correct for all gratings
            plate_scale_xd_dic = dict(G130M=100/1000, G160M=90/1000, G140L=90/1000, G230L=24/1000,
                                      MIRRORA=23.5/1000, MIRRORB=23.5/1000)
            plate_scale_xd = plate_scale_xd_dic[h[0].header['opt_elem']]
            if h[0].header['exptype'] == 'ACQ/SEARCH':
                continue # these should always be followed by a more precise acq according to STScI policy
            if h[0].header['exptype'] == 'ACQ/PEAKXD':
                print('PEAKXD acq')
                print(f'\tcounts: {h[1].data['counts']}')
                centroid_offset = (h[0].header['acqmeasy'] - h[0].header['acqprefy']) * plate_scale_xd
                print(f'\tcentroid offset: {centroid_offset} arcsec')
                print(f'\tslew: {h[0].header['ACQSLEWY']} arcsec')
                print(f'\tcounts should be > 0 and the two bottom values should be close for a good acquisition')
            if h[0].header['exptype'] == 'ACQ/PEAKD':
                print('PEAKD acq')
                print(f'\tcounts: {h[1].data['counts']}')
                print(f'\tdisp offsets: {h[1].data['DISP_OFFSET']*plate_scale_d} arcsec')
                print(f'\tslew: {-h[0].header['ACQSLEWX']/2} arcsec') # the factor of 2 is a kludge, values seem consistently off by that amount
                print(f'\tcounts should be > 0 and slew should be an mount that moves scope from final dwell point'
                      f'\n\tto the point where counts peak')
            if h[0].header['exptype'] == 'ACQ/IMAGE':
                stages = ['initial', 'confirmation']
                fig = plt.figure(figsize=[5,3])
                for j in range(2):
                    hh = h['sci', j+1]
                    ax = plot_acq_image(hh, coords, fig, (1, 2, j+1),
                                        zoom_region=3*u.arcsec)
                    ax.set_title(stages[j])
                fig.suptitle(acq_file.name)
                fig.tight_layout()

                print('Click outside the plots to continue.')
                xy = utils.click_coords(fig)

        answer = input('Mark acq as bad? (enter for no, b for bad)')
        if answer in ['b', 'y']:
            bad_acq = True
        plt.close('all')

        if bad_acq:
            obs_tbl['usable'][assoc_obs_mask] = False
            obs_tbl['reason unusable'][assoc_obs_mask] = 'Target not acquired or other acquisition issue.'


#%% delete files associated with bad obs

    if np.any(~obs_tbl['usable'].filled(True)):
        dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=True, verbose=True, directory=data_dir)
        answer = input('Proceed with file deletion? y/n')
        if answer == 'y':
            dbutils.delete_files_for_unusable_observations(obs_tbl, dry_run=False, directory=data_dir)


#%% take a gander

    print(f'\n{target} observation table after checks:\n')
    obs_tbl.pprint(-1,-1)


#%% save obs_tbl

    print(f'\nSaving obs_tbl for {target}.\n')
    obs_tbl.sort('start')
    obs_tbl.meta['last acq check'] = datetime.now().isoformat()
    obs_tbl.write(obs_tbl_tools.get_path(target), overwrite=True)

    utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    break