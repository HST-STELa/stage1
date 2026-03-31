import warnings
import re
from datetime import datetime
from math import nan

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import matplotlib
from matplotlib import pyplot as plt

import stistools as stis # we'll need these so import just to be sure present

import database_utilities as dbutils
import paths
import utilities as utils
import catalog_utilities as catutils
import hst_utilities as hstutils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import observation_table as obt


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

human_reviewer = 'Parke'
targets = (
    set(target_lists.data_modified_after('2026-03-05')) |
    set(target_lists.bespoke['lya archival 2026-03-11'])
)
targets = sorted(list(targets))
targets = targets[8:]
clear_flags = True
batch_mode = True
care_level = 1 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step
matplotlib.use('Qt5Agg')


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

    ax.imshow(h.data, origin='lower')
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

    return ax, coords_at_obs


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
    data_dir = paths.target_hst_data(target)

    obs_tbl = obt.ObsTable.load_from_targname(target)

    # clear any flags other than no data or shutter closed
    if clear_flags:
        backup_obs_tbl = obs_tbl.copy()
        reason_col = backup_obs_tbl['reason unusable'].filled('').astype(str)
        keep_rows_mask = (reason_col == 'No data taken.') | (reason_col == 'Shutter closed.')
        keep_rows_idx, = np.nonzero(keep_rows_mask)
        obs_tbl = obs_tbl.clear_usability_values(obs_tbl,other_columns_to_clear=['flags', 'usability status'])
        obs_tbl['usable'][keep_rows_idx] = False # gotta use idx instead of mask for bool column, weird astropy bug?
        obs_tbl['reason unusable'][keep_rows_idx] = reason_col[keep_rows_idx]
        obs_tbl['usability status'][keep_rows_idx] = 'unusable' # maybe gotta use idx here too

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



#%% acquisition checking

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
        acq_issues = False
        image_shown = False
        warning_msgs = []

        # find associated science files, print info
        def associated(sfs):
            return not np.ma.is_masked(sfs) and acq_name in list(sfs.values())
        assoc_obs_mask = [associated(sfs) for sfs in obs_tbl['supporting files']]
        acq_file, = dbutils.find_stela_files_from_hst_filenames(acq_name, data_dir)
        id_list_str = ', '.join(obs_tbl['archive id'][assoc_obs_mask])
        print(f'\nAcquistion file {acq_file.name} associated with: {id_list_str}')

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
            h = fits.open(acq_file)
            print(f'STIS {h[0].header['obsmode']}')
            warning_msgs = hstutils.auto_validate_stis_acq(acq_file, verbosity=1)
            if warning_msgs: # simplify
                warning_msgs = ['stistools.tastis logged acquisition warnings']

            # now plot the acq images
            stages = ['coarse', 'fine']
            if 'mirvis' in acq_file.name and 'PEAK' not in h[0].header['obsmode']:
                fig = plt.figure(figsize=[7,3])
                axs = []
                for j in range(2):
                    hh = h['sci', j+1]
                    ax, coords_at_obs = plot_acq_image(hh, coords, fig, (1, 2, j+1))
                    ax.set_title(stages[j])
                    if j == 0:
                        ax.scatter(coords_at_obs.ra, coords_at_obs.dec,
                                   transform=ax.get_transform('icrs'),
                                   marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
                    else:
                        ax.scatter(0.5, 0.5, transform=ax.transAxes,
                                   marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
                image_shown = True
        else:
            h = fits.open(acq_file)
            exptype = h[0].header['exptype']
            print(f'COS {exptype}')
            if exptype == 'ACQ/SEARCH':
                print('no checks performed') # these should always be followed by a more precise acq according to STScI policy
            if exptype == 'ACQ/PEAKXD':
                warning_msgs = hstutils.auto_validate_cos_acq_peakxd(h, verbosity=1)
            if exptype == 'ACQ/PEAKD':
                warning_msgs = hstutils.auto_validate_cos_acq_peakd(h, verbosity=1)
            if exptype == 'ACQ/IMAGE':
                stages = ['initial', 'confirmation']
                fig = plt.figure(figsize=[5,3])
                for j in range(2):
                    hh = h['sci', j+1]
                    ax, _ = plot_acq_image(hh, coords, fig, (1, 2, j+1),
                                        zoom_region=3*u.arcsec)
                    if j == 1:
                        ax.scatter(0.5, 0.5, transform=ax.transAxes,
                                   marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
                    ax.set_title(stages[j])
                fig.suptitle(acq_file.name)

                image_shown = True

        if warning_msgs:
            acq_issues = True
            for msg in warning_msgs:
                obs_tbl.add_flag(assoc_obs_mask, msg)

        if image_shown:
            fig.suptitle(acq_file.name)
            fig.tight_layout()
            print('Click outside the plots to continue.')
            xy = utils.click_coords(fig)
            answer = input('Did the target appear in the acquisition image (enter/n)')
            if answer != '':
                acq_issues = True
                obs_tbl.add_flag(assoc_obs_mask, f'Human reviewer ({human_reviewer}) could not identify target in acquisition image.')
            plt.close('all')

        if acq_issues:
            obs_tbl.add_flag(assoc_obs_mask, 'Acquisition untrustworthy.')
            obs_tbl['usability status'][assoc_obs_mask] = 'has issues'


#%% now look through the spectra to add flags and notes

    """Data could be good but look like crap, so spectra should only be flagged unusable
    if they clearly differ from the norm in a serious way, I think."""

    plt.close('all')

    viewcols = ['archive id', 'usable', 'usability status', 'reason unusable', 'flags', 'notes']

    catutils.set_index(obs_tbl, 'archive id')
    usbl_mask = obs_tbl['usable'].filled(True)
    usbl_tbl = obs_tbl[usbl_mask]
    configs = np.unique(usbl_tbl['science config'])
    for config in configs:
        config_mask = usbl_tbl['science config'] == config
        cnfg_tbl = usbl_tbl[config_mask]
        cnfg_tbl['flags'] = cnfg_tbl['flags'].filled('')

        spectra = []

        # mark any with all zeros or nans as unusable and don't bother plotting
        for row in cnfg_tbl:
            id = row['archive id']
            i_main = obs_tbl.loc_indices[id]
            file, = data_dir.glob(f'*{id}_x1d.fits')
            data = fits.getdata(file, 1)

            wave = np.ravel(np.asarray(data['wavelength'], dtype=float))
            flux = np.ravel(np.asarray(data['flux'], dtype=float))

            finite = np.isfinite(wave) & np.isfinite(flux)
            if not np.any(finite):
                obs_tbl['usable'][i_main] = False
                obs_tbl['usability status'][i_main] = 'unusable'
                obs_tbl['reason unusable'][i_main] = 'Spectrum is entirely NaN or non-finite.'
                obs_tbl.add_flag(i_main, 'All spectral values are NaN/non-finite.')
                continue

            wave = wave[finite]
            flux = flux[finite]

            order = np.argsort(wave)
            wave = wave[order]
            flux = flux[order]

            if np.all(flux == 0):
                obs_tbl['usable'][i_main] = False
                obs_tbl['usability status'][i_main] = 'unusable'
                obs_tbl['reason unusable'][i_main] = 'Spectrum is all zeros.'
                continue

            spectra.append(dict(
                id=id,
                file=file,
                row=row,
                wavelength=wave,
                flux=flux,
            ))

        # interpolate the remaining data onto the same wavelength grid and find median and median abs dev
        if len(spectra) == 0:
            continue

        wmins = [np.nanmin(spec['wavelength']) for spec in spectra]
        wmaxs = [np.nanmax(spec['wavelength']) for spec in spectra]
        wmin = max(wmins)
        wmax = min(wmaxs)

        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
            raise ValueError(f'No common wavelength overlap for {target} {config}')

        ngrid = int(np.median([len(spec['wavelength']) for spec in spectra]))
        ngrid = max(ngrid, 200)
        wave_grid = np.linspace(wmin, wmax, ngrid)

        interp_fluxes = []
        for spec in spectra:
            interp_flux = np.interp(
                wave_grid,
                spec['wavelength'],
                spec['flux'],
                left=np.nan,
                right=np.nan,
            )
            spec['interp_flux'] = interp_flux
            interp_fluxes.append(interp_flux)

        interp_fluxes = np.asarray(interp_fluxes)
        median_flux = np.nanmedian(interp_fluxes, axis=0)
        mad_flux = np.nanmedian(np.abs(interp_fluxes - median_flux), axis=0)

        # plot the spectra together in batches of no more than 5 on top of a thick background line
        # showing median and a light shaded region showing median abs dev
        figs = []
        m = 5
        for start in range(0, len(spectra), m):
            batch = spectra[start:start+m]
            fig, ax = plt.subplots(figsize=(10, 5))
            figs.append(fig)

            ax.fill_between(
                wave_grid,
                median_flux - mad_flux,
                median_flux + mad_flux,
                color='k',
                alpha=0.2,
                zorder=1,
                label = '_'
            )
            ax.plot(
                wave_grid,
                median_flux,
                lw=2,
                color='k',
                alpha=0.7,
                zorder=2,
                label='median',
            )

            for spec in batch:
                ax.step(
                    spec['wavelength'],
                    spec['flux'],
                    where='mid',
                    zorder=3,
                    label=str(spec['id'])[-6:],
                )

            if 'e140m' in config:
                lyamask = (wave_grid > 1214) & (wave_grid < 1217)
                ylo = 2*np.min(median_flux[lyamask])
                yhi = 2*np.max(median_flux[lyamask])
                ax.set_ylim(ylo, yhi)

            title_ids = ', '.join([str(spec['id'])[-6:] for spec in batch])
            print(f'Fig {fig.number}: {title_ids}')
            ax.set_title(f'{target} | {config} | {title_ids}')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Flux')
            ax.legend(fontsize='small', ncol=2)

            notes = []
            for spec in batch:
                flags = spec['row']['flags']
                if np.ma.is_masked(flags) or flags == '':
                    continue
                if isinstance(flags, str):
                    flag_text = flags
                else:
                    try:
                        flag_text = ', '.join(map(str, flags))
                    except TypeError:
                        flag_text = str(flags)
                notes.append(f"{str(spec['id'])[-6:]}: {flag_text}")

            if len(notes) > 0:
                ax.annotate(
                    '\n'.join(notes),
                    xy=(0.02, 0.98),
                    xycoords='axes fraction',
                    va='top',
                    fontsize='small',
                )

            fig.tight_layout()

        print('Click outside the plots to continue.')
        xy = utils.click_coords(figs[-1])

        while True:
            id_endings = input('Any spectra you want to mark unusable, add flags, or add notes?\n'
                               'Give last few letters of the ids, separated by commas.\n'
                               'Hit enter if none. Prompt will loop until an empty answer is given.')
            if id_endings == '':
                break
            id_endings = re.split(', *', id_endings)
            mask = np.zeros(len(obs_tbl), bool)
            ids_all_good = True
            for id_ending in id_endings:
                matches = np.char.endswith(obs_tbl['archive id'].astype(str), id_ending)
                if sum(matches) == 0:
                    ids_all_good = False
                    print(f'No matches for {id_ending}. Retry.')
                    break
                elif sum(matches) > 1:
                    ids_all_good = False
                    print(f'Multiple matches for {id_ending}. Retry.')
                    break
                mask |= np.char.endswith(obs_tbl['archive id'].astype(str), id_ending)
            if not ids_all_good:
                continue

            i_mask, = np.nonzero(mask)

            obs_tbl[viewcols][i_mask].pprint(-1,-1)
            while True:
                usable_ans = input(f'Update usability? enter=no change, u=unusable, i=has issues, a=all clear')
                if usable_ans == '':
                    break
                elif usable_ans.startswith('a'):
                    obs_tbl['usability status'][i_mask] = 'all clear'
                    obs_tbl['usable'][i_mask] = True
                    break
                elif usable_ans.startswith('u'):
                    obs_tbl['usable'][i_mask] = False
                    obs_tbl['usability status'][i_mask] = 'unusable'
                    reason = input(f'Enter reason unusable.')
                    obs_tbl['reason unusable'][i_mask] = reason
                    break
                elif usable_ans.startswith('i'):
                    obs_tbl['usable'].mask[i_mask] = True
                    obs_tbl['usability status'][i_mask] = 'has issues'
                    break
                else:
                    print('Bad input.')

            while True:
                flagstring = input(f'Enter flags to be added, separated by commas (enter if none):')
                if len(flagstring) < 3:
                    break
                else:
                    obs_tbl.add_flags(i_mask, flagstring)

            while True:
                notestring = input(f'Enter notes to be added, separated by commas (enter if none):')
                if len(notestring) < 3:
                    break
                else:
                    obs_tbl.add_notes(i_mask, notestring)

            print('Tbl updated to:')
            obs_tbl[viewcols][i_mask].pprint(-1, -1)

        plt.close('all')

    utils.query_next_step(batch_mode, care_level, 2)


#%% mark files not listed as unusable and without flags as usable

    no_flags = [np.ma.is_masked(flags) or len(flags) == 0 for flags in obs_tbl['flags'] ]
    status = obs_tbl['usability status'].filled('')
    no_issues = (status == 'all clear') | (status == 'unchecked') | (status == '')
    mark_usable = no_flags & no_issues & obs_tbl['usable'].filled(True)
    obs_tbl['usable'][mark_usable] = True
    obs_tbl['usability status'][mark_usable] = 'all clear'


#%% clean nulls and duplicates

    for name in ['flags', 'notes']:
        obs_tbl.clean_nulls_col_of_lists(name)
        obs_tbl.clean_duplicates_col_of_lists(name)

    if any(obs_tbl['usability status'].mask):
        raise ValueError('Some rows still have masked usability status. This should be filled for all.')


#%% take a gander

    print(f'\n{target} observation table after checks:\n')
    obs_tbl.pprint(-1,-1)


#%% save obs_tbl

    print(f'\nSaving obs_tbl for {target}.\n')
    obs_tbl.sort('start')
    obs_tbl.meta['last data review'] = datetime.now().isoformat()
    obs_tbl.meta['last review by'] = human_reviewer
    obs_tbl.write(obs_tbl.get_path(target), overwrite=True)

    utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    break