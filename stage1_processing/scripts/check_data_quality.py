import warnings
import re
from datetime import datetime
from math import nan
import sys
import io

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import matplotlib
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.colors import qualitative

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

human_reviewer = 'parke'
human_review_acq = 'issues' # use one of yes, no, issues
human_review_spec = 'no' # use one of yes, no, issues

targets = target_lists.last_data_review_before('2026-04-01')
clear_usability = True
clear_other = ['flags', 'usability status', 'notes']
preserve_notes = (
    'no acquisition found',
    'identified target in acquisition image',
    'deemed target absent in acquisition image',
)
batch_mode = True
care_level = 1 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step

acq_target_flux_n_chunks = 7
acq_target_flux_sigma_threshold = 5.0

min_spectra_for_band_comparison = 3
anomalous_flux_sigma_threshold = 3.0
zero_flux_sigma_threshold = 1.0


#%% show plots or not

if human_review_acq in ['yes', 'issues'] or human_review_spec in ['yes', 'issues']:
    matplotlib.use('qt5agg') # show
else:
    matplotlib.use('agg') # don't show


#%% properties table

with catutils.catch_QTable_unit_warnings():
    targprops = preloads.hosts.copy()
targprops.add_index('tic_id')


#%% setup to capture printouts

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def start_capturing_printouts():
    global _old_stdout, _buffer
    buffer = _buffer =  io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = Tee(_old_stdout, _buffer)
    return buffer

def stop_and_save_capture(filepath):
    global _old_stdout, _buffer
    sys.stdout = _old_stdout
    txt = _buffer.getvalue()
    with open(filepath, 'w') as f:
        f.write(txt)


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
    diagnostics_dir = data_dir / 'diagnostics'
    diagnostics_dir.mkdir(exist_ok=True)
    acq_img_dir = diagnostics_dir / 'acquisition images'
    acq_img_dir.mkdir(exist_ok=True)
    spec_plt_dir = diagnostics_dir / 'spectra vs median plots'
    spec_plt_dir.mkdir(exist_ok=True)

    obs_tbl = obt.ObsTable.load_from_targname(target)

    if clear_usability:
        preserved_notes_per_row = [[] for _ in range(len(obs_tbl))]
        for i in range(len(obs_tbl)):
            for note in obt.ObsTable._iter_nonnull_cell_items(obs_tbl['notes'][i]):
                text = note if isinstance(note, str) else str(note)
                if any(sub in text for sub in preserve_notes):
                    preserved_notes_per_row[i].append(text)
        obs_tbl = obs_tbl.clear_usability_values(other_columns_to_clear=clear_other)
        n_preserved = sum(len(notes) for notes in preserved_notes_per_row)
        if n_preserved:
            print(f'Restored {n_preserved} preserved note(s) for {target} after usability clear.')
        for i, notes in enumerate(preserved_notes_per_row):
            for note in notes:
                obs_tbl.add_note(i, note)

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


#%% identify files missing data

    buffer = start_capturing_printouts()

    print(f'\nChecking for empty science files for {target}.')
    for i, row in enumerate(obs_tbl):
        if not row.usable(True):
            continue
        shortnames = row['key science files'][:]  # copy so the table row is not aliased
        scifiles = dbutils.find_stela_files_from_hst_filenames(shortnames, data_dir)
        assessment = hstutils.assess_key_science_files_data_quality(scifiles)
        if assessment.odd_expflag is not None:
            raise NotImplementedError(f'Odd exposure flag value of {assessment.odd_expflag} for {shortnames[0]}.')
        if assessment.check_zero_exptime_repair:
            repaired = hstutils.repair_zero_exptime_from_photon_times(scifiles, shortnames)
            if repaired:
                print(f'Clock rollover found for {row['archive id']}.')
                obs_tbl.add_note(i, obt.notes_menu['clock rollover'], verbose=True)
        if assessment.reject:
            print(f'Marking {row['archive id']} unusable for reason: {assessment.reason}')
            obs_tbl.update_usability(i, 'unusable', assessment.reason)

    path_basic_printout = diagnostics_dir / f'{target}.basic-data-review-printout.txt'
    stop_and_save_capture(path_basic_printout)

    care_level = utils.query_next_step(batch_mode, care_level, 2)


#%% verify that acq files are present for every observation

    known_missing = obs_tbl.substring_match_mask('notes', obt.notes_menu['acq not found'][:30])
    for row in obs_tbl:
        if not row.usable(True) or known_missing[row.index]:
            continue
        sfs = row['supporting files']
        config = row['science config']
        if obs_tbl._is_null_like(sfs):
            raise ValueError
        sfkeys = list(sfs.keys())
        if 'cos' in config:
            if not ('acq/image' in sfkeys or 'acq/peakxd' in sfkeys):
                raise ValueError
        elif 'stis' in config:
            if not 'acq' in sfkeys:
                raise ValueError


#%% acquisition checking

    buffer = start_capturing_printouts()

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
        test_image = None
        image_shown = False
        msgs = []

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
            hdr = fits.getheader(acq_file)
            zero_motion = 0*u.deg/u.yr
            acq_coords = SkyCoord(hdr['ra_targ']*u.deg, hdr['dec_targ']*u.deg,
                                  pm_ra_cosdec=zero_motion, pm_dec=zero_motion,
                                  obstime=Time('2000-01-01'))
            msg = ("Target names in the FITS headers for the science file and acq data differ. "
                   "This is likely an offset acquisition."
                   f"\n\tscience target:     {targname_sci}"
                   f"\n\tacquisition target: {targname_acq}")
            warnings.warn(msg)
        else:
            acq_coords = coords


        if 'hst-stis' in acq_file.name:
            # run builtin STIS tool for acq diagnosis
            h = fits.open(acq_file)
            print(f"STIS {h[0].header['obsmode']}")
            acq_issues, msgs, full_output = hstutils.auto_validate_stis_acq(acq_file, verbosity=1, return_full_output=True)
            print(full_output, file=buffer)

            # now assess the acq images
            stages = ['coarse', 'fine']
            if 'mirvis' in acq_file.name and 'PEAK' not in h[0].header['obsmode']:
                fig = plt.figure(figsize=[7,3])
                axs = []
                for j in range(2):
                    hh = h['sci', j+1]
                    ax, coords_at_obs, ary = hstutils.plot_acq_image(hh, acq_coords, fig, (1, 2, j + 1))
                    ax.set_title(stages[j])
                    if j == 0:
                        ax.scatter(coords_at_obs.ra, coords_at_obs.dec,
                                   transform=ax.get_transform('icrs'),
                                   marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
                    else:
                        test_image = ary
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
                acq_issues, msgs = hstutils.auto_validate_cos_acq_peakxd(h, verbosity=1)
            if exptype == 'ACQ/PEAKD':
                acq_issues, msgs = hstutils.auto_validate_cos_acq_peakd(h, verbosity=1)
            if exptype == 'ACQ/IMAGE':
                stages = ['initial', 'confirmation']
                fig = plt.figure(figsize=[5,3])
                for j in range(2):
                    hh = h['sci', j+1]
                    ax, _, ary = hstutils.plot_acq_image(
                        hh, acq_coords, fig, (1, 2, j + 1), zoom_region=3*u.arcsec)
                    if j == 1:
                        test_image = ary
                        ax.scatter(0.5, 0.5, transform=ax.transAxes,
                                   marker='+', linewidth=0.5, s=500, color='r', alpha=0.5)
                    ax.set_title(stages[j])
                fig.suptitle(acq_file.name)

                image_shown = True

        if test_image is not None:
            issueflag, flux_msgs = hstutils.acq_image_eval(
                test_image, acq_target_flux_n_chunks, acq_target_flux_sigma_threshold
            )
            print('\n'.join(flux_msgs))
            print(f'Flux test {'failed' if issueflag else 'passed'}')
            msgs.extend(flux_msgs)
            if issueflag:
                acq_issues = True

        if image_shown:
            fig.suptitle(acq_file.name)
            fig.tight_layout()

            acq_fig_path = acq_img_dir / acq_file.name.replace('.fits', '.png')
            fig.savefig(acq_fig_path, dpi=300)

            if human_review_acq == 'yes' or (human_review_acq == 'issues' and acq_issues):
                print('Click outside the plots to continue.')
                xy = utils.click_coords(fig)
                answer = input('Did the target appear in the acquisition image (enter/n)')
                if answer == '':
                    msgs.append(obt.notes_menu['can see target in acq'].format(user=human_reviewer))
                else:
                    acq_issues = True
                    msgs.append(obt.notes_menu['cannot see target in acq'].format(user=human_reviewer))

            plt.close('all')

        if msgs:
            obs_tbl.add_notes(assoc_obs_mask, msgs, separator='list', verbose=True)

        if acq_issues:
            obs_tbl.update_usability(assoc_obs_mask, 'has issues')
            obs_tbl.add_flag(assoc_obs_mask, obt.flag_menu['bad acq'], verbose=True)

        print()
        print()

    path_acq_printout = diagnostics_dir / f'{target}.acquistion-review-printout.txt'
    stop_and_save_capture(path_acq_printout)


#%% now look through the spectra to add flags and notes

    """Data could be good but look like crap, so spectra should only be flagged unusable
    if they clearly differ from the norm in a serious way, I think."""

    plt.close('all')

    viewcols = ['archive id', 'usable', 'usability status', 'reason unusable', 'flags', 'notes']

    catutils.set_index(obs_tbl, 'archive id')
    acq_issues_mask = obs_tbl.substring_match_mask('flags', 'acquisition untrustworthy')
    usbl_mask = obs_tbl['usable'].filled(True)
    usbl_tbl = obs_tbl[usbl_mask]
    configs = np.unique(usbl_tbl['science config'])
    for config in configs:
        buffer = start_capturing_printouts()

        config_mask = usbl_tbl['science config'] == config
        cnfg_tbl = usbl_tbl[config_mask]
        cnfg_tbl['flags'] = cnfg_tbl['flags'].filled('')

        spectra = []

        # mark any with all zeros or nans as unusable and don't bother plotting
        print('Checking for all zero or all non finite data.')
        for row in cnfg_tbl:
            id = row['archive id']
            i_main = obs_tbl.loc_indices[id]
            file, = data_dir.glob(f'*{id}_x1d.fits')
            data = fits.getdata(file, 1)

            wave = np.ravel(np.asarray(data['wavelength'], dtype=float))
            flux = np.ravel(np.asarray(data['flux'], dtype=float))

            finite = np.isfinite(wave) & np.isfinite(flux)
            if not np.any(finite):
                print(f'{id}')
                obs_tbl.update_usability(i_main, 'unusable', obt.reasons_menu['nans'])
                obs_tbl.add_flag(i_main, obt.flag_menu['nans'], verbose=True)
                continue

            wave = wave[finite]
            flux = flux[finite]

            order = np.argsort(wave)
            wave = wave[order]
            flux = flux[order]

            if np.all(flux == 0):
                print(f'{id}')
                obs_tbl.update_usability(i_main, 'unusable', obt.reasons_menu['zeros'])
                obs_tbl.add_flag(i_main, obt.flag_menu['zeros'], verbose=True)
                continue

            spectra.append(dict(
                id=id,
                file=file,
                row=row,
                wavelength=wave,
                flux=flux,
                wave_by_order=data['wavelength'],
                flux_by_order=data['flux'],
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

        spec_issues = False
        if len(spectra) >= min_spectra_for_band_comparison:
            band_pick = hstutils.select_spectral_comparison_band(wave_grid, config)
            if band_pick is None:
                raise NotImplementedError(
                    f'\nSpectral band integration: no priority band had enough pixels '
                    f'for {target} | {config}. Automatic comparison cannot be conducted.'
                )
            else:
                comp_band_name, band_mask = band_pick
                _, sig_med_arr, sig_z_arr = hstutils.band_integrated_flux_sigmas_vs_median_and_zero(
                    wave_grid, interp_fluxes, band_mask
                )
                w_band = wave_grid[band_mask]
                wa = float(np.nanmin(w_band))
                wb = float(np.nanmax(w_band))
                for j, spec in enumerate(spectra):
                    print(f'Assessing flux of {spec['id']}.')
                    id_mask = obs_tbl['archive id'] == spec['id']
                    sig_med = float(sig_med_arr[j])
                    sig_z = float(sig_z_arr[j])
                    if not np.isfinite(sig_med):
                        continue
                    note = obt.notes_menu['line flux vs med note'].format(
                        line=comp_band_name,
                        sigma=sig_med,
                        wa=wa,
                        wb=wb,
                    )
                    obs_tbl.add_note(id_mask, note, verbose=True)

                    acq_issue = np.any(acq_issues_mask & id_mask)
                    if np.isfinite(sig_z) and sig_z < zero_flux_sigma_threshold:
                        znote = obt.notes_menu['line flux vs zero note'].format(
                            line=comp_band_name,
                            sigma=sig_z,
                            wa=wa,
                            wb=wb,
                        )
                        obs_tbl.add_note(id_mask, znote, verbose=True)
                        obs_tbl.add_note(
                            id_mask,
                            obt.notes_menu['line flux vs zero warning'].format(
                                tol=zero_flux_sigma_threshold
                            ),
                            verbose=True,
                        )
                        if acq_issue:
                            obs_tbl.add_flag(id_mask, obt.flag_menu['no flux'], verbose=True)
                            obs_tbl.update_usability(id_mask, 'has issues')
                        spec_issues = True
                    else:
                        if abs(sig_med) > anomalous_flux_sigma_threshold:
                            obs_tbl.add_note(
                                id_mask,
                                obt.notes_menu['line flux vs med warning'].format(
                                    tol=anomalous_flux_sigma_threshold
                                ),
                                verbose=True,
                            )
                        if sig_med < -anomalous_flux_sigma_threshold:
                            loflag = obt.flag_menu['lo flux']
                            obs_tbl.add_flag(id_mask, loflag, verbose=True)
                            obs_tbl.update_usability(id_mask, 'has issues')
                            spec_issues = True
                        if sig_med > anomalous_flux_sigma_threshold:
                            hiflag = obt.flag_menu['hi flux']
                            obs_tbl.add_flag(id_mask, hiflag, verbose=True)
                            obs_tbl.update_usability(id_mask, 'has issues')
                            spec_issues = True

        # plot the spectra together in batches of no more than 5 on top of a thick background line
        # showing median and a light shaded region showing median abs dev
        spec_html_paths = []
        m = 5
        line_colors = qualitative.D3
        for start in range(0, len(spectra), m):
            batch = spectra[start:start + m]
            title_ids = '-'.join([str(spec['id'])[-6:] for spec in batch])

            fig = go.Figure()
            upper = median_flux + mad_flux
            lower = median_flux - mad_flux
            ok_band = (
                np.isfinite(wave_grid)
                & np.isfinite(upper)
                & np.isfinite(lower)
            )
            wg = wave_grid[ok_band]
            up = upper[ok_band]
            lo = lower[ok_band]
            fig.add_trace(
                go.Scatter(
                    x=wg,
                    y=up,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=wg,
                    y=lo,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,0,0,0.2)',
                    showlegend=False,
                    hoverinfo='skip',
                    name='±MAD',
                )
            )

            ok_med = np.isfinite(wave_grid) & np.isfinite(median_flux)
            fig.add_trace(
                go.Scatter(
                    x=wave_grid[ok_med],
                    y=median_flux[ok_med],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='median',
                    opacity=0.85,
                )
            )

            for k, spec in enumerate(batch):
                wx = spec['wavelength']
                fx = spec['flux']
                okf = np.isfinite(wx) & np.isfinite(fx)
                fig.add_trace(
                    go.Scatter(
                        x=wx[okf],
                        y=fx[okf],
                        mode='lines',
                        line=dict(color=line_colors[k % len(line_colors)], width=1),
                        line_shape='hv',
                        name=str(spec['id'])[-6:],
                        hovertemplate='λ=%{x:.4f} Å<br>flux=%{y:.4g}<extra></extra>',
                    )
                )

            layout_kw = dict(
                title=f'{target} | {config} | {title_ids}',
                xaxis_title='Wavelength (Å)',
                yaxis_title='Flux',
                height=520,
                width=1000,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                template='plotly_white',
            )
            if 'e140m' in config:
                lyamask = (wave_grid > 1214) & (wave_grid < 1217)
                if np.any(lyamask):
                    ylo = float(2 * np.nanmin(median_flux[lyamask]))
                    yhi = float(2 * np.nanmax(median_flux[lyamask]))
                    layout_kw['yaxis'] = dict(range=[ylo, yhi])

            fig.update_layout(**layout_kw)

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
                fig.add_annotation(
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    text='<br>'.join(notes),
                    showarrow=False,
                    align='left',
                    font=dict(size=11),
                )

            spec_fig_filename = f'{target}.{config}.{title_ids}.comparison.html'
            out_path = spec_plt_dir / spec_fig_filename
            fig.write_html(out_path, include_plotlyjs='cdn', config={'responsive': True})
            spec_html_paths.append(out_path)
            print(f'Wrote spectrum comparison: {out_path}')

        if human_review_spec == 'yes' or (human_review_spec == 'issues' and spec_issues):
            print('Open the spectrum comparison HTML file(s) in a browser, then continue here.')
            for p in spec_html_paths:
                print(f'  file://{p.resolve()}')
            input('Press Enter when ready to continue.')

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
                        obs_tbl.update_usability(i_mask, 'all clear')
                        break
                    elif usable_ans.startswith('u'):
                        reason = input(
                            f'Enter reason unusable. Use a reasons_menu key or exact value from '
                            f'observation_table.reasons_menu:'
                        )
                        obs_tbl.update_usability(i_mask, 'unusable', reason)
                        break
                    elif usable_ans.startswith('i'):
                        obs_tbl.update_usability(i_mask, 'has issues')
                        break
                    else:
                        print('Bad input.')

                while True:
                    flagstring = input(
                        f'Enter flags to be added, separated by commas (enter if none). '
                        f'Use flag_menu keys or exact flag strings from observation_table.flag_menu:'
                    )
                    if len(flagstring) < 3:
                        break
                    else:
                        obs_tbl.add_flags(i_mask, flagstring, separator=', *', verbose=False)

                while True:
                    notestring = input(
                        f'Enter notes/warnings to be added, separated by commas (enter if none). '
                        f'Use notes_menu keys or full strings (note … / warning …) from '
                        f'observation_table.notes_menu:'
                    )
                    if len(notestring) < 3:
                        break
                    else:
                        obs_tbl.add_notes(i_mask, notestring, separator=', *', verbose=False)

                print('Tbl updated to:')
                obs_tbl[viewcols][i_mask].pprint(-1, -1)

        plt.close('all')

        path_spec_diagnostics = diagnostics_dir / f'{target}.data-review-printout.{config}.txt'
        stop_and_save_capture(path_spec_diagnostics)

    utils.query_next_step(batch_mode, care_level, 2)


#%% mark files not listed as unusable and without flags as usable

    no_flags = [np.ma.is_masked(flags) or len(flags) == 0 for flags in obs_tbl['flags'] ]
    status = obs_tbl['usability status'].filled('')
    no_issues = (status == 'all clear') | (status == 'unchecked') | (status == '')
    mark_usable = no_flags & no_issues & obs_tbl['usable'].filled(True)
    obs_tbl.update_usability(mark_usable, 'all clear')


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

    obs_tbl.sort('start')
    obs_tbl.meta['last data review'] = datetime.now().isoformat()
    obs_tbl.meta['last review by'] = human_reviewer

    obs_tbl.meta['acq signal check number of tiles'] = 7
    obs_tbl.meta['acq signal check sigma threshold'] = 5.0
    obs_tbl.meta['flux median comparison min spectra required'] = 3
    obs_tbl.meta['flux median comparison sigma threshold'] = 3.0
    obs_tbl.meta['negligible flux sigma threshold'] = 1.0

    print(f'\nSaving obs_tbl for {target}.\n')
    obs_tbl.write(obs_tbl.get_path(target), overwrite=True)

    # save a more human readable version of the table
    diag_tbl_path = diagnostics_dir / f'{target}_observation_table.txt'
    diag_tbl_path.write_text(obs_tbl.pretty_string_with_flags_notes(), encoding='utf-8')
    print(f'Wrote diagnostic pretty-print to {diag_tbl_path}\n')

    utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    break