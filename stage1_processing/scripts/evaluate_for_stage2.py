from functools import lru_cache

from astropy.table import Table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import paths
import catalog_utilities as catutils
import utilities as utils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu
from stage1_processing import transit_evaluation_utilities as tutils

from lya_prediction_tools import stis
from lya_prediction_tools import lya


#%% occasional use: move model transit spectra into target folders

import os
import shutil as sh
import database_utilities as dbutils

models_inbox = paths.inbox / '2025-07-30 transit predictions'

files = list(models_inbox.glob('*.h5'))

targnames_ethan = [file.name[:-4] for file in files]
targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
targnames_file = dbutils.target_names_stela2file(targnames_stela)

def move_files(dry_run=True):
    for targname, file in zip(targnames_file, files):
        planet = file.name[-4]
        newname = f'{targname}.outflow-tail-model.na.na.transit-{planet}.h5'
        newfolder = paths.target_data(targname) / 'transit predictions'

        if dry_run:
            print(f'{file.name} --> {'/'.join(newfolder.parts[-2:])}/{newname}')
        else:
            if not newfolder.exists():
                os.mkdir(newfolder)
            sh.copy(file, newfolder / newname)

move_files(dry_run=True)
for_reals = input('\nProceed with copying the files? (enter/n)')
if for_reals == '':
    move_files(dry_run=False)


#%% targets and code running options

targets = target_lists.eval_no(1)[:3]
mpl.use('Agg') # plots in the backgrounds so new windows don't constantly interrupt my typing
# mpl.use('qt5agg') # plots are shown
np.seterr(divide='raise', over='raise', invalid='raise') # whether to raise arithmetic warnings as errors
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))


#%% instrument details

apertures_to_consider = dict(
    g140m='52x0.5 52x0.2 52x0.1 52x0.05'.split(),
    e140m='6x0.2 52x0.05'.split(),
    g130m=('psa',)
)

acq_exptime_guess = 5
def exptime_fn(aperture):
    """nominal exposure times shortened by peakups. accounts for the tradeoff between
    less airglow contamination and shorter exposures."""
    exptimes_mod =  stis.shorten_exposures_by_peakups(
        aperture,
        acq_exptime_guess,
        exptimes,
        visit_start_indices=[0,2]
    )
    return exptimes_mod


#%% set assumed observation timing

obstimes = [-22.5, -21., -3., -1.5,  0.,  1.5,  3.] * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s
offsets = range(0, 17)*u.h
safe_offsets = range(0, 4)*u.h # offsets we will actually consider at this stage
baseline_exposures = slice(0, 2)
transit_exposures = slice(4, None)
baseline_apertures = dict(g140m='52x0.2', e140m='6x0.2')
cos_consideration_threshold_flux = 2e-14


#%% ranges within which to search for integration bands that maximize SNR

normalization_search_rvs = ((-400, -150), (150, 400)) * u.km / u.s
search_model_transit_within_rvs = (-150, 50) * u.km / u.s
search_simple_transit_within_rvs = (-150, 100) * u.km / u.s
simple_transit_range = (-150, 100) * u.km / u.s


#%% assumed jitter and rotation variability as a function of Ro

variability_predictor = tutils.VariabilityPredictor(
    Ro_break=0.1,
    jitter_saturation=0.1,
    jitter_Ro1=0.01,
    rotation_amplitude_saturation=0.25,
    rotation_amplitude_Ro1=0.05,
)


#%% tables

with catutils.catch_QTable_unit_warnings():
    planet_catalog = preloads.planets.copy()
planet_catalog.add_index('tic_id')


#%% helper to construct host object and then add variability guesses based on assumptions above

@lru_cache
def get_host_objects(name):
    host = tutils.Host(name)
    # add variability guesses based on the variability_predictor set earlier in this script
    host_variability = tutils.HostVariability(host, variability_predictor)
    return host, host_variability


#%% define function to do a nested exploration of different observational setups

def explore_snrs(
        host: tutils.Host,
        host_variability: tutils.HostVariability,
        transit: tutils.TransitModelSet,
        transit_search_rvs: u.Quantity,
        exptime_fn
):

    snr_cases, snr_single = tutils.build_snr_sampler_fns(
        host,
        host_variability,
        transit,
        exptime_fn,
        obstimes,
        baseline_exposures,
        transit_exposures,
        normalization_search_rvs,
        transit_search_rvs
    )

    # first try various time offsets for a single aperture and lya case
    print('Comparing mid-transit observation to offest(s).')
    grating = host.anticipated_grating
    base_aperture = baseline_apertures[grating]
    tbl1 = snr_cases(offsets, [(grating, base_aperture)], [0])

    # record best offset overall
    best_offset = tutils.best_by_mean_snr(tbl1, 'time offset')
    tbl1.meta['best time offset'] = best_offset

    # pick offset to use from a smaller "safe" range
    tbl1.add_index('time offset')
    tbl1_safe = tbl1.loc[safe_offsets.value]
    best_safe_offset = tutils.best_by_mean_snr(tbl1_safe, 'time offset')
    tbl1.meta['best safe time offset'] = best_safe_offset

    # run for all apertures at best safe, pick best aperture, record
    print('Finding the best aperture.')
    apertures = apertures_to_consider[grating]
    grating_apertures = [(grating, ap) for ap in apertures]
    tbl2 = snr_cases([best_safe_offset], grating_apertures, [0])
    aperture = tutils.best_by_mean_snr(tbl2, 'aperture')
    tbl2.meta['best stis aperture'] = str(aperture)

    # run for all lya cases at 0, best_safe, and best offset
    print('Running for the plausible Lya range.')
    cases = np.arange(-2, 3)
    tbl3 = snr_cases([0*u.h, best_safe_offset, best_offset], [(grating, aperture)], cases)

    tbls = [tbl1, tbl2, tbl3]

    # optionally add cos
    lya = host.lya_reconstruction
    Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
    if Flya > cos_consideration_threshold_flux:
        print('Target Lya flux makes it eligible for COS: adding estimates for COS SNRs.')
        tbl3.meta['COS considered'] = True
        tbl4 = snr_cases([best_safe_offset], [('g130m', 'psa')], cases)
        tbls.append(tbl4)
    else:
        tbl3.meta['COS considered'] = False
        tbl3.meta['notes'] = 'COS not considered because flux too low.'

    return catutils.table_vstack_flexible_shapes(tbls), snr_single


#%% loop through planets and targets and compute transit sigmas

for target in utils.printprogress(targets, prefix='host '):
    host, host_variability = get_host_objects(target)

    for planet in utils.printprogress(host.planets, 'dbname', prefix='\tplanet '):
        transit = tutils.get_transit_from_simulation(host, planet)
        snrs, get_snr = explore_snrs(
            host,
            host_variability,
            transit,
            search_model_transit_within_rvs,
            exptime_fn)

        filenamer = tutils.FileNamer('model', planet)

        snrs.write(host.transit_folder / filenamer.snr_tbl, overwrite=True)

        # diagnostic plots
        best_ap = snrs.meta['best stis aperture']
        best_offset = snrs.meta['best safe time offset']
        best_snrs = tutils.filter_to_obs_choices(snrs, best_ap, best_offset)
        isort = np.argsort(best_snrs['transit sigma'])
        cases = {'max': isort[-1],
                 'median': isort[len(isort) // 2]}
        for case, k in cases.items():
            wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, best_snrs[k])
            tutils.save_diagnostic_plots(wfig, tfig, case, host, filenamer)

        # corner-like plot
        labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],σ_Lya'.split(',')
        lTion = best_snrs['Tion'].to_value('dex(h)')
        lMdot = best_snrs['mdot_star'].to_value('dex(g s-1)')
        leta = np.log10(best_snrs['eta'])
        lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in best_snrs['lya reconstruction case']]
        param_vecs = [leta, lTion, lMdot, lya_sigma]
        snr_vec = best_snrs['transit sigma']
        cfig, _ = pu.detection_sigma_corner(param_vecs, snr_vec, labels=labels,
                                            levels=(3,), levels_kws=dict(colors='w'))
        cfig.suptitle(planet.dbname)
        utils.save_pdf_png(cfig, host.transit_folder / filenamer.corner_basename)

        plt.close('all')


#%% flat opaque tail transit

        flat_transit = tutils.construct_flat_transit(
            planet, host, obstimes, exptimes,
            rv_grid_span=(-500, 500) * u.km/u.s,
            rv_range=simple_transit_range,
        )

        flat_snrs, get_flat_snr = explore_snrs(
            host,
            host_variability,
            flat_transit,
            search_simple_transit_within_rvs,
            exptime_fn
        )

        flat_filenamer = tutils.FileNamer('flat', planet)
        flat_snrs.write(host.transit_folder / flat_filenamer.snr_tbl, overwrite=True)

        # diagnostic plots
        best_flat_ap = flat_snrs.meta['best stis aperture']
        best_flat_offset = flat_snrs.meta['best safe time offset']
        best_flat_snrs = tutils.filter_to_obs_choices(flat_snrs, best_flat_ap, best_flat_offset)
        isort = np.argsort(best_flat_snrs['transit sigma'])
        i_med = isort[len(isort) // 2]
        wfig, tfig = tutils.make_diagnostic_plots(planet, flat_transit, get_flat_snr, best_flat_snrs[i_med])
        tutils.save_diagnostic_plots(wfig, tfig, 'median', host, flat_filenamer)
        plt.close('all')


#%% make table of properties

sigma_threshold = 3
lya_bins = (-150, -50, 50, 150) * u.km/u.s

eval_rows = []
for target in targets:
    host, host_variability = get_host_objects(target)
    for planet in host.planets:
        # add entries to the row in the order they should appear in the table
        row = {}
        
        row['hostname'] = host.hostname
        row['planet'] = planet.stela_suffix

        def add_config_info(snr_table, label):
            row[f'{label}\nbest aperture'] = snr_table.meta['best stis aperture']
            row[f'{label}\nbest offset'] = snr_table.meta['best time offset']

        def add_detection_stats(snr_table, label, fraction=False):
            maxsnr = np.max(snr_table['transit sigma'])
            row[f'{label}\nmax snr'] = maxsnr
            if fraction:
                detectable = snr_table['transit sigma'] > sigma_threshold
                detectability_fraction = np.sum(detectable) / len(snr_table)
                row[f'{label}\nfrac w snr > {sigma_threshold}'] = detectability_fraction
                return maxsnr, detectability_fraction
            return maxsnr

        # region model snrs
        modlbl = 'sim'

        # load model snr table
        filenamer = tutils.FileNamer('model', planet)
        sigma_tbl_path = host.transit_folder / filenamer.snr_tbl
        snrs = Table.read(sigma_tbl_path)

        best_aperture = snrs.meta['best stis aperture']

        lbl = f'{modlbl} no offset'
        chosen_mode_snrs = tutils.filter_to_obs_choices(snrs, aperture=best_aperture, offset=0*u.h)
        maxsnr, frac = add_detection_stats(chosen_mode_snrs, lbl, fraction=True)

        best_safe_offset = snrs.meta['best safe time offset']
        row['best safe\ntransit offset'] = best_safe_offset
        lbl = f'{modlbl} safe offset'
        chosen_mode_snrs = tutils.filter_to_obs_choices(snrs, aperture=best_aperture, offset=best_safe_offset)
        maxsnr, frac = add_detection_stats(chosen_mode_snrs, lbl, fraction=True)

        best_offset = snrs.meta['best time offset']
        row['best overall\ntransit offset'] = best_offset
        lbl = f'{modlbl} best offset'
        chosen_mode_snrs = tutils.filter_to_obs_choices(snrs, aperture=best_aperture, offset=best_offset)
        maxsnr, frac = add_detection_stats(chosen_mode_snrs, lbl, fraction=True)

        cos = snrs.meta['COS considered']
        row['COS\nconsidered?'] = cos
        if cos:
            cos_snrs = snrs[snrs['aperture'] == 'psa']
            maxcossnr, cosfrac = add_detection_stats(cos_snrs, modlbl + ' COS', fraction=True)
            if frac > 0:
                row['cos det\nfrac ratio'] = cosfrac/frac
            row['cos snr\nratio'] = maxcossnr/maxsnr

        add_config_info(snrs, modlbl)
        # endregion

        # region flat transit
        # load snr table
        flat_filenamer = tutils.FileNamer('flat', planet)
        flat_sigma_tbl_path = host.transit_folder / filenamer.snr_tbl
        flat_sigma_tbl = Table.read(flat_sigma_tbl_path)

        flatlbl = f'flat transit \n[{simple_transit_range[0].value:.0f}, {simple_transit_range[1].value:.0f}]'
        best_flat_ap = flat_sigma_tbl.meta['best stis aperture']
        flat_obs_snrs = tutils.filter_to_obs_choices(flat_sigma_tbl, best_flat_ap, 0*u.h)
        add_detection_stats(flat_obs_snrs, 'flat transit', fraction=False)
        add_config_info(flat_sigma_tbl, flatlbl)
        # endregion

        wlya = host.lya_reconstruction.wavegrid_earth
        y = host.lya_reconstruction.fluxes[0]
        Flya = np.trapz(y, wlya)
        row['Lya Flux\n(erg s-1 cm-2)'] = Flya

        Mp = planet.params['pl_bmasse'].to_value('Mearth')
        Mp_err = 0.5 *(planet.params['pl_bmasseerr1']
                       - planet.params['pl_bmasseerr2'])
        Mp_prec = Mp_err/Mp
        if not np.isfinite(Mp_prec) or Mp_prec == 0:
            Mp_prec = np.ma.masked
        row['mass (Me)'] = Mp
        row['mass\nprecision'] = Mp_prec
        if planet.params['pl_bmassesrc'] == 'inferred from Rp':
            mass_source = 'M-R relationship'
        else:
            mass_source_rename = {'Mass': 'known'}
            mass_source = str(planet.params['pl_bmassprov'])
            mass_source = mass_source_rename[mass_source]
        row['mass\nsource'] = mass_source
        mass_flag = np.ma.masked
        if planet.params['pl_rade'] > 7*u.Rearth:
            mass_flag = 'giant'
        if planet.params['flag_young']:
            mass_flag = 'young'
        row['mass\nflag'] = mass_flag

        row['radius (Re)'] = planet.params['pl_rade'].to_value('Rearth')
        row['orbital\nperiod (d)'] = planet.params['pl_orbper'].to_value('d')
        row['stellar\neff temp (K)'] = host.params['st_teff'].to_value('K')
        age = host.params['st_age'].to_value('Gyr')
        if not np.ma.is_masked(age):
            agelim_int = host.params['st_agelim']
            if not np.ma.is_masked(agelim_int):
                row['age\nlimit'] = catutils.limit_int2str[agelim_int]
            row['age (Gyr)'] = host.params['st_age'].to_value('Gyr')

        row['obsvtn\ngrating'] = host.anticipated_grating

        flag_cols = [name for name in planet_catalog.colnames if 'flag_' in name]
        for col in flag_cols:
            row[col] = planet.params[col]

        transit = tutils.get_transit_from_simulation(host, planet)
        row['H ionztn\ntime (h)'] = np.median(transit.params['Tion']).to_value('h')

        eval_rows.append(row)

# get column ordering from longest row
imax = np.argmax([len(row) for row in eval_rows])
ordered_cols = list(eval_rows[imax].keys())

eval_table = Table(rows=eval_rows)
eval_table = eval_table[ordered_cols]

# some grooming
for col in eval_table.colnames:
    if 'flag_' in col:
        eval_table[col] = eval_table[col].astype(bool)
formats_general = {
    'period': '.1f',
    'frac w': '.3f',
    'max snr': '.2f',
    'ratio': '.2f',
    'flux': '.1e',
    'radius': '.2f',
    'temp': '.0f',
    'ionztn': '.2f'
}
formats = {}
for substr, fmt in formats_general.items():
    for name in eval_table.colnames:
        if substr in name.lower():
            eval_table[name].format = fmt
            formats[name] = fmt

# save csv to open in spreadsheet viewers
eval_filename = 'stage2_evaluation_metrics.csv'
eval_path = paths.catalogs / eval_filename
eval_table.write(eval_path, overwrite=True, formats=formats)

# save as an ecsv too for round tripping
eval_table_ecsv = eval_table.copy()
for name in eval_table_ecsv.colnames:
    eval_table_ecsv.rename_column(name, name.replace('\n', ' '))
eval_path_ecsv = paths.catalogs / eval_filename.replace('csv', 'ecsv')
eval_table_ecsv.write(eval_path_ecsv, overwrite=True)
