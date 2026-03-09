from functools import lru_cache
import re
from typing import Literal
import os
import shutil as sh

from astropy.table import Table
from astropy import table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import paths
import catalog_utilities as catutils
import database_utilities as dbutils
import utilities as utils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu
from stage1_processing import transit_evaluation_utilities as tutils

from lya_prediction_tools import stis
from lya_prediction_tools import lya


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = set(target_lists.eval_no(1)) | set(target_lists.eval_no(2)) - {'v1298tau'}

tst_types = ('model', 'flat')

mpl.use('Agg') # plots in the background so new windows don't constantly interrupt my typing
# mpl.use('qt5agg') # plots are shown

staging_area = paths.packages / '2025-09-26.stage2.eval2.staging_area'

np.seterr(divide='raise', over='raise', invalid='raise') # whether to raise arithmetic warnings as errors
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))

sigma_threshold = 1
min_samples = 5**4 # used as a check later to ensure all grid pts of Ethan's sims were sampled


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
max_safe_offset = 3*u.h # offsets we will actually consider at this stage
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


#%% planet and host catalogs

with catutils.catch_QTable_unit_warnings():
    planet_catalog = catutils.load_and_mask_ecsv(staging_area / 'planet_catalog.ecsv')
    planet_catalog = table.QTable(planet_catalog)
    host_catalog = catutils.planets2hosts(planet_catalog)
    planet_catalog.add_index('tic_id')
    host_catalog.add_index('tic_id')


#%% a few loose closures

def get_transit(planet, host, tst_type: Literal['model', 'flat']):
    if tst_type == 'model':
        return tutils.get_transit_from_simulation(host, planet)
    elif tst_type == 'flat':
        transit_flat = tutils.construct_flat_transit(
            planet, host, obstimes, exptimes,
            rv_grid_span=(-500, 500) * u.km / u.s,
            rv_range=simple_transit_range,
        )
        return transit_flat
    else:
        raise ValueError('tst_type not recognized')

def path_snrs(planet, host, tst_type: Literal['model', 'flat']):
    filenamer = tutils.FileNamer(tst_type, planet, host)
    return filenamer.snr_tbl_full

def load_snr_db(planet, host, tst_type: Literal['model', 'flat']):
    path = path_snrs(planet, host, tst_type)
    return tutils.DetectabilityDatabase.from_file(path)

def load_best_snrs(planet, host, tst_type: Literal['model', 'flat']):
    snrs = load_snr_db(planet, host, tst_type)
    best_snrs = snrs.filter_obs_config(aperture='best', offset='best safe')
    best_snrs = best_snrs.clean_duplicates()
    return best_snrs

def get_lya_flux(host):
    lya = host.lya_reconstruction
    Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
    return Flya

def get_obs_config_info(host):
    # observational configs to consider
    grating = host.anticipated_grating
    base_aperture = baseline_apertures[grating]
    all_apertures = apertures_to_consider[grating]

    # consider COS too?
    Flya = get_lya_flux(host)
    consider_cos = Flya > cos_consideration_threshold_flux
    return grating, base_aperture, all_apertures, consider_cos

def consrtuct_snr_samplers(host, transit, tst_type):
    if tst_type == 'flat':
        transit_search_rvs = search_simple_transit_within_rvs
    elif tst_type == 'model':
        transit_search_rvs = search_model_transit_within_rvs
    else:
        raise ValueError
    host_variability = tutils.HostVariability(host, variability_predictor)
    get_snr_iterable, get_snr_single = tutils.build_snr_sampler_fns(
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
    return get_snr_iterable, get_snr_single


#%% match in catalog of flags in case any were left out of staging area catalog

with catutils.catch_QTable_unit_warnings():
    planet_catalog_current = preloads.planets.copy()
    newflags = [name for name in planet_catalog_current.colnames if 'flag_' in name and name not in planet_catalog.colnames]
    planetcols_for_joining = planet_catalog_current[['id'] + newflags]
    planet_catalog = table.join(planet_catalog, planetcols_for_joining, keys='id', join_type='left')

planet_catalog.add_index('tic_id')

#%% assemble table of properties

lya_bins = (-150, -50, 50, 150) * u.km/u.s

eval_rows = []
targets = sorted(targets)
for target in tqdm(targets):
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        # add entries to the row in the order they should appear in the table
        row = {}
        
        row['hostname'] = host.hostname
        row['planet'] = planet.stela_suffix

        # region model snrs
        snr_db = load_snr_db(planet, host, 'model')
        slctd_offsets, det_fracs, max_snrs = snr_db.offset_stats(sigma_threshold, min_sample_check=5**4)

        row['best safe\ntransit offset'] = slctd_offsets[1]
        row['best overall\ntransit offset'] = slctd_offsets[2]

        off_lbls = ['no', 'safe', 'best']
        stat_sets = zip(det_fracs, max_snrs, off_lbls)
        for det_frac, max_snr, off_lbl in stat_sets:
            row[f'sim {off_lbl} offset \nmax snr'] = max_snr
            row[f'sim {off_lbl} offset \nfrac w snr > {sigma_threshold}'] = det_frac

        row['sim\nbest aperture'] = snr_db.meta['best base grating aperture']

        cos = 'g130m' in snr_db['grating']
        row['COS\nconsidered?'] = cos
        if cos:
            cos_snrs = snr_db.filter_obs_config(grating='g130m', aperture='psa', offset='best safe')
            cos_snrs = cos_snrs.clean_duplicates()
            cosfrac, maxcossnr = cos_snrs.det_frac_and_max_snr(sigma_threshold)
            row['sim COS safe offset\nmax snr'] = maxcossnr
            row[f'sim COS safe offset\nfrac w snr > {sigma_threshold}'] = cosfrac
            if det_fracs[1] > 0:
                row['cos det\nfrac ratio'] = cosfrac/det_fracs[1]
            row['cos snr\nratio'] = maxcossnr/max_snrs[1]
        # endregion

        # region flat transit
        # load snr table
        snr_db_flat = load_snr_db(planet, host, 'flat')
        snr_db_flat = snr_db_flat.filter_obs_config(aperture='best', offset='best safe')
        snr_db_flat = snr_db_flat.clean_duplicates()
        flat_snr = snr_db_flat.median_case()['transit sigma']
        row['flat transit\nsnr'] = flat_snr
        row['flat transit\nbest aperture'] = snr_db_flat.meta['best base grating aperture']
        # endregion

        Flya = get_lya_flux(host)
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
            mass_source_rename = {'Mass': 'known', 'M-R relationship': 'M-R relationship'}
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

eval_table['TIC'] = preloads.stela_names.loc['hostname', eval_table['hostname']]['tic_id']

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


#%% match in the catalog of escape detections

escape_detections = catutils.escape_catalog_merge_targets('download')

det_ids = np.char.add(escape_detections['Target Star'], escape_detections['Planet Letter'])
pcat_ids = np.char.add(planet_catalog_current['hostname'].astype(str),
                       dbutils.planet_suffixes(planet_catalog_current).astype(str))
eval_ids = np.char.add(eval_table['hostname'], eval_table['planet'])

# check to be sure names in escape detections match into planet catalog
suspect = ~np.isin(det_ids, pcat_ids)
if np.any(suspect):
    print("These names don't have a match in the planet catalog."
          "(Note this could be do to cuts to planet catalog, such as requiring < 100 d periods (e.g., HD 136352d).)")
    print(det_ids[suspect])

# match using name + letter
det_colnames = [name for name in escape_detections.colnames if 'detected' in name.lower()]
det_slim = escape_detections[det_colnames]
det_slim['temp'] = det_ids
eval_table['temp'] = eval_ids
eval_table = table.join(eval_table, det_slim, keys='temp', join_type='left')
eval_table.remove_column('temp')


#%% match in requested targets

files = list(paths.stage2_requests.glob('*.txt'))
path_check_table, = paths.selection_intermediates.glob('*pt1*.ecsv')
check_table = catutils.load_and_mask_ecsv(path_check_table)

def any_tois(planet_names):
    for name in planet_names:
        x = re.findall(r'TOI-\d+\.0\d', name)
        if x:
            return True
    return False

eval_table['TICletter'] = np.char.add( # temporary column for matchin
    eval_table['TIC'].astype(str),
    eval_table['planet']
)
check_table['TICletter'] = np.char.add( # temporary column for matching
    check_table['tic_id'].astype(str),
    check_table['pl_letter'].astype(str).filled('')
)
for file in files:
    requested_planets = catutils.read_requested_targets(file)
    if any_tois(requested_planets):
        raise NotImplementedError

    # match with TIC to ensure avoid misses
    hosts_letters = list(map(dbutils.split_hostname_planet_letter, requested_planets))
    hosts, letters = zip(*hosts_letters)
    tics = dbutils.query.query_simbad_for_tic_ids(hosts)
    tics_letters = np.char.add(tics, letters)

    # mark matches in a "requested" column
    request_name = 'requested\n' + requested_planets.name
    eval_table[request_name] = np.isin(eval_table['TICletter'], tics_letters)

    # print requested targets with no match in STELa tables
    not_in_list = ~np.isin(tics_letters, check_table['TICletter'])
    if np.any(not_in_list):
        print(f'\n{requested_planets.name} planets not matched to any planet known to STELa:')
        for name in requested_planets[not_in_list]:
            print(f'\n\t{name}')
        print('Consider checking that they are correctly named in the requested list txt file.')

eval_table.remove_column('TICletter')


#%% save table

# save csv to open in spreadsheet viewers
eval_filename = 'stage2_evaluation_metrics.csv'
eval_path = paths.catalogs / eval_filename
eval_table.write(eval_path, overwrite=True, formats=formats, fast_writer=False)

# save as an ecsv too for round tripping
eval_table_ecsv = eval_table.copy()
for name in eval_table_ecsv.colnames:
    eval_table_ecsv.rename_column(name, name.replace('\n', ' '))
eval_path_ecsv = paths.catalogs / eval_filename.replace('csv', 'ecsv')
eval_table_ecsv.write(eval_path_ecsv, overwrite=True)
