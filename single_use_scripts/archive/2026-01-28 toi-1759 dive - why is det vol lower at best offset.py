import re
from typing import Literal
import os

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

from processing import target_lists
from processing import preloads
from processing import processing_utilities as pu
from processing import transit_evaluation_utilities as tutils

from lya_prediction_tools import stis
from lya_prediction_tools import lya


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

staging_area = paths.packages / '2025-09-26.stage2.eval2.staging_area'

target = 'toi-1759'
scratchfolder = paths.scratch / '2026-01 toi-1759 dive - why is det vol lower at best offset'
os.makedirs(scratchfolder, exist_ok=True)

sigma_threshold = 3
min_samples = 5**4 # used as a check later to ensure all grid pts of Ethan's sims were sampled

# mpl.use('Agg') # plots in the background so new windows don't constantly interrupt my typing
mpl.use('qt5agg') # plots are shown
np.seterr(divide='raise', over='raise', invalid='raise') # whether to raise arithmetic warnings as errors
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))

plot_from_excel = False
if plot_from_excel:
    url_google_sheet = 'https://docs.google.com/spreadsheets/d/1G77756ETnRfSoCjw-ZOD7rCVP124XmpjYLzRW2-ckzQ/export?format=xlsx&gid=985313130'


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


#%% planet catalog

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


#%% get host, planet, transit, snr_db

host = tutils.Host(target, host_catalog, planet_catalog)
planet = host.planets[0]
transit = get_transit(planet, host, 'model')
snr_db = load_snr_db(planet, host, 'model')


#%% test different metric for picking best offset

"""best offset (and aperture, etc.) were picked using best mean snr. This is probably what causes confusion
since we later use det vol as the primary comparison metric. And it explains the drop in det vol bc
I see in the table that max snr increases even for systems where det vol drops as offset increases. """

snr_db_start = snr_db.filter_obs_config(grating='base', aperture='base', offset='all')
snr_db_start = snr_db_start.filtered({'lya reconstruction case': 'median'})
snr_db_start = snr_db_start.clean_duplicates()

best_off_by_mean = tutils.best_by_mean_snr(snr_db_start.snrs, 'time offset')
best_off_by_frac = tutils.best_by_det_frac(snr_db_start.snrs, 'time offset', 3)

"""huh, I get a best offset that doesn't even match what is in the table. let me try rerunning this case"""


#%% rerun toi-1759


build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration

grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)
tst_type = 'model'
transit = get_transit(planet, host, tst_type)
get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
def build_planet_snrs(grating, base_aperture, all_apertures):
    snrs = build_snrs(get_snr_iterable, grating, base_aperture, all_apertures,
                      offsets, max_safe_offset, verbose=True)
    return snrs

snrs = build_planet_snrs(grating, base_aperture, all_apertures)

"""okay yeah I get a different offset. not sure why Ava got the offset she did.
and using det vol gives me 0 instead of 3 h as the best offset. so I think that is the source of the problems
I've encountered. I'll go ahead and start rerunning them all."""