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

# %% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = 'toi-2443 toi-431 hd63935 l98-59 ltt1445a'.split()

tst_type = 'model'

mpl.use('Agg')  # plots in the background so new windows don't constantly interrupt my typing
# mpl.use('qt5agg') # plots are shown

staging_area = paths.packages / '2025-09-26.stage2.eval2.staging_area'

np.seterr(divide='raise', over='raise', invalid='raise')  # whether to raise arithmetic warnings as errors
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))

sigma_threshold = 1
min_samples = 5 ** 4  # used as a check later to ensure all grid pts of Ethan's sims were sampled

# %% instrument details

apertures_to_consider = dict(
    g140m='52x0.5 52x0.2 52x0.1 52x0.05'.split(),
    e140m='6x0.2 52x0.05'.split(),
    g130m=('psa',)
)

acq_exptime_guess = 5


def exptime_fn(aperture):
    """nominal exposure times shortened by peakups. accounts for the tradeoff between
    less airglow contamination and shorter exposures."""
    exptimes_mod = stis.shorten_exposures_by_peakups(
        aperture,
        acq_exptime_guess,
        exptimes,
        visit_start_indices=[0, 2]
    )
    return exptimes_mod


# %% set assumed observation timing

obstimes = [-22.5, -21., -3., -1.5, 0., 1.5, 3.] * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s
offsets = range(0, 17) * u.h
max_safe_offset = 3 * u.h  # offsets we will actually consider at this stage
baseline_exposures = slice(0, 2)
transit_exposures = slice(4, None)
baseline_apertures = dict(g140m='52x0.2', e140m='6x0.2')
cos_consideration_threshold_flux = 2e-14

# %% ranges within which to search for integration bands that maximize SNR

normalization_search_rvs = ((-400, -150), (150, 400)) * u.km / u.s
search_model_transit_within_rvs = (-150, 50) * u.km / u.s
search_simple_transit_within_rvs = (-150, 100) * u.km / u.s
simple_transit_range = (-150, 100) * u.km / u.s

# %% assumed jitter and rotation variability as a function of Ro

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



# %% a few loose closures

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



# %% compare det vols

for target in targets:
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        transit = get_transit(planet, host, tst_type)
        _, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
        snrs = load_snr_db(planet, host, tst_type)

        best_off = snrs.meta['best safe time offset']
        grating = snrs.meta['base grating']
        configs = {'time offset': best_off,
                   'grating': grating}
        ap_test_snrs = snrs.filtered(configs)
        ap_test_snrs = ap_test_snrs.clean_duplicates()

        snrs_pt2 = snrs.filtered({'aperture': '52x0.2'})
        det_vol_pt2 = snrs_pt2.detection_fraction(sigma_threshold)

        best_ap = snrs.meta['best base grating aperture']
        snrs_best_ap = snrs.filtered({'aperture': best_ap})
        det_vol_best_ap = snrs_best_ap.detection_fraction(sigma_threshold)

        print('')
        print(f'{planet.dbname}')
        print(f'Det vol with 52x0.2 aperture: {det_vol_pt2:.3f}')
        print(f'Det vol with {best_ap} (best) aperture: {det_vol_best_ap:.3f}')

#%% conclusion

"""The 0.5 aperture makes all the difference in these cases. The SNR calcs are factoring in
 the RMS variability from slit breathing cataloged by Bohlin and Hartig, so I think we need
 to trust that this does indeed make the difference. It is likely that the transit will be
 under the noise of the breathing variability otherwise. """