"""lots of vestigial code here, this was just a quick and dirty operation"""


from functools import lru_cache
import re

from astropy.table import Table
from astropy import table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import paths
import catalog_utilities as catutils
import database_utilities as dbutils
import utilities as utils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu
from stage1_processing import transit_evaluation_utilities as tutils

from target_selection_tools import query

from lya_prediction_tools import stis
from lya_prediction_tools import lya


#%% targets and code running options

targets = set(target_lists.eval_no(1)) | set(target_lists.eval_no(2)) - {'v1298tau'}
mpl.use('Agg') # plots in the backgrounds so new windows don't constantly interrupt my typing
np.seterr(divide='raise', over='raise', invalid='raise') # whether to raise arithmetic warnings as errors
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))
targets = list(targets)


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


#%% find targets with incomplete tables

from tqdm import tqdm
sigma_threshold = 3
lya_bins = (-150, -50, 50, 150) * u.km / u.s
min_samples = 5 ** 4

incomplete_targets = []
for target in tqdm(targets):
    host, host_variability = get_host_objects(target)
    for planet in host.planets:
        # region model snrs
        modlbl = 'sim'

        # load model snr table
        filenamer = tutils.FileNamer('model', planet)
        sigma_tbl_path = host.transit_folder / filenamer.snr_tbl
        snrs = Table.read(sigma_tbl_path)

        grating = host.anticipated_grating
        base_aperture = baseline_apertures[grating]
        best_aperture = snrs.meta['best stis aperture']

        base_filters = { # should have been run for 18 offsets
            'grating': grating,
            'aperture': base_aperture,
            'lya reconstruction case': 'median'
        }
        base_snrs = catutils.filter_table(snrs, base_filters)
        if len(base_snrs)/min_samples < 18:
            incomplete_targets.append(target)
            continue

        lbl = f'{modlbl} no offset'
        no_offset_filters = { # should have been run for all lya cases
            'grating': grating,
            'aperture': best_aperture,
            'time offset': 0
        }
        no_offset_snrs = catutils.filter_table(snrs, no_offset_filters)
        if len(np.unique(no_offset_snrs['lya reconstruction case'])) < 5:
            incomplete_targets.append(target)
            continue

        lbl = f'{modlbl} safe offset'
        best_safe_offset = tutils.best_offset(base_snrs, safe_offsets.max())
        safe_filters = { # should have been run for all lya cases and all apertures
            'grating': grating,
            'time offset': best_safe_offset
        }
        best_safe_snrs = catutils.filter_table(snrs, safe_filters)
        if len(np.unique(best_safe_snrs['lya reconstruction case'])) < 5:
            incomplete_targets.append(target)
            continue
        if len(np.unique(best_safe_snrs['aperture'])) < len(apertures_to_consider[grating]):
            incomplete_targets.append(target)
            continue


        lbl = f'{modlbl} best offset'
        best_offset = tutils.best_offset(base_snrs)
        best_filters = { # should have been run for all lya cases
            'grating': grating,
            'aperture': best_aperture,
            'time offset': best_offset
        }
        best_overall_snrs = catutils.filter_table(snrs, best_filters)
        if len(np.unique(best_overall_snrs['lya reconstruction case'])) < 5:
            incomplete_targets.append(target)
            continue

        lya = host.lya_reconstruction
        Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
        if Flya > cos_consideration_threshold_flux:
            if np.sum(snrs['grating'] == 'g130m') == 0:
                incomplete_targets.append(target)
                continue


targets = list(set(incomplete_targets))





