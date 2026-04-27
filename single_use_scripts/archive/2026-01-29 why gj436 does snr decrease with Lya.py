import re
from typing import Literal

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

pass
# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

staging_area = paths.packages / '2025-09-26.stage2.eval2.staging_area'

target = 'gj436'
scratchfolder = paths.scratch / '2026-01 gj436 dive'

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


#%% construct host and planet

host = tutils.Host(target, host_catalog, planet_catalog)
planet = host.planets[0]
transit = get_transit(planet, host, 'model')
snr_db = load_snr_db(planet, host, 'model')
best_snrs = load_best_snrs(planet, host, 'model')

#%% are there duplicates?

len(best_snrs)
test = best_snrs.clean_duplicates()
len(test)
"""yes there are"""

#%% get median snrs

cases = (
    'low_2sig',
    'low_1sig',
    'median',
    'high_1sig',
    'high_2sig',
)
for case in cases:
    temp = best_snrs.filtered({'lya reconstruction case':case})
    sig = np.median(temp['transit sigma'])
    print(f'{case}: {sig:.3f}')

print('')

for case in cases:
    temp = test.filtered({'lya reconstruction case':case})
    sig = np.median(temp['transit sigma'])
    print(f'{case}: {sig:.3f}')

"""okay weird. snr decreases from 0 to 1 sig for both cleaned and full snr table.
looking at "max snr" plot, it seems weird, and its different than when I made 
plots based on Ava's results. going to try rerunning with an old version of transit.py"""

#%% reprocess det vol

"""here I checked out the 12/15 revision of transit.py"""

tst_type = 'model'
grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)
transit = get_transit(planet, host, tst_type)
get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration


def build_planet_snrs(grating, base_aperture, all_apertures):
    snrs = build_snrs(get_snr_iterable, grating, base_aperture, all_apertures,
                      offsets, max_safe_offset, verbose=True)
    return snrs


snrs = build_planet_snrs(grating, base_aperture, all_apertures)
# optionally add COS
if consider_cos:
    cos_snrs = build_planet_snrs('g130m', 'psa', ['psa'])
    cos_snrs.snrs.meta = {}
    snrs += cos_snrs

snrs.write(path_snrs(planet, host, tst_type), overwrite=True)


#%% diagnostics

filenamer = tutils.FileNamer(tst_type, planet, host)
transit = get_transit(planet, host, tst_type)
_, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
best_snrs = load_best_snrs(planet, host, tst_type)
best_snrs.snrs = table.QTable(best_snrs.snrs)

label_case_pairs = [('median', best_snrs.median_case())]
if tst_type == 'model':
    label_case_pairs.append(('max', best_snrs.best_case()))
for label, case_snr_row in label_case_pairs:
    wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
    tutils.save_diagnostic_plots(wfig, tfig, 'max', host, filenamer)

"""well, I get the same thing with the old transit.py as the new one. I'm a little suspicious of the integration range
the code is using. It seems to be leaving out the peak of the line, maybe in favor or deeper transit where
 flux is lower, but still seems odd. going to dive into hd 63433 c. """

#%% diagnostics for hd 63433 c, gj436 b, whatever else I want

target = 'toi-776'
host = tutils.Host(target, host_catalog, planet_catalog)
planet = host.planets[1]
planet.sim_letter

transit = get_transit(planet, host, tst_type)
_, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
best_snrs = load_best_snrs(planet, host, tst_type)
best_snrs.snrs = table.QTable(best_snrs.snrs)

case_snr_row = best_snrs.best_case()
wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)


"""
hmmm well plotting the difference line has me convinced the range picker is actually doing a good job.
the error bars are growing in regions it excludes

so... back to figureing out why the snr drops with increasing lya for gj 436
"""

#%% back to gj 436

target = 'gj436'
host = tutils.Host(target, host_catalog, planet_catalog)
planet = host.planets[0]
transit = get_transit(planet, host, 'model')
snr_db = load_snr_db(planet, host, 'model')
best_snrs = load_best_snrs(planet, host, 'model')
best_snrs = best_snrs.clean_duplicates()


#%% pick and plot best snr case for gj436 with differing lya

_, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
lya0snrs = best_snrs.filtered({'lya reconstruction case':'median'})
lya1snrs = best_snrs.filtered({'lya reconstruction case':'high_1sig'})

np.median(lya0snrs['transit sigma'])
np.median(lya1snrs['transit sigma'])
np.max(lya1snrs['transit sigma'])
np.max(lya0snrs['transit sigma'])
"""
interesting, the max values are higher for higher lya. makes me think this is maybe a fluke.

I'll compute difference in sigma for same cases and try to plot the biggest one.
"""

sortcols = 'eta Tion mdot_star mass'.split()
lya0snrs.snrs.sort(sortcols)
lya1snrs.snrs.sort(sortcols)

diff = lya1snrs['transit sigma'] - lya0snrs['transit sigma']
iworst = np.argmin(diff)

case_snr_row = lya0snrs[iworst]
wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
wfig.suptitle('median lya')
tfig.suptitle('median lya')

case_snr_row = lya1snrs[iworst]
wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
wfig.suptitle('+1sig lya')
tfig.suptitle('+1sig lya')

"""
okay, this seems like a fluke. for the median case the code picks only one hump of lya
whereas for the +1 sig case it is picking both, and this seems to drop the sigma slightly
probably a result of when variability gets added in, so I'm not going to worry about it further since
it is so slight
"""

