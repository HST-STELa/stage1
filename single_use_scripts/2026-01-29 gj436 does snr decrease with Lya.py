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


#%% lightcurve comparison

"""
just making a note here that to do the lightcurve comparison I had to checkout old commits
for both the database and codebase so that I could load up and plot the pre-bugfix sim
so it doesn't lend itself to a nice clean script. hence no saved script for that.
"""


#%% load snr db for 10 h offset

snr_db = load_snr_db(planet, host, 'model')
snrs_10 = snr_db.filtered({'time offset': 10 * u.h})
snrs_10.snrs = table.QTable(snrs_10.snrs)


#%% corner plot for 10 h offset

labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],log10(M_planet)\n[Mearth]'.split(',')

# construct parameter vectors
lTion = snrs_10['Tion'].quantity.to_value('dex(h)')
lMdot = snrs_10['mdot_star'].quantity.to_value('dex(g s-1)')
leta = np.log10(snrs_10['eta'].data)
lMp = snrs_10['mass'].quantity.to_value('dex(Mearth)')
lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in snrs_10['lya reconstruction case']]
param_vecs = [leta, lTion, lMdot, lMp]

snr_vec = snrs_10['transit sigma']

cfig, _ = pu.detection_volume_corner(param_vecs, snr_vec, snr_threshold=sigma_threshold, labels=labels)
cfig.suptitle(planet.dbname)
# utils.save_pdf_png(cfig, host.transit_folder / filenamer.det_vol_corner_basename)

cfig, _ = pu.median_snr_corner(param_vecs, snr_vec, labels=labels)
cfig.suptitle(planet.dbname)
# utils.save_pdf_png(cfig, host.transit_folder / filenamer.mdn_snr_corner_basename)
pass


#%% diagnostic plot for best case 10 h offset

tst_type = 'model'
transit = get_transit(planet, host, tst_type)
_, get_snr = consrtuct_snr_samplers(host, transit, tst_type)

case_snr_row = snrs_10.best_case()
viewcols = [
    'transit sigma',
    'eta',
    'mdot_star',
    'Tion',
    'mass',
    'time offset',
    'grating',
    'aperture',
    'lya reconstruction case'
]
case_snr_row[viewcols]

keycols = [
    'eta',
    'mdot_star',
    'Tion',
    'mass'
]
tt = transit.loc_transmission(**dict(case_snr_row[keycols]), rtol=0.01)
v = lya.w2v(transit.wavegrid) - host.params['st_radv'].to_value('km s-1')
# by inspecting the later plot I see the integration band spans -95–-45
# so I just found the corresponding range of indices
plt.figure()
plt.plot(transit.timegrid, tt[:,53:77], color='C0', alpha=0.3)
plt.xlabel('time')

wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
# tutils.save_diagnostic_plots(wfig, tfig, 'max', host, filenamer)
pass

#%% diagnostic plots for other offset, otherwise same params

from copy import copy
off_row = copy(case_snr_row)

for offset in [0, 2, 5, 10]:
    # snrs_off = snr_db.filtered({'time offset': offset * u.h})
    # snrs_off.snrs = table.QTable(snrs_off.snrs)
    off_row['time offset'] = offset*u.h
    wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, off_row)
    offlbl = f'time offset = {offset} h'
    wfig.suptitle(offlbl)
    tfig.suptitle(offlbl)
    wfig.savefig(scratchfolder / f'diagnostic spectrum {offlbl}.png', dpi=300)
    tfig.savefig(scratchfolder / f'diagnostic lightcurve {offlbl}.png', dpi=300)