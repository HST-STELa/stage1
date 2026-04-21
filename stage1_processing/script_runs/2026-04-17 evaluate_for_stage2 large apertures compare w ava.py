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


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = [
    'hd63935',
    # 'l98-59',
    # 'ltt1445a',
    # 'toi-2443',
    # 'toi431',
]

tst_types = ('model',)

# mpl.use('Agg') # plots in the background so new windows don't constantly interrupt my typing
mpl.use('qt5agg') # plots are shown

staging_area1 = paths.packages / '2025-06-16.stage2.eval1.staging_area'
staging_area2 = paths.packages / '2025-09-26.stage2.eval2.staging_area'

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
    planet_catalog1 = catutils.load_and_mask_ecsv(staging_area1 / 'planet_catalog.ecsv')
    planet_catalog2 = catutils.load_and_mask_ecsv(staging_area2 / 'planet_catalog.ecsv')
    add_to_list = ~np.isin(planet_catalog1['id'], planet_catalog2['id'])
    planet_catalog = table.vstack((planet_catalog1[add_to_list], planet_catalog2))
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


#%% function to make tables and diagnostic plots

build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration
corner_lbls = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],log10(M_planet)\n[Mearth],σ_Lya'.split(',')

def analyze(target, planet, tst_type, plots=False):
    host = tutils.Host(target, host_catalog, planet_catalog)
    grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)

    results = {}

    transit = get_transit(planet, host, tst_type)
    get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
    def build_planet_snrs(grating, base_aperture, all_apertures):
        snrs = build_snrs(get_snr_iterable, grating, base_aperture, all_apertures,
                          offsets, max_safe_offset, verbose=True)
        return snrs

    snrs = build_planet_snrs(grating, base_aperture, all_apertures)
    # optionally add COS
    snrs.meta['COS considered'] = consider_cos
    if consider_cos:
        cos_snrs = build_planet_snrs('g130m', 'psa', ['psa'])
        cos_snrs.snrs.meta = {}
        snrs += cos_snrs

    results['snrtbl'] = snrs

    snrs.write(path_snrs(planet, host, tst_type), overwrite=True)

    if plots:
        filenamer = tutils.FileNamer(tst_type, planet, host)
        transit = get_transit(planet, host, tst_type)
        _, get_snr = consrtuct_snr_samplers(host, transit, tst_type)

        best_snrs = load_best_snrs(planet, host, tst_type)
        best_snrs.snrs = table.QTable(best_snrs.snrs)

        label_case_pairs = [('median', best_snrs.median_case())]
        if tst_type == 'model':
            label_case_pairs.append(('max', best_snrs.best_case()))
        for label, case_snr_row in label_case_pairs:
            wkey = f'plot wave {case_snr_row}'
            tkey = f'plot time {case_snr_row}'
            wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
            results[wkey] = wfig
            results[tkey] = tfig
            tutils.save_diagnostic_plots(wfig, tfig, 'max', host, filenamer)

        if tst_type == 'model':

            # construct parameter vectors
            lTion = best_snrs['Tion'].quantity.to_value('dex(h)')
            lMdot = best_snrs['mdot_star'].quantity.to_value('dex(g s-1)')
            leta = np.log10(best_snrs['eta'].data)
            lMp = best_snrs['mass'].quantity.to_value('dex(Mearth)')
            lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in best_snrs['lya reconstruction case']]
            param_vecs = [leta, lTion, lMdot, lMp, lya_sigma]

            snr_vec = best_snrs['transit sigma']

            cfig, _ = pu.detection_volume_corner(param_vecs, snr_vec, snr_threshold=sigma_threshold, labels=corner_lbls)
            cfig.suptitle(planet.dbname)
            utils.save_pdf_png(cfig, host.transit_folder / filenamer.det_vol_corner_basename)
            results['plot corner vol'] = cfig

            cfig, _ = pu.median_snr_corner(param_vecs, snr_vec, labels=corner_lbls)
            cfig.suptitle(planet.dbname)
            utils.save_pdf_png(cfig, host.transit_folder / filenamer.mdn_snr_corner_basename)
            results['plot corner snr'] = cfig

    return results


#%% 2026-04-17

"""
ran this for hd63935 and got results different than before
"""

#%% 2026-04-20 notes git checkout earlier database

"""
checked out git database just after I stopped editing (2/21) the final eval sheet to see if best aps match
checked out c9ef68187a83297f3660990ead80b5f4b5a2241d
but I probably didn't need to do this because I can already see that the snr tables are dated 2/17
"""

#%% checking if best aps are what I put in sheet

targets = ['hd63935', 'l98-59', 'ltt1445a', 'toi-2443', 'toi-431']
for target in targets:
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        for tst_type in ('model', 'flat'):
            snrs = load_snr_db(planet, host, tst_type)

            if "best base grating aperture" not in snrs.meta:
                print(f'{planet.dbname} {tst_type}: no best base grating aperture')
                continue
            print(f'{planet.dbname} {tst_type}: {snrs.meta["best base grating aperture"]}')

            if "best time offset" not in snrs.meta:
                print(f'{planet.dbname} {tst_type}: no best time offset')
                continue
            print(f'{planet.dbname} {tst_type}: {snrs.meta["best time offset"]}')

"""
only hd63935 c seems to differ from what is recorded in its snr table, I tracked the change to 2/9 in the eval
google sheet but I can't see exactly what I was doing that changed it, so it's not very helpful 
"""

#%% rerunning the 0.5" cases

tpsets = (
    ('hd63935', 1),
    ('l98-59', 1),
    ('l98-59', 2),
    ('ltt1445a', 1),
    ('toi-2443', 0),
    ('toi-431', 1),
)

snr_sets = {}
for target, planet_index in tpsets:
    host = tutils.Host(target, host_catalog, planet_catalog)
    planet = host.planets[planet_index]
    for tst_type in ('model', 'flat'):
        snr_sets[planet.dbname] = {}
        results = analyze(target, planet, tst_type, plots=False)
        snrs = results['snrtbl']
        best_ap = snrs.meta['best base grating aperture']
        best_offset = snrs.meta['best safe time offset']
        print(f'{target} {planet.dbname} {tst_type}: {best_ap} {best_offset}')
        snr_sets[planet.dbname][tst_type] = snrs
        
#%% results

"""
NEW 
hd63935-c  2 52x0.2 52x0.05
l98-59-c   1 52x0.2 52x0.5
l98-59-d   1 52x0.2 52x0.05
ltt1445a-c 0 52x0.5 52x0.2
toi-2443-b 0 52x0.5 52x0.05
toi-431-d  2 52x0.2 52x0.05

1/29 db checkout
hd63935-c	8 52x0.1 52x0.05
l98-59-c	1 52x0.2 52x0.5
l98-59-d	2 52x0.1 52x0.05
ltt1445a-c	1 52x0.1 52x0.2
toi-2443-b	0 52x0.5 52x0.05
toi-431-d	0 52x0.2 52x0.05

FROM AVA
HD 63935	c	3.0	52x0.1	52x0.05
L 98-59	    c	1.0	52x0.2	52x0.2
L 98-59	    d	1.0	52x0.2	52x0.05
LTT 1445 A	c	0.0	52x0.5	52x0.2
TOI-2443	b	0.0	52x0.5	52x0.05
TOI-431	    d	0.0	52x0.05	52x0.05

IN EVAL TABLE
offset sim flat
HD 63935 c   1 52x0.5 52x0.05
L 98-59 c    1 52x0.5 52x0.5
L 98-59 d    0 52x0.5 52x0.5
LTT 1445 A c 1 52x0.5 52x0.5
TOI-2443 b   0 52x0.5 52x0.5
TOI-431 d    0 52x0.5 52x0.5

I notice Ava and I differ in aperture where we differ in best safe. 
I think that is because the best aperture is evaluated at best safe. 
"""

#%% note from digging around in code history

"""
12/19 I implemnted the DetectabilityDatabase class which defined the snr exploration method 
and best aperture based on mean snr

1/29 I updated that to use detectability volume instead of mean snr

I notice it is set to using a default snr threshold of 3 instead of 1, but I do have it so that the main
builder function properly uses whatever snr threshold has been provided. I do wonder though if Ava and I 
might be handling this differently.
"""

#%% checked out database from 2/5 and reran

"d6244208f868d062f12706d0e019161e449a0962 updated with latest sims"

"""
results
"""

#%% checked out database from 1/29 and reran

"eb39086854107c66dee124cc43159d4aee04466a rerun done"

"""
results

"""


#%% computing the ratio of detectability volumes using different apertures again

def compare_vol_by_aperture(target, planet_index, comp_aperture='best', confine_lya=False):
    tst_type = 'model'
    host = tutils.Host(target, host_catalog, planet_catalog)
    planet = host.planets[planet_index]
    transit = get_transit(planet, host, tst_type)
    _, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
    snrs = load_snr_db(planet, host, tst_type)

    best_off = snrs.meta['best safe time offset']
    grating = snrs.meta['base grating']
    configs = {'time offset': best_off,
               'grating': grating}
    if confine_lya:
        configs['lya reconstruction case'] = 'median'
    ap_test_snrs = snrs.filtered(configs)
    ap_test_snrs = ap_test_snrs.clean_duplicates()

    snrs_pt2 = ap_test_snrs.filtered({'aperture': '52x0.2'})
    det_vol_pt2 = snrs_pt2.detection_fraction(sigma_threshold)

    if comp_aperture == 'best':
        comp_ap = snrs.meta['best base grating aperture']
    else:
        comp_ap = comp_aperture
    snrs_comp_ap = ap_test_snrs.filtered({'aperture': comp_ap})
    det_vol_comp_ap = snrs_comp_ap.detection_fraction(sigma_threshold)

    print('')
    print(f'{planet.dbname}')
    print(f'Det vol with 52x0.2 aperture: {det_vol_pt2:.3f}')
    print(f'Det vol with {comp_ap} aperture: {det_vol_comp_ap:.3f}')

tpsets = (
    ('hd63935', 1),
    ('l98-59', 1),
    ('l98-59', 2),
    ('ltt1445a', 1),
    ('toi-2443', 0),
    ('toi-431', 1),
)

for target, pidx in tpsets:
    compare_vol_by_aperture(target, pidx)

#%% with lya confined to median case

for target, pidx in tpsets:
    compare_vol_by_aperture(target, pidx, confine_lya=True)

#%% how big a deal is using 52x0.5 for l98-59 c

compare_vol_by_aperture('l98-59', 1, '52x0.5')

#%% comparing 0.2 to 0.5 for all cases

for target, pidx in tpsets:
    compare_vol_by_aperture(target, pidx, '52x0.5', confine_lya=True)