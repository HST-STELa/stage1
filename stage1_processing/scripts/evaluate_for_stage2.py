from functools import lru_cache
import warnings

from astropy.table import Table, QTable, vstack
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

# from pathlib import Path
# import database_utilities as dbutils
#
# delivery_folder = Path('/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2025-07-30 transit predictions')
# files = list(delivery_folder.glob('*.h5'))
#
# targnames_ethan = [file.name[:-4] for file in files]
# targnames_stela = dbutils.resolve_stela_name_flexible(targnames_ethan)
# targnames_file = dbutils.target_names_stela2file(targnames_stela)
#
# for targname, file in zip(targnames_file, files):
#     planet = file.name[-4]
#     newname = f'{targname}.outflow-tail-model.na.na.transit-{planet}.h5'
#     newfolder = paths.target_data(targname) / 'transit predictions'
#     if not newfolder.exists():
#         os.mkdir(newfolder)
#     sh.copy(file, newfolder / newname)


#%% targets and code running options

targets = target_lists.eval_no(1)
mpl.use('Agg') # plots in the backgrounds so new windows don't constantly interrupt my typing
# mpl.use('qt5gg') # plots are shown
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
baseline_range = u.Quantity((obstimes[0] - 1*u.h, obstimes[1] + 1*u.h))
offsets = (0, 3)*u.h
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

# @lru_cache
def get_host_objects(name):
    host = tutils.Host(name)
    # add variability guesses based on the variability_predictor set earlier in this script
    host_variability = tutils.HostVariability(host, variability_predictor)
    return host, host_variability


#%% define function to do a nested exploration of different observational setups

def explore_snrs(
        planet: tutils.Planet,
        host: tutils.Host,
        host_variability: tutils.HostVariability,
        transit: tutils.Transit,
        transit_search_rvs: u.Quantity,
        exptime_fn
) -> QTable:

    snr_cases, snr_single = tutils.build_snr_sampler_fn(
        planet,
        host,
        host_variability,
        transit,
        exptime_fn,
        obstimes,
        baseline_range,
        normalization_search_rvs,
        transit_search_rvs
    )

    # first run two time offsets for a single aperture and lya case, pick better time, record
    print('Comparing mid-transit observation to offest(s).')
    grating = host.anticipated_grating
    base_aperture = baseline_apertures[grating]
    tbl1 = snr_cases(offsets, [(grating, base_aperture)], [0])
    offset = tutils.best_by_mean_snr(tbl1, 'time offset')
    tbl1.meta['best time offset'] = offset

    # run for all apertures, pick best aperture, record
    print('Finding the best aperture.')
    apertures = apertures_to_consider[grating]
    grating_apertures = [(grating, ap) for ap in apertures]
    tbl2 = snr_cases([offset], grating_apertures, [0])
    aperture = tutils.best_by_mean_snr(tbl2, 'aperture')
    tbl2.meta['best stis aperture'] = aperture

    # run for all lya cases
    print('Running for the plausible Lya range.')
    cases = np.arange(-2, 3)
    tbl3 = snr_cases([offset], [(grating, aperture)], cases)

    tbls = [tbl1, tbl2, tbl3]

    # optionally add cos
    lya = host.lya_reconstruction
    Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
    if Flya > cos_consideration_threshold_flux:
        print('Target Lya flux makes it eligible for COS: adding estimates for COS SNRs.')
        tbl3.meta['COS considered'] = True
        tbl4 = snr_cases([offset], [('cos', 'psa')], cases)
        tbls.append(tbl4)
    else:
        tbl3.meta['COS considered'] = False
        tbl3.meta['notes'] = 'COS not considered because flux too low.'

    def diagnostic_plot_fn(offset, grating, aperture, lya_sigma):


    return vstack(tbls), diagnostic_plot_fn


#%% loop through planets and targets and compute transit sigmas

for target in utils.printprogress(targets):
    host, host_variability = get_host_objects(target)

    for planet in host.planets:
        transit = tutils.get_transit_from_simulation(host, planet)
        snrs, daignostic_plot_fn = explore_snrs(
            planet,
            host,
            host_variability,
            transit,
            search_model_transit_within_rvs,
            exptime_fn)

        snr_tbl_name = f'{planet.dbname}.outflow-tail-model.detection-sigmas.ecsv'
        snrs.write(host.transit_folder / snr_tbl_name, overwrite=True)

        # diagnostic plots
        best_snrs = tutils.filter_to_obs_choices(snrs)
        isort = np.argsort(best_snrs['transit sigma'])
        cases = {'max': isort[-1],
                 'median': isort[len(isort) // 2]}
        for case, k in cases.items():
            tutils.make_diagnostic_plots(
                title=planet.dbname,
                outprefix=snr_tbl_name.replace('.ecsv', f'{case}-snr'),
                case_row=best_snrs[k],
                snr_fn=get_snr
            )

        # corner-like plot
        labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],σ_Lya'.split(',')
        lTion = np.log10(best_snrs['Tion'])
        Mdot = best_snrs['mdot_star']
        lMdot = np.log10(Mdot)
        eta = best_snrs['eta']
        leta = np.log10(eta)
        lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in best_snrs['lya reconstruction case']]
        param_vecs = [leta, lTion, lMdot, lya_sigma]
        snr_vec = best_snrs['transit sigma']
        cfig, _ = pu.detection_sigma_corner(param_vecs, snr_vec, labels=labels,
                                            levels=(3,), levels_kws=dict(colors='w'))
        cfig.suptitle(planet.dbname)
        cname_pdf = snr_tbl_name.replace('.ecsv', f'.plot-snr-corner.pdf')
        cname_png = snr_tbl_name.replace('.ecsv', f'.plot-snr_corner.png')
        cfig.savefig(host.transit_folder / cname_pdf)
        cfig.savefig(host.transit_folder / cname_png, dpi=300)

        plt.close('all')

#%% flat opaque tail transit

        flat_transit = tutils.construct_flat_transit(
            planet, host, obstimes, exptimes,
            rv_grid_span=(-500, 500) * u.km/u.s,
            rv_range=simple_transit_range,
            search_rvs=search_simple_transit_within_rvs
        )

        flat_snrs, get_flat_snr = explore_snrs(
            planet,
            host,
            host_variability,
            transit,
            search_simple_transit_within_rvs,
            exptime_fn
        )

        flat_name = f'{planet.dbname}.simple-opaque-tail.detection-sigmas.ecsv'
        snrs.write(host.transit_folder / flat_name, overwrite=True)

        # diagnostic plots
        best_snrs = tutils.filter_to_obs_choices(snrs)
        isort = np.argsort(best_snrs['transit sigma'])
        i_med = isort[len(isort) // 2]
        tutils.make_diagnostic_plots(
            title=planet.dbname,
            outprefix=flat_name.replace('.ecsv', 'median-snr'),
            case_row=best_snrs[i_med],
            snr_fn=get_snr
        )

        plt.close('all')


#%% make table of properties

sigma_threshold = 3
lya_bins = (-150, -50, 50, 150) * u.km/u.s

eval_rows = []
for target in targets:
    host, host_variability = get_host_objects(target)
    for i, planet in host.planets:
        row = {}
        
        row['hostname'] = host.hostname
        row['obsvtn\ngrating'] = host.anticipated_grating
        row['stellar\neff temp (K)'] = host.params['st_teff'] * preloads.col_units['st_teff']
        row['age\nlimit'] = host.age_limit
        row['age'] = host.age

        row['planet'] = planet.stela_suffix
        row['planet\nradius (Re)'] = planet['pl_rade'] * preloads.col_units['pl_rade']
        row['orbital\nperiod (d)'] = planet['pl_orbper'] * preloads.col_units['pl_orbper']

        flag_cols = [name for name in planet_catalog.colnames if 'flag_' in name]
        for col in flag_cols:
            row[col] = planet[col]

        transit = tutils.get_transit_from_simulation(host, planet)
        row['H ionztn\ntime (h)'] = (1/transit.x_params['phion']).to('h')

        sigma_tbl_path, = host.folder.rglob(f'*outflow-tail*transit-{planet.stela_suffix}*sigmas.ecsv')
        snrs = Table.read(sigma_tbl_path)
        detectable = snrs['transit sigma'] > sigma_threshold
        detectability_fraction = np.sum(detectable)/len(snrs)
        row[f'frac models\nw snr > {sigma_threshold}'] = detectability_fraction
        row['outflow model\nmax snr'] = np.max(snrs['transit sigma'])

        # aperture that yields the best average snr
        apertures = np.unique(snrs['aperture'])
        snrs.add_index('aperture')
        mean_snrs = [np.mean(snrs.loc[aperture]['transit sigma']) for aperture in apertures]
        ibest = np.argmax(mean_snrs)
        row['outflow model\nbest aperture'] = apertures[ibest]

        flat_sigma_tbl_path, = host.folder.rglob(f'*simple-opaque-tail*transit-{planet.stela_suffix}*sigmas.ecsv')
        flat_sigma_tbl = Table.read(flat_sigma_tbl_path)
        snr = np.max(flat_sigma_tbl['transit sigma'])
        colname = f'flat transit snr\n[{simple_transit_range[0].value:.0f}, {simple_transit_range[1].value:.0f}]'
        row[colname] = snr
        ibest = np.argmax(flat_sigma_tbl['transit sigma'])
        aperture = flat_sigma_tbl['aperture'][ibest]
        row['flat transit\nbest aperture'] = aperture

        eval_rows.append(row)

eval_table = Table(rows=eval_rows)

column_order_and_names = {
    'hostname' : '',
    'planet' : '',
    f'frac models\nw snr > {sigma_threshold}' : '',
    'outflow model\nmax snr' : '',
    f'flat transit snr\n[-150, 100]' : '',
    'lya recnstcnt flag' : '',
    'planet\nradius (Re)': '',
    'H ionztn\ntime (h)': '',
    'orbital\nperiod (d)': '',
    'stellar\neff temp (K)': '',
    'age\nlimit': '',
    'age': '',
    'flag_measured_mass': 'planet has\nmsrd mass',
    'flag_high_TSM': 'high\nTSM',
    'flag_gaseous': 'gaseous',
    'flag_gas_and_rocky_in_sys': 'system has\ngaseous & rocky',
    'flag_water_world': 'water\nworld',
    'flag_gap_upper_cusp': 'on upper cusp\nof radius valley',
    'flag_super_puff': 'super\npuff',
    'lya flux\n[-150, -50]' : '',
    'lya flux\n[-150, -50] +err' : '',
    'lya flux\n[-150, -50] -err' : '',
    'lya flux\n[-50, 50]' : '',
    'lya flux\n[-50, 50] +err' : '',
    'lya flux\n[-50, 50] -err' : '',
    'lya flux\n[50, 150]' : '',
    'lya flux\n[50, 150] +err' : '',
    'lya flux\n[50, 150] -err' : '',
    'lya flux\ncore' : '',
    'lya flux\ncore +err' : '',
    'lya flux\ncore -err' : '',
    'obsvtn\ngrating' : '',
    'outflow model\nbest aperture' : '',
    'flat transit\nbest aperture' : ''
}
eval_table = eval_table[list(column_order_and_names.keys())]
for oldcol, newcol in column_order_and_names.items():
    if newcol != '':
        eval_table.rename_column(oldcol, newcol)

# some grooming
for col in eval_table.colnames:
    if 'flag_' in col:
        eval_table[col] = eval_table[col].astype(bool)

# save
eval_filename = 'stage2_evalution_metrics.csv'
eval_path = paths.catalogs / eval_filename
eval_table.write(eval_path, overwrite=True)

eval_table_ecsv = eval_table.copy()
for name in eval_table_ecsv.colnames:
    eval_table_ecsv.rename_column(name, name.replace('\n', ' '))
eval_path_ecsv = paths.catalogs / eval_filename.replace('csv', 'ecsv')
eval_table_ecsv.write(eval_path_ecsv, overwrite=True)
