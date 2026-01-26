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

targets_all = set(target_lists.eval_no(1)) | set(target_lists.eval_no(2)) - {'v1298tau'}
targets_redo = 'ltt1445a toi-1203 toi-561'.split()

sigma_threshold = 3
min_samples = 5**4 # used as a check later to ensure all grid pts of Ethan's sims were sampled

mpl.use('Agg') # plots in the background so new windows don't constantly interrupt my typing
# mpl.use('qt5agg') # plots are shown
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


#%% loop through planets of all hosts to compute transit sigmas

targets = targets_redo

build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration

for target in utils.printprogress(targets, prefix='host '):
    host = tutils.Host(target, host_catalog, planet_catalog)
    grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)
    for planet in utils.printprogress(host.planets, 'dbname', prefix='\tplanet '):
        for tst_type in ('model', 'flat'):
            transit = get_transit(planet, host, tst_type)
            get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
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

#%% loop through planets of all hosts to compute transit sigmas

targets = targets_all

build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration

for target in utils.printprogress(targets, prefix='host '):
    host = tutils.Host(target, host_catalog, planet_catalog)
    grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)
    for planet in utils.printprogress(host.planets, 'dbname', prefix='\tplanet '):
        for tst_type in ('flat',):
            transit = get_transit(planet, host, tst_type)
            get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
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



#%% redo planets that gave errors of too few samples or other problems

pass
# targets = ['hd63433', 'l98-59', 'gj9827']
# targets = ['l98-59', 'gj9827']
# targets = ['gj9827']
targets = ['gj436']
# targets = ['gj3470']

build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration

for target in utils.printprogress(targets, prefix='host '):
    host = tutils.Host(target, host_catalog, planet_catalog)
    grating, base_aperture, all_apertures, consider_cos = get_obs_config_info(host)
    for planet in utils.printprogress(host.planets, 'dbname', prefix='\tplanet '):
        for tst_type in ('flat', 'model'):
            transit = get_transit(planet, host, tst_type)
            get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)
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


#%% make diagnostic plots

pass
# targets = targets_all
# targets = ['hd63433', 'l98-59', 'gj9827']
targets = ['gj3470', 'gj436']

for target in tqdm(targets):
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        for tst_type in ('model', 'flat'):
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

            plt.close('all')


#%% make corner plots

labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],log10(M_planet)\n[Mearth],σ_Lya'.split(',')

# targets = targets_all
# targets = ['hd63433', 'l98-59', 'gj9827']
targets = ['gj3470', 'gj436']

for target in tqdm(targets):
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        best_snrs = load_best_snrs(planet, host,'model')
        filenamer = tutils.FileNamer('model', planet, host)

        # construct parameter vectors
        lTion = best_snrs['Tion'].quantity.to_value('dex(h)')
        lMdot = best_snrs['mdot_star'].quantity.to_value('dex(g s-1)')
        leta = np.log10(best_snrs['eta'].data)
        lMp = best_snrs['mass'].quantity.to_value('dex(Mearth)')
        lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in best_snrs['lya reconstruction case']]
        param_vecs = [leta, lTion, lMdot, lMp, lya_sigma]

        snr_vec = best_snrs['transit sigma']

        cfig, _ = pu.detection_volume_corner(param_vecs, snr_vec, snr_threshold=sigma_threshold, labels=labels)
        cfig.suptitle(planet.dbname)
        utils.save_pdf_png(cfig, host.transit_folder / filenamer.det_vol_corner_basename)

        cfig, _ = pu.median_snr_corner(param_vecs, snr_vec, labels=labels)
        cfig.suptitle(planet.dbname)
        utils.save_pdf_png(cfig, host.transit_folder / filenamer.mdn_snr_corner_basename)

        plt.close('all')


#%% assemble table of properties

targets = targets_all

lya_bins = (-150, -50, 50, 150) * u.km/u.s

def fix_meta(snr_db):
    snr_db.meta['best time offset'], = np.unique(snr_db['best time offset'].quantity)
    snr_db.meta['best safe time offset'], = np.unique(snr_db['best safe time offset'].quantity)

toofewsamples = []
eval_rows = []
for target in tqdm(targets):
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        # add entries to the row in the order they should appear in the table
        row = {}
        
        row['hostname'] = host.hostname
        row['planet'] = planet.stela_suffix

        # region model snrs
        snr_db = load_snr_db(planet, host, 'model')
        if 'best time offset' not in snr_db.meta:
            fix_meta(snr_db)
        try:
            slctd_offsets, det_fracs, max_snrs = snr_db.offset_stats(sigma_threshold, min_sample_check=5**4)
        except ValueError as e:
            if 'Num of samples' in str(e):
                toofewsamples.append(target)
                # continue
                break

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
        if 'best time offset' not in snr_db_flat.meta:
            fix_meta(snr_db_flat)
        snr_db_flat = snr_db_flat.filter_obs_config(aperture='best', offset='best')
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

with catutils.catch_QTable_unit_warnings():
    more_planets = preloads.planets.copy()

det_ids = np.char.add(escape_detections['Target Star'], escape_detections['Planet Letter'])
pcat_ids = np.char.add(more_planets['hostname'].astype(str),
                       dbutils.planet_suffixes(more_planets).astype(str))
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


#%% load table and make labels

if plot_from_excel:
    eval_table = catutils.read_excel(url_google_sheet)
else:
    eval_path_ecsv = paths.catalogs / eval_filename.replace('csv', 'ecsv')
    eval_table = catutils.load_and_mask_ecsv(eval_path_ecsv)
    for name in eval_table.colnames:
        eval_table.rename_column(name, name.replace('  ', ' '))

labels = np.char.add(eval_table['hostname'], ' ')
labels = np.char.add(labels, eval_table['planet'])
eval_table['labels'] = labels

#%% make some masks and some global style settings

slide_size = [9, 5]
mpl.rcParams['savefig.dpi'] = 300

masks = dict(
    allpts=np.ones(len(eval_table), dtype=bool),
    detections=eval_table['Lya detected?'] == 'Y',
    tentative=eval_table['Lya detected?'] == 'T',
    nondetections=eval_table['Lya detected?'] == 'ND',
    unknown=eval_table['Lya detected?'] == 'UK'
)

pt_styles = dict(
    allpts=dict(color='0.5', label='all'),
    detections=dict(color='C2', zorder=10, label='known detections'),
    tentative=dict(color='C1', label='known tentative'),
    nondetections=dict(color='C3', label='known nondetections'),
    unknown=dict(color='C4', label='(to be) observed, result unknown')
)


#%% plot detection fractions vs lya snr

def make_detectability_plot(ykey, yfloor=0):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xscale('log')
    ax.set_xlabel('Lya Flux (erg s-1 cm-2)')
    ax.set_yscale('log')

    x = eval_table['Lya Flux (erg s-1 cm-2)']
    y = eval_table[ykey]
    y = np.clip(y, yfloor, np.inf)

    def plot_set(mask, *args, addlbls=False, **kws):
        xx = x[mask]
        yy = y[mask]
        ln, = ax.plot(xx, yy, 'o', *args, **kws)
        if addlbls:
            lbls = eval_table['labels'][mask]
            for _x,_y,lbl in zip(xx, yy, lbls):
                ax.annotate(' ' + lbl, xy=(_x,_y), fontsize='xx-small', rotation=45)
        return ln

    plot_set(masks['allpts'], addlbls=False, **pt_styles['allpts'])
    plot_set(masks['detections'], addlbls=True,  **pt_styles['detections'])
    plot_set(masks['tentative'], addlbls=True, **pt_styles['tentative'])
    plot_set(masks['nondetections'], addlbls=True, **pt_styles['nondetections'])
    plot_set(masks['unknown'], addlbls=True, **pt_styles['unknown'])

    plt.legend(fontsize='small')

    return fig, ax

date = dbutils.timestamp(date_only=True)

frac_fig, frac_ax = make_detectability_plot('sim safe offset frac w snr > 3', 1e-4)
frac_ax.set_ylabel('Simulation Detectability Fraction')
frac_fig.tight_layout()
frac_fig.savefig(paths.scratch / f'{date} detectability fraction vs lya flux plot.png')

flat_fig, flat_ax = make_detectability_plot('flat transit snr', 0)
flat_ax.set_ylabel('Simple Flat Transit SNR')
flat_fig.tight_layout()
flat_fig.savefig(paths.scratch / f'{date} flat transit SNR vs lya flux plot.png')


#%% load in confirmed table

all_confirmed_path = paths.ipac / 'confirmed_uncut.ecsv'
confirmed = table.Table.read(all_confirmed_path)


#%% plot detection fractions on Teq-Rad plot

def make_scatter(skey, scalefunc=None):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xlabel('Orbital Period (d)')
    ax.set_ylabel('Radius ($R_\\oplus$)')


    s = eval_table[skey]
    if scalefunc is not None:
        s = scalefunc(s)

    high_precision = confirmed['pl_radeerr1']/confirmed['pl_rade'] < 0.1

    ax.loglog(
        confirmed['pl_orbper'][high_precision],
        confirmed['pl_rade'][high_precision],
        '.', alpha=0.2, mew=0, color='C5'
    )

    def plot_set(mask, addlbls=False, **kws):
        ln = ax.scatter(
            eval_table['orbital period (d)'][mask],
            eval_table['radius (Re)'][mask],
            s=s[mask],
            **kws
        )
        return ln

    plot_set(masks['allpts'], addlbls=False, **pt_styles['allpts'])
    plot_set(masks['detections'], addlbls=True, **pt_styles['detections'])
    plot_set(masks['tentative'], addlbls=True, **pt_styles['tentative'])
    plot_set(masks['nondetections'], addlbls=True, **pt_styles['nondetections'])
    plot_set(masks['unknown'], addlbls=True, **pt_styles['unknown'])

    ax.set_xlim(0.3, 200)
    ax.set_ylim(0.3, 30)

    plt.legend(fontsize='small')

    return fig, ax

date = dbutils.timestamp(date_only=True)

def frac_scale(s):
    s = np.clip(s, 1e-4, np.inf)
    s = np.cbrt(s)*100
    return s
frac_fig_scat, frac_ax_scat = make_scatter('sim safe offset frac w snr > 3', frac_scale)
frac_ax_scat.set_title('Simulation Detectability Fraction')
frac_fig_scat.tight_layout()
frac_fig_scat.savefig(paths.scratch / f'{date} detectability fraction population.png')

def flat_scale(s):
    s = s*5
    return s
flat_fig_scat, flat_ax_scat = make_scatter('flat transit snr', flat_scale)
flat_ax_scat.set_title('Flat Transit SNR')
flat_fig_scat.tight_layout()
flat_fig_scat.savefig(paths.scratch / f'{date} flat transit SNR population.png')