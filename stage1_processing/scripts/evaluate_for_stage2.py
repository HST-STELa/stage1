import warnings
from functools import lru_cache

from astropy.table import Table, Column, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import h5py
from matplotlib import pyplot as plt
import matplotlib as mpl

import empirical
import hst_utilities
import paths
import utilities as utils
import catalog_utilities as catutils
import database_utilities as dbutils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu
from stage1_processing import transit_evaluation_utilities as tutils

from lya_prediction_tools import variability
from lya_prediction_tools import transit
from lya_prediction_tools import stis
from lya_prediction_tools import cos
from lya_prediction_tools import lya
from lya_prediction_tools import ism
from lya_prediction_tools.spectrograph import Spectrograph


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


#%% assumed variability

rotation_amplitude_saturation = 0.25
rotation_amplitude_Ro1 = 0.05
jitter_saturation = 0.1
jitter_Ro1 = 0.01
Ro_break = 0.1
def satdecay_jitter(Ro):
    return variability.saturation_decay_loglog(
        Ro, Ro_break, jitter_saturation, 1, jitter_Ro1).to_value('')
def satdecay_rotation(Ro):
    return variability.saturation_decay_loglog(
        Ro, Ro_break, rotation_amplitude_saturation, 1, rotation_amplitude_Ro1).to_value('')


#%% tables

stela_name_tbl = preloads.stela_names.copy()
planet_catalog = preloads.planets.copy()
planet_catalog.add_index('tic_id')
host_catalog = preloads.hosts
host_catalog.add_index('tic_id')


#%% caching and thin closures

@lru_cache
def get_host_object(name):
    host = tutils.Host(name)

    # add variability based on assumptions set earlier
    # activity guesses
    if host.params['st_teff'] > 3500:
        Ro = variability.rossby_number(Mstar, Prot)
        jitter_star = satdecay_jitter(Ro)
        Arot = satdecay_rotation(Ro)
    else:  # assume mid-late Ms are always quite variable
        jitter_star = jitter_saturation
        Arot = rotation_amplitude_saturation
    host.jitter = jitter_star
    host.rotation_amplitude = Arot

    return host

#%% define what observational setups to explore

def explore_snrs(
        planet: tutils.Planet,
        host: tutils.Host,
        transit: tutils.Transit,
        exptime_fn
) -> Table:

    get_snr = tutils.build_snr_sampler_fn(
        planet,
        host,
        transit,
        exptime_fn,
        obstimes,
        baseline_range,
        normalization_search_rvs,
    )

    # first run two time offsets for a single aperture and lya case, pick better time, record
    grating = host.anticipated_grating
    base_aperture = baseline_apertures[grating]
    tbl1 = get_snr(offsets, [(grating, base_aperture)], [0])
    offset = tutils.best_by_mean_snr(tbl1, 'time offset')
    tbl1.meta['best time offset'] = offset

    # run for all apertures, pick best aperture, record
    apertures = apertures_to_consider[grating]
    grating_apertures = [(grating, ap) for ap in apertures]
    tbl2 = get_snr([offset], grating_apertures, [0])
    aperture = tutils.best_by_mean_snr(tbl2, 'aperture')
    tbl2.meta['best stis aperture'] = aperture

    # run for all lya cases
    cases = np.arange(-2, 3)
    tbl3 = get_snr([offset], [(grating, aperture)], cases)

    tbls = (tbl1, tbl2, tbl3)

    # optionally add cos
    Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
    if Flya > cos_consideration_threshold_flux:
        tbl4 = get_snr([offset], [('cos', 'psa')], cases)
        tbls.append(tbl4)
    else:
        tbl3.meta['notes'] = 'COS not considered because flux too low.'

    return catutils.flexible_table_vstack(tbls), get_snr



#%% loop through planets and targets and compute transit sigmas

for target in targets:
    host = get_host_object(target)

    for planet in host.planets:
        transit = tutils.get_transit_from_simulation(host, planet)
        snrs, get_snr = explore_snrs(planet, host, transit, exptime_fn)

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
                outprefix=snr_tbl_name.strip('.ecsv'),
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
            cfig.suptitle(title)
            cname_pdf = snr_tbl_name.replace('.ecsv', f'.plot-snr-corner.pdf')
            cname_png = snr_tbl_name.replace('.ecsv', f'.plot-snr_corner.png')
            cfig.savefig(host.transit_folder / cname_pdf)
            cfig.savefig(host.transit_folder / cname_png, dpi=300)

            plt.close('all')


#%% flat opaque tail transit

        try:
            ta = (obstimes[0] - exptimes[0]/2).to_value('h')
            tb = (obstimes[-1] + exptimes[-1]/2).to_value('h')
            dt = 0.25
            flat_tgrid = np.arange(ta, tb + dt, dt)
            flat_transit_timemask = utils.is_in_range(flat_tgrid, *in_transit_range.to_value('h'))

            flat_wgrid = np.arange(*spec.wavegrid[[0,-1]], spec.binwidth[0]/3)
            flat_vgrid = lya.w2v(flat_wgrid) - rv_star.to_value('km s-1')

            flat_depth, = transit.opaque_tail_depth(planets[[i]]) # takes a table as input, hence the [[i]]

            flat_transit_wavemask = utils.is_in_range(flat_vgrid, *simple_transit_range.to_value('km s-1'))
            flat_transmission = np.ones((len(flat_tgrid), len(flat_wgrid)))
            flat_transmission[np.ix_(flat_transit_timemask, flat_transit_wavemask)] = 1 - flat_depth

            lya_case = 'median'
            flat_lya_flux = lya_recon[f'lya_model unconvolved_{lya_case}']

            # get detection sigmas for a range of apertures
            flat_tbls = []
            for grating, aperture in grating_aperture_combos:
                spec = get_spectrograph_object(grating, aperture, host)
                jitter = star_plus_breathing_jitter(aperture)
                exptimes_mod = exptime_w_peakups(aperture)

                flat_sigma_tbl = transit.generic_transit_snr(
                    exptimes=exptimes_mod,
                    transit_timegrid = flat_tgrid,
                    transit_wavegrid = flat_wgrid,
                    transit_transmission_ary = flat_transmission,
                    lya_recon_flux_ary = flat_lya_flux,
                    transit_search_rvs= search_simple_transit_within_rvs,
                    spectrograph_object=spec,
                    jitter=jitter,
                    **global_transit_kws,
                    **target_transit_kws,
                    **planet_transit_kws,
                )

                # store flat model parameters
                flat_sigma_tbl['aperture'] = aperture
                flat_sigma_tbl['injected times'] = [in_transit_range.to_value('h')]
                flat_sigma_tbl['injected rvs'] = [search_simple_transit_within_rvs.to_value('km s-1')]
                flat_sigma_tbl['depth'] = flat_depth
                flat_sigma_tbl['lya reconstruction level'] = lya_case

                flat_tbls.append(flat_sigma_tbl)

            flat_sigma_tbl = catutils.flexible_table_vstack(flat_tbls)

            flat_filename = f'{target}.simple-opaque-tail.na.na.transit-{letter}.{grating}-detection-sigmas.ecsv'
            flat_path = targ_transit_folder / flat_filename
            flat_sigma_tbl.write(flat_path, overwrite=True)

            # diagnostic plots
            flat_sigma_tbl_antpd = flat_sigma_tbl[snrs['grating'] == anticipated_grating]
            k_max = np.argmax(flat_sigma_tbl['transit sigma'])
            aperture = flat_sigma_tbl_antpd['aperture'][k_max]
            transmission = flat_transmission
            spec = get_spectrograph_object(grating, aperture, host)
            jitter = star_plus_breathing_jitter(aperture)
            exptimes_mod = exptime_w_peakups(aperture)

            _, wfigs, tfigs = transit.generic_transit_snr(
                exptimes = exptimes_mod,
                transit_timegrid=flat_tgrid,
                transit_wavegrid=flat_wgrid,
                transit_transmission_ary=transmission,
                lya_recon_flux_ary=lya_flux,
                transit_search_rvs=search_model_transit_within_rvs,
                spectrograph_object=spec,
                jitter=jitter,
                diagnostic_plots=True,
                **global_transit_kws,
                **target_transit_kws,
                **planet_transit_kws,
            )

            title = f'{hostname} {letter}'
            wfig, = wfigs
            wfig.tight_layout()
            wfig.suptitle(title)
            wname_pdf = flat_filename.replace('.ecsv', f'.plot-spectra-{case}-snr.pdf')
            wname_png = flat_filename.replace('.ecsv', f'.plot-spectra-{case}-snr.png')
            wfig.savefig(targ_transit_folder / wname_pdf)
            wfig.savefig(targ_transit_folder / wname_png, dpi=300)

            tfig, = tfigs
            tfig.tight_layout()
            tfig.suptitle(title)
            tname_pdf = flat_filename.replace('.ecsv', f'.plot-lightcurve-{case}-snr.pdf')
            tname_png = flat_filename.replace('.ecsv', f'.plot-lightcurve-{case}-snr.png')
            tfig.savefig(targ_transit_folder / tname_pdf)
            tfig.savefig(targ_transit_folder / tname_png, dpi=300)

            plt.close('all')

        except Exception as e:
            missed_planets_flat.append(planet)
            missed_targets_flat.append(target)
            exceptions_flat.append(e)
            raise

    utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    mpl.use('qt5agg')
    break


#%% targets

targets = target_lists.eval_no(1)


#%% make table of properties


lya_recon_flag_txt = """| HD 5278   | LOW SIGNAL  |
| HD 86226   | LOW SIGNAL  |
| HD 149026   | LOW SIGNAL  |
| K2-136   | MEDIOCRE FIT  |
| LHS 1140   | LOW SIGNAL  |
| TOI-178   | BAD AIRGLOW SUBTRACTION  |
| TOI-561   | BAD AIRGLOW SUBTRACTION  |
| TOI-836   | LOW SIGNAL  |
| TOI-1201   | BAD AIRGLOW SUBTRACTION  |
| TOI-1203   | BAD AIRGLOW SUBTRACTION  |
| TOI-1224   | LOW SIGNAL  |
| TOI-1231   | BAD AIRGLOW SUBTRACTION  |
| TOI-2015   | LOW SIGNAL  |
| TOI-2285   | BAD AIRGLOW SUBTRACTION  |
| TOI-4438   | BAD AIRGLOW SUBTRACTION  |
| TOI-4576   | BAD AIRGLOW SUBTRACTION  |
| TOI-6078   | BAD AIRGLOW SUBTRACTION  |
| WASP-107   | BAD AIRGLOW SUBTRACTION  |
| Wolf 503   | LOW SIGNAL  |"""
lines = lya_recon_flag_txt.split('\n')
pairs = [line[2:-3].split('   | ') for line in lines]
recon_flag_dict = dict(pairs)

sigma_threshold = 3
lya_bins = (-150, -50, 50, 150) * u.km/u.s

eval_rows = []
for target in targets:
    host_row = {}
    tic_id, hostname = stela_name_tbl.loc['hostname_file', target][['tic_id', 'hostname']]
    host_row['hostname'] = hostname
    grating = get_required_grating(target)
    host_row['obsvtn\ngrating'] = grating

    # hijinks to keep .loc from returning a row instead of a table if there is just one planet
    i_planet = planet_catalog.loc_indices[tic_id]
    i_planet = np.atleast_1d(i_planet)
    planets = planet_catalog[i_planet]

    host = host_catalog.loc[tic_id]
    targfolder = paths.target_data(target)


    #%% stellar props

    host_row['stellar\neff temp (K)'] = host['st_teff'] * host_catalog['st_teff'].unit
    Mstar = host['st_mass'] * host_catalog['st_mass'].unit
    limit_dicionary = {-1:'>', 0:'', 1:'<'}
    if np.ma.is_masked(host['st_age']):
        if not np.ma.is_masked(host['st_rotp']) and host['st_rotplim'] == 0:
            Prot = host['st_rotp'] * host_catalog['st_rotp'].unit
            age = empirical.age_from_Prot_johnstone21(Prot, Mstar)
            host_row['age\nlimit'] = ''
            host_row['age'] = age.to('Gyr')
    else:
        lim_flag = np.ma.filled(host['st_agelim'], 0)
        host_row['age\nlimit'] = limit_dicionary[int(lim_flag)]
        host_row['age'] = host['st_age'] * host_catalog['st_age'].unit

    if np.ma.is_masked(host['st_radv']):
        rv_star = 0 * u.km/u.s
    else:
        rv_star = host['st_radv'] * host_catalog['st_radv'].unit


    # %% lya reconstruction fluxes

    recon_flag = recon_flag_dict.get(hostname, '')
    host_row['lya recnstcnt flag'] = recon_flag.lower()

    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='OverflowError converting to FloatType in column')
        lya_recon = Table.read(lya_reconstruction_file)
    w = lya_recon['wave_lya']
    v_sys = lya.w2v(w) - rv_star.to_value('km s-1')

    v_bins_earth = lya_bins + rv_star
    w_bins = lya.v2w(v_bins_earth.to_value('km s-1'))

    lya_fluxes = []
    lya_core_fluxdenses = []
    for case in 'low_1sig median high_1sig'.split():
        f = lya_recon[f'lya_model unconvolved_{case}']
        lya_binned = utils.intergolate(w_bins, w, f, 0, 0)
        lya_fluxes.append(lya_binned)
        lya_core = np.interp(0, v_sys, f)
        lya_core_fluxdenses.append(lya_core)
    lya_fluxes = np.asarray(lya_fluxes)
    lya_core_fluxdenses = np.asarray(lya_core_fluxdenses)

    flux_unit = u.Unit('erg s-1 cm-2')
    lya_flux_nom = lya_fluxes[1]
    lya_flux_poserr = lya_fluxes[-1] - lya_flux_nom
    lya_flux_negerr = lya_flux_nom - lya_fluxes[0]
    for j in range(len(lya_flux_nom)):
        va, vb = lya_bins[j:j+2].to_value('km s-1')
        key = f'lya flux\n[{va:.0f}, {vb:.0f}]'
        host_row[key] = lya_flux_nom[j] * flux_unit
        host_row[key + ' +err'] = lya_flux_poserr[j] * flux_unit
        host_row[key + ' -err'] = lya_flux_negerr[j] * flux_unit

    fluxdens_unit = flux_unit / u.AA
    lya_coreflux = lya_core_fluxdenses[1]
    lya_coreflux_poserr = lya_core_fluxdenses[-1] - lya_coreflux
    lya_coreflux_negerr = lya_coreflux - lya_core_fluxdenses[0]
    key = 'lya flux\ncore'
    host_row[key] = lya_coreflux * fluxdens_unit
    host_row[key + ' +err'] = lya_coreflux_poserr * fluxdens_unit
    host_row[key + ' -err'] = lya_coreflux_negerr * fluxdens_unit


#%% loop through planets
    for i, planet in enumerate(planets):
        planet_row = host_row.copy()

        letter = get_letter(planet, i)
        planet_row['planet'] = letter
        planet_row['planet\nradius (Re)'] = planet['pl_rade'] * planet_catalog['pl_rade'].unit
        planet_row['orbital\nperiod (d)'] = planet['pl_orbper'] * planet_catalog['pl_orbper'].unit

        flag_cols = [name for name in planet_catalog.colnames if 'flag_' in name]
        for col in flag_cols:
            planet_row[col] = planet[col]

        # verify that I got the right planet and get
        transit_simulation_file, = targfolder.rglob(f'*outflow-tail-model*transit-{letter}.h5')
        with h5py.File(transit_simulation_file) as f:
            params = f['system_parameters'].attrs
            a_sim = params['semimajoraxis'] * u.cm
            phion = params['phion_rate'] / u.s
        a_cat = planet['pl_orbsmax'] * planets['pl_orbsmax'].unit
        if not np.isclose(a_cat, a_sim, rtol=0.1):
            raise ValueError

        planet_row['H ionztn\ntime (h)'] = (1/phion).to('h')

        sigma_tbl_path, = targfolder.rglob(f'*outflow-tail*transit-{letter}*sigmas.ecsv')
        snrs = Table.read(sigma_tbl_path)
        detectable = snrs['transit sigma'] > sigma_threshold
        detectability_fraction = np.sum(detectable)/len(snrs)
        planet_row[f'frac models\nw snr > {sigma_threshold}'] = detectability_fraction
        planet_row['outflow model\nmax snr'] = np.max(snrs['transit sigma'])

        # aperture that yields the best average snr
        apertures = np.unique(snrs['aperture'])
        snrs.add_index('aperture')
        mean_snrs = [np.mean(snrs.loc[aperture]['transit sigma']) for aperture in apertures]
        ibest = np.argmax(mean_snrs)
        planet_row['outflow model\nbest aperture'] = apertures[ibest]

        flat_sigma_tbl_path, = targfolder.rglob(f'*simple-opaque-tail*transit-{letter}*sigmas.ecsv')
        flat_sigma_tbl = Table.read(flat_sigma_tbl_path)
        snr = np.max(flat_sigma_tbl['transit sigma'])
        colname = f'flat transit snr\n[{simple_transit_range[0].value:.0f}, {simple_transit_range[1].value:.0f}]'
        planet_row[colname] = snr
        ibest = np.argmax(flat_sigma_tbl['transit sigma'])
        aperture = flat_sigma_tbl['aperture'][ibest]
        planet_row['flat transit\nbest aperture'] = aperture

        eval_rows.append(planet_row)

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
