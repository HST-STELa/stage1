import re
import warnings

from astropy.table import Table, Column, vstack
from astropy import units as u
import numpy as np
import h5py
from matplotlib import pyplot as plt

import empirical
import paths
import utilities as utils

from stage1_processing import target_lists
from stage1_processing import preloads

from lya_prediction_tools import variability
from lya_prediction_tools import transit
from lya_prediction_tools import stis
from lya_prediction_tools import lya
from lya_prediction_tools import ism

#%% option to loop (batch mode)

batch_mode = True
care_level = 0 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step


#%% raise arithmetic warnings as errors

np.seterr(divide='raise', over='raise', invalid='raise')


#%% assumed observation timing

obstimes_temp = [-1.5, 0, 18, 19.5, 21, 22.5, 24] * u.h
obstimes = obstimes_temp - 21 * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s
baseline_range = u.Quantity((obstimes[0] - 1*u.h, obstimes[1] + 1*u.h))
def get_in_transit_range(planet):
    transit_duration = planet['pl_trandur'] * planet_catalog['pl_trandur'].unit
    in_transit_range = u.Quantity((-transit_duration/2, 10 * u.h))  # long egress for tails
    return in_transit_range


#%% ranges within which to search for integration bands that maximize SNR
normalization_within_rvs = ((-400, -150), (150, 400)) * u.km / u.s
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


#%% store "global" transit settings

global_transit_kws = dict(
    obstimes=obstimes,
    exptimes=exptimes,
    baseline_time_range=baseline_range,
    normalization_within_rvs=normalization_within_rvs,
)


#%% tables

stela_name_tbl = preloads.stela_names.copy()
planet_catalog = preloads.planets.copy()
planet_catalog.add_index('tic_id')
host_catalog = preloads.hosts
host_catalog.add_index('tic_id')


#%% planet lettering

def get_letter(planet, i):
    letter = planet['pl_letter']
    if np.ma.is_masked(letter):
        letter = 'bcdefg'[i]
    return letter


#%% picking the right grating
def get_required_grating(target):
    targfolder = paths.target_data(target)

    # base the choice on whether the existing Lya data are e140m
    g140m_files = list(targfolder.rglob('*hst-stis-g140m.*_x1d.fits'))
    e140m_files = list(targfolder.rglob('*hst-stis-e140m.*_x1d.fits'))
    g130m_files = list(targfolder.rglob('*hst-cos-g130mm.*_x1d.fits'))
    if g140m_files or g130m_files:
        grating = 'g140m'
    else:
        if e140m_files:
            grating = 'e140m'
        else:
            raise ValueError(f"No lya data for {hostname}. That ain't right.")

    return grating


#%% move model transit spectra into target folders

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


#%% targets

targets = target_lists.eval_no(1)


#%% SKIP? set up loop for batch processing (skip if not in batch mode)

if batch_mode:
    print("When 'Continue?' prompts appear, hit enter to continue, anything else to break out of the loop.")

itertargets = iter(targets)
while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:

#%% move to next target

    target = next(itertargets)
    tic_id, hostname = stela_name_tbl.loc['hostname_file', target][['tic_id', 'hostname']]

    # hijinks to keep .loc from returning a row instead of a table if there is just one planet
    i_planet = planet_catalog.loc_indices[tic_id]
    i_planet = np.atleast_1d(i_planet)
    planets = planet_catalog[i_planet]

    host = host_catalog.loc[tic_id]
    targfolder = paths.target_data(target)


#%% some stellar params

    Mstar = host['st_mass'] * host_catalog['st_mass'].unit
    if np.ma.is_masked(host['st_rotp']):
        # assume Prot for a 5 Gyr star of the same mass
        Minput = min(1.2 * u.Msun, Mstar)
        Minput = max(0.1 * u.Msun, Minput)
        ageunit = host_catalog['st_age'].unit
        if np.ma.is_masked(host['st_age']):
            age = 5 * u.Gyr
        else:
            if host['st_agelim'] == 0:
                age = host['st_age'] * ageunit
            elif host['st_agelim'] == 1:
                age = host['st_age']/2 * ageunit
            else:
                age = 5 * u.Gyr
        Prot = empirical.Prot_from_age_johnstone21(age, Minput)
    else:
        Prot = host['st_rotp'] * host_catalog['st_rotp'].unit

    if np.ma.is_masked(host['st_radv']):
        rv_star = 0 * u.km/u.s
    else:
        rv_star = host['st_radv'] * host_catalog['st_radv'].unit
    rv_ism = ism.ism_velocity(host['ra']*u.deg, host['dec']*u.deg)


#%% stellar variability guess

    if host['st_teff'] > 3500:
        Ro = variability.rossby_number(Mstar, Prot)
        jitter = satdecay_jitter(Ro)
        Arot = satdecay_rotation(Ro)
    else: # assume mid-late Ms are always quite variable
        jitter = jitter_saturation
        Arot = rotation_amplitude_saturation


#%% lya reconstruction

    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='OverflowError converting to FloatType in column')
        lya_recon = Table.read(lya_reconstruction_file)
    lya_cases = 'low_2sig low_1sig median high_1sig high_2sig'.split()
    lya_sigs = np.arange(-2, 3)
    lya_wavegrid = lya_recon['wave_lya']
    lya_flux_ary = [lya_recon[f'lya_model unconvolved_{lya_case}'] for lya_case in lya_cases]
    lya_flux_ary = np.asarray(lya_flux_ary)


#%% pick appropriate spectrograph

    grating = get_required_grating(target)
    etc_filenames = dict(
        g140m = 'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv',
        e140m = 'etc.hst-stis-e140m.2025-08-05.2025092.exptime900_flux1e-13_aperture0.2x0.2.csv'
    )
    etc_file = paths.stis / etc_filenames[grating]
    lsf_file = paths.stis / f'LSF_{grating.upper()}_1200.txt'
    aperture, = re.findall(r'\d\.?\d+x\d\.?\d+', etc_file.name)

    etc = stis.read_etc_output(etc_file)
    lsf_x, lsf_y = stis.read_lsf(lsf_file, aperture=aperture)

    # slim down to just around the lya line for speed
    window = lya.v2w((-500, 500))
    in_window = utils.is_in_range(etc['wavelength'], *window)
    etc = etc[in_window]

    spec = stis.Spectrograph(lsf_x, lsf_y, etc)


#%% store target-specific transit settings

    target_transit_kws = dict(
        lya_recon_wavegrid = lya_wavegrid,
        spectrograph_object = spec,
        rv_star = rv_star,
        rv_ism = rv_ism,
        rotation_period = Prot,
        rotation_amplitude = Arot,
        jitter = jitter
    )


#%% loop through planets and compute transit sigmas

    missed_planets_mod, missed_targets_mod, exceptions_mod = [], [], []
    missed_planets_flat, missed_targets_flat, exceptions_flat = [], [], []
    for i, planet in enumerate(planets):
        in_transit_range = get_in_transit_range(planet)

        letter = get_letter(planet, i)

        planet_transit_kws = dict(
            in_transit_time_range = in_transit_range
        )

#%% outflow model scenarios
        try:
            transit_simulation_file, = targfolder.rglob(f'*outflow-tail-model*transit-{letter}.h5')

            # verify that I got the right planet
            with h5py.File(transit_simulation_file) as f:
                a_sim = f['system_parameters'].attrs['semimajoraxis'] * u.cm
            a_cat = planet['pl_orbsmax'] * planets['pl_orbsmax'].unit
            if not np.isclose(a_cat, a_sim, rtol=0.1):
                raise ValueError

            # load in the transit models
            with h5py.File(transit_simulation_file) as f:
                transit_timegrid = f['tgrid'][:]
                transit_wavegrid = f['wavgrid'][:] * 1e8
                transmission_array = f['intensity'][:]
                eta_sim = f['eta'][:]
                wind_scaling_sim = f['mdot_star_scaling'][:]
                phion_scaling_sim = f['phion_scaling'][:]

            # there seem to be odd "zero-point" offsets in some of the transmission vectors. correct these
            transmaxs = np.max(transmission_array, axis=2)
            offsets = 1 - transmaxs
            transmission_corrected = transmission_array + offsets[:, :, None]

            # expand lya and transit models to include a range of lya line cases
            n = lya_flux_ary.shape[0]
            m = transmission_array.shape[0]
            x_lya_flux = np.repeat(lya_flux_ary, m, axis=0)
            x_lya_sigma = np.repeat(lya_sigs, m)
            x_lya_cases = np.repeat(lya_cases, m)
            x_transmission = np.tile(transmission_corrected, (n,1,1))
            x_eta = np.tile(eta_sim, n)
            x_wind_scaling = np.tile(wind_scaling_sim, n)
            x_phion_scaling = np.tile(phion_scaling_sim, n)

            # get detection sigmas
            sigma_tbl = transit.generic_transit_snr(
                transit_timegrid = transit_timegrid,
                transit_wavegrid = transit_wavegrid,
                transit_transmission_ary = x_transmission,
                lya_recon_flux_ary = x_lya_flux,
                transit_within_rvs=search_model_transit_within_rvs,
                **global_transit_kws,
                **target_transit_kws,
                **planet_transit_kws,
            )

            # add parameters to table
            sigma_tbl['lya reconstruction case'] = Column(x_lya_cases)
            sigma_tbl['eta'] = Column(x_eta, format='.2g')
            sigma_tbl['wind scaling'] = Column(x_wind_scaling, format='.2g')
            sigma_tbl['phion scaling'] = Column(x_phion_scaling, format='.2g')

            sigma_tbl_name = transit_simulation_file.name.replace('.h5', f'.{grating}-detection-sigmas.ecsv')
            sigma_tbl_path = targfolder / 'transit predictions' / sigma_tbl_name
            sigma_tbl.write(sigma_tbl_path, overwrite=True)

        except Exception as e:
            missed_planets_mod.append(planet)
            missed_targets_mod.append(target)
            exceptions_mod.append(e)
            raise

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

            flat_sigma_tbl = transit.generic_transit_snr(
                transit_timegrid = flat_tgrid,
                transit_wavegrid = flat_wgrid,
                transit_transmission_ary = flat_transmission,
                lya_recon_flux_ary = flat_lya_flux,
                transit_within_rvs = search_simple_transit_within_rvs,
                **global_transit_kws,
                **target_transit_kws,
                **planet_transit_kws,
            )

            # store flat model parameters
            flat_sigma_tbl['injected times'] = [in_transit_range.to_value('h')]
            flat_sigma_tbl['injected rvs'] = [search_simple_transit_within_rvs.to_value('km s-1')]
            flat_sigma_tbl['depth'] = flat_depth
            flat_sigma_tbl['lya reconstruction level'] = lya_case

            flat_filename = f'{target}.simple-opaque-tail.na.na.transit-{letter}.{grating}-detection-sigmas.ecsv'
            flat_path = targfolder / 'transit predictions' / flat_filename
            flat_sigma_tbl.write(flat_path, overwrite=True)

        except Exception as e:
            missed_planets_flat.append(planet)
            missed_targets_flat.append(target)
            exceptions_flat.append(e)
            raise

    utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
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

lya_bins = (-150, -50, 50, 150) * u.km/u.s

eval_rows = []
for target in targets:
    host_row = {}
    tic_id, hostname = stela_name_tbl.loc['hostname_file', target][['tic_id', 'hostname']]
    host_row['hostname'] = hostname
    grating = get_required_grating(target)
    host_row['obs_grating'] = grating

    # hijinks to keep .loc from returning a row instead of a table if there is just one planet
    i_planet = planet_catalog.loc_indices[tic_id]
    i_planet = np.atleast_1d(i_planet)
    planets = planet_catalog[i_planet]

    host = host_catalog.loc[tic_id]
    targfolder = paths.target_data(target)


    #%% stellar props

    host_row['st_teff'] = host['st_teff'] * host_catalog['st_teff'].unit
    Mstar = host['st_mass'] * host_catalog['st_mass'].unit
    limit_dicionary = {-1:'>', 0:'', 1:'<'}
    if np.ma.is_masked(host['st_age']):
        if not np.ma.is_masked(host['st_rotp']) and host['st_rotplim'] == 0:
            Prot = host['st_rotp'] * host_catalog['st_rotp'].unit
            age = empirical.age_from_Prot_johnstone21(Prot, Mstar)
            host_row['st_agelim'] = ''
            host_row['st_age'] = age.to('Gyr')
    else:
        lim_flag = np.ma.filled(host['st_agelim'], 0)
        host_row['st_agelim'] = limit_dicionary[int(lim_flag)]
        host_row['st_age'] = host['st_age'] * host_catalog['st_age'].unit

    if np.ma.is_masked(host['st_radv']):
        rv_star = 0 * u.km/u.s
    else:
        rv_star = host['st_radv'] * host_catalog['st_radv'].unit


    # %% lya reconstruction fluxes

    recon_flag = recon_flag_dict.get(hostname, '')
    host_row['st_lya_reconstruction_flag'] = recon_flag.lower()

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
        key = f'st_lyaflux_{va:.0f}_{vb:.0f}'
        host_row[key] = lya_flux_nom[j] * flux_unit
        host_row[key + 'err1'] = lya_flux_poserr[j] * flux_unit
        host_row[key + 'err2'] = lya_flux_negerr[j] * flux_unit

    fluxdens_unit = flux_unit / u.AA
    lya_coreflux = lya_core_fluxdenses[1]
    lya_coreflux_poserr = lya_core_fluxdenses[-1] - lya_coreflux
    lya_coreflux_negerr = lya_coreflux - lya_core_fluxdenses[0]
    key = 'st_lyaflux_core'
    host_row[key] = lya_coreflux * fluxdens_unit
    host_row[key + 'err1'] = lya_coreflux_poserr * fluxdens_unit
    host_row[key + 'err2'] = lya_coreflux_negerr * fluxdens_unit

    for i, planet in enumerate(planets):
        planet_row = host_row.copy()

        letter = get_letter(planet, i)
        planet_row['pl_letter'] = letter
        planet_row['pl_rade'] = planet['pl_rade'] * planet_catalog['pl_rade'].unit

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

        planet_row['pl_ionization_time'] = (1/phion).to('h')

        sigma_tbl_path, = targfolder.rglob(f'*outflow-tail*transit-{letter}*sigmas.ecsv')
        sigma_tbl = Table.read(sigma_tbl_path)
        detectable = sigma_tbl['transit sigma'] > 3
        detectability_fraction = np.sum(detectable)/len(sigma_tbl)
        planet_row['pl_lyatransit_detectability'] = detectability_fraction

        flat_sigma_tbl_path, = targfolder.rglob(f'*simple-opaque-tail*transit-{letter}*sigmas.ecsv')
        flat_sigma_tbl = Table.read(flat_sigma_tbl_path)
        snr, = flat_sigma_tbl['transit sigma']
        planet_row['pl_lyatransit_optimistic_snr'] = snr

        eval_rows.append(planet_row)

eval_table = Table(rows=eval_rows)

column_order = [
    'hostname',
    'pl_letter',
    'pl_lyatransit_detectability',
    'pl_lyatransit_optimistic_snr',
    'st_lya_reconstruction_flag',
    'st_lyaflux_-150_-50',
    'st_lyaflux_-150_-50err1',
    'st_lyaflux_-150_-50err2',
    'st_lyaflux_-50_50',
    'st_lyaflux_-50_50err1',
    'st_lyaflux_-50_50err2',
    'st_lyaflux_50_150',
    'st_lyaflux_50_150err1',
    'st_lyaflux_50_150err2',
    'st_lyaflux_core',
    'st_lyaflux_coreerr1',
    'st_lyaflux_coreerr2',
    'obs_grating',
    'pl_rade',
    'pl_ionization_time',
    'st_teff',
    'st_agelim',
    'st_age',
    'flag_young',
    'flag_measured_mass',
    'flag_high_TSM',
    'flag_gaseous',
    'flag_gas_and_rocky_in_sys',
    'flag_water_world',
    'flag_gap_upper_cusp',
    'flag_super_puff',
    'flag_outflow',
    'flag_outflow_and_not_in_sys',
]
eval_table = eval_table[column_order]

# some grooming
for col in eval_table.colnames:
    if 'flag_' in col:
        eval_table[col] = eval_table[col].astype(bool)

# save
eval_filename = 'stage2_evalution_metrics.csv'
eval_path = paths.catalogs / eval_filename
eval_table.write(eval_path, overwrite=True)