import warnings

from astropy.table import Table, Column
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
transit_within_rvs = (-150, 50) * u.km / u.s

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
    transit_within_rvs=transit_within_rvs,
)


#%% tables

stela_name_tbl = preloads.stela_names.copy()
planet_catalog = preloads.planets.copy()
planet_catalog.add_index('tic_id')
host_catalog = preloads.hosts
host_catalog.add_index('tic_id')


#%% move model transit spectra into target folders

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

    Ro = variability.rossby_number(Mstar, Prot)
    jitter = satdecay_jitter(Ro)
    Arot = satdecay_rotation(Ro)


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

    # base the choice on whether the existing Lya data are e140m
    g140m_files = list(targfolder.rglob('*hst-stis-g140m.*_x1d.fits'))
    e140m_files = list(targfolder.rglob('*hst-stis-e140m.*_x1d.fits'))
    g130m_files = list(targfolder.rglob('*hst-cos-g130mm.*_x1d.fits'))
    if g140m_files or g130m_files:
        etc_file = paths.stis / 'g140m_counts_2025-08-05_exptime900_flux1e-13_aperture52x0.2.csv'
        lsf_file = paths.stis / 'LSF_G140M_1200.txt'
        aperture = '52x0.2'
    else:
        if e140m_files:
            etc_file = paths.stis / 'e140m_counts_2025-08-06_exptime900_flux1e-19_aperture0.2x0.2.csv'
            lsf_file = paths.stis / 'LSF_E140M_1200.txt'
            aperture = '0.2x0.2'
        else:
            raise ValueError(f"No lya data for {hostname}. That ain't right.")

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

        letter = planet['pl_letter']
        if np.ma.is_masked(letter):
            letter = 'bcdefg'[i]

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
                **global_transit_kws,
                **target_transit_kws,
                **planet_transit_kws,
            )

            # add parameters to table
            sigma_tbl['lya sigma'] = Column(x_lya_sigma, format='%i')
            sigma_tbl['eta'] = Column(x_eta, format='.2g')
            sigma_tbl['wind scaling'] = Column(x_wind_scaling, format='.2g')
            sigma_tbl['phion scaling'] = Column(x_phion_scaling, format='.2g')

            sigma_tbl_name = transit_simulation_file.name.replace('.h5', '-detection-sigmas.ecsv')
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

            flat_wgrid = np.arange(*spec.wavegrid[[0,-1]], spec.binwidth[0]/3)
            flat_vgrid = lya.w2v(flat_wgrid) - rv_star.to_value('km s-1')

            flat_transit_wavemask = utils.is_in_range(flat_vgrid, *transit_within_rvs.to_value('km s-1'))
            flat_transit_timemask = utils.is_in_range(flat_tgrid, *in_transit_range.to_value('h'))

            flat_depth, = transit.opaque_tail_depth(planets[[i]]) # takes a table as input, hence the [[i]]
            flat_transmission = np.ones((len(flat_tgrid), len(flat_wgrid)))
            flat_transmission[np.ix_(flat_transit_timemask, flat_transit_wavemask)] = 1 - flat_depth

            lya_sigma = 0
            flat_lya_flux = lya_recon['lya_model unconvolved_median']

            flat_sigma_tbl = transit.generic_transit_snr(
                transit_timegrid = flat_tgrid,
                transit_wavegrid = flat_wgrid,
                transit_transmission_ary = flat_transmission,
                lya_recon_flux_ary = flat_lya_flux,
                **global_transit_kws,
                **target_transit_kws,
                **planet_transit_kws,
            )

            # store flat model parameters
            flat_sigma_tbl['injected times'] = [in_transit_range.to_value('h')]
            flat_sigma_tbl['injected rvs'] = [transit_within_rvs.to_value('km s-1')]
            flat_sigma_tbl['depth'] = flat_depth
            flat_sigma_tbl['lya sigma'] = lya_sigma

            flat_filename = f'{target}.simple-opaque-tail.na.na.transit-{letter}-detection-sigmas.ecsv'
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


#%% make table of properties