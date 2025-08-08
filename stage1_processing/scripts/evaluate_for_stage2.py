from astropy import units as u
import numpy as np

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

batch_mode = False
care_level = 2 # 0 = just loop with no stopping, 1 = pause before each loop, 2 = pause at each step


#%% assumed observation timing

obstimes_temp = [-1.5, 0, 18, 19.5, 21, 22.5, 24] * u.h
obstimes = obstimes_temp - 21 * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s
baseline_range = u.Quantity((obstimes[0] - 1*u.h, obstimes[1] + 1*u.h))
def get_in_transit_range(planet):
    transit_duration = planet['pl_trandur'] * planet_catalog['pl_trandur'].unit
    in_transit_range = u.Quantity((-transit_duration, 10 * u.h))  # long egress for tails
    return in_transit_range


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
    planets = planet_catalog.loc[tic_id]
    host = host_catalog.loc[tic_id]
    targfolder = paths.target_data(target)


#%% some stellar params

    Mstar = host['st_mass'] * host_catalog['st_mass'].unit
    Prot = host['st_rotp'] * host_catalog['st_rotp'].unit
    if np.ma.is_masked(Prot):
        # assume Prot for a 5 Gyr star of the same mass
        Minput = min(1.2 * u.Msun, Mstar)
        Minput = max(0.1 * u.Msun, Minput)
        Prot = empirical.Prot_from_age_johnstone21(5 * u.Gyr, Minput)

    if np.ma.is_masked(host['st_radv']):
        rv_star = 0 * u.km/u.s
    else:
        rv_star = host['st_radv'] * host_catalog['st_radv'].unit
    rv_ism = ism.ism_velocity(host['ra']*u.deg, host['dec']*u.deg)


#%% stellar variability guess

    Ro = variability.rossby_number(Mstar, Prot)
    jitter = satdecay_jitter(Ro)
    Arot = satdecay_rotation(Ro)


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


#%% loop through planets and compute sigmas based on outflow sims

    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    for planet in planets:
        in_transit_range = get_in_transit_range(planet)

        letter = planet['pl_letter']
        transit_simulation_file, = targfolder.rglob(f'*outflow-tail-model*transit-{letter}.h5')

        sigma_tbl = transit.model_transit_snr(
            obstimes,
            exptimes,
            in_transit_range,
            baseline_range,
            transit_simulation_file,
            lya_reconstruction_file,
            spec,
            rv_star,
            rv_ism,
            Prot,
            Arot,
            jitter
        )

        sigma_tbl_name = transit_simulation_file.name.replace('.h5', '-detection-sigmas.ecsv')
        sigma_tbl_path = targfolder / 'transit predictions' / sigma_tbl_name
        sigma_tbl.write(sigma_tbl_path, overwrite=True)

        utils.query_next_step(batch_mode, care_level, 1)


#%% loop close

  except StopIteration:
    break