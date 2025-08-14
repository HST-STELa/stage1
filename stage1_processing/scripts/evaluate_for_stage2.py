import warnings

from astropy.table import Table, Column, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import h5py
from dask.cache import overhead
from matplotlib import pyplot as plt
import matplotlib as mpl

import empirical
import paths
import utilities as utils
import catalog_utilities as catutils

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu

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


#%% instrument details
etc_filenames = dict(
    g140m={
        '52x0.5': 'etc.hst-stis-g140m.2025-08-05.2025103.exptime900_flux1e-13_aperture52x0.5.csv',
        '52x0.2': 'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv',
        '52x0.1': 'etc.hst-stis-g140m.2025-08-05.2025102.exptime900_flux1e-13_aperture52x0.1.csv',
        '52x0.05': 'etc.hst-stis-g140m.2025-08-05.2025101.exptime900_flux1e-13_aperture52x0.05.csv'
    },
    e140m = {
        '6x0.2': 'etc.hst-stis-e140m.2025-08-05.2025104.exptime900_flux1e-13_aperture6x0.2.csv',
        '52x0.05': 'etc.hst-stis-e140m.2025-08-05.2025105.exptime900_flux1e-13_aperture52x0.05.csv'
    }
)


apertures_to_consider = dict(
    g140m='52x0.5 52x0.2 52x0.1 52x0.05'.split(),
    e140m='6x0.2 52x0.05'.split()
)

def exptime_w_peakups(aperture):
    if aperture in stis.peakup_overhead:
        guess_at_acquisition_exptime = 5
        overhead = stis.peakup_overhead[aperture] + stis.peakup_num_exposures[aperture] * guess_at_acquisition_exptime
        exptimes_mod = exptimes.copy()
        exptimes_mod[[0,2]] -= overhead * u.s
    else:
        exptimes_mod = exptimes.copy()
    return exptimes_mod

def get_spectrograph_object(grating, aperture, host):
    # load in spectrograph info
    etc_file = paths.stis / stis.default_etc_filenames[grating][aperture]
    lsf_file = paths.stis / f'LSF_{grating.upper()}_1200.txt'
    etc = stis.read_etc_output(etc_file)
    proxy_aperture = stis.proxy_lsf_apertures[grating].get(aperture, aperture)
    lsf_x, lsf_y = stis.read_lsf(lsf_file, aperture=proxy_aperture)

    # slim down to just around the lya line for speed
    window = lya.v2w((-500, 500))
    in_window = utils.is_in_range(etc['wavelength'], *window)
    etc = etc[in_window]

    # expand the airglow by earth's rv along the LOS as a worst case
    earth_rv_amplitude = 2*np.pi*u.AU/u.year
    earth_rv_amplitude = earth_rv_amplitude.to('km s-1')
    coord = SkyCoord(host['ra']*u.deg, host['dec']*u.deg)
    ecliptic_coord = coord.transform_to('barycentrictrueecliptic')
    unit_vector = ecliptic_coord.cartesian.xyz.value
    parallel_component = unit_vector - unit_vector[2] * np.array([0, 0, 1])
    parallel_fraction = np.linalg.norm(parallel_component)
    expand_dv = earth_rv_amplitude * parallel_fraction
    expand_dv = expand_dv.to_value('km s-1')
    w = etc['wavelength']
    v = lya.w2v(w)
    y = etc['sky_counts']
    vres = (v[1] - v[0])/3
    v_shifts = np.linspace(-expand_dv, expand_dv, int(2*expand_dv/vres))
    interp = lambda vmod: np.interp(v, vmod, y)
    vv = v[None,:] + v_shifts[:,None]
    yy = np.apply_along_axis(interp, 1, vv)
    y_expanded = np.max(yy, axis=0)
    etc['sky_counts'] = y_expanded

    # initialize spectrograph object
    spec = stis.Spectrograph(lsf_x, lsf_y, etc)

    return spec


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


#%% store "global" transit settings

global_transit_kws = dict(
    obstimes=obstimes,
    baseline_time_range=baseline_range,
    normalization_within_rvs=normalization_within_rvs,
)


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


#%% get standard wavgrid from Ethan's sims to use in cases where it seems like there was a bug that zerod it out

temp_simfile, = list(paths.data_targets.rglob(f'hd149026*outflow-tail-model*transit-b.h5'))
with h5py.File(temp_simfile) as f:
    default_sim_wavgrid = f['wavgrid'][:] * 1e8


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
  mpl.use('Agg') # so plot windows don't constantly interrupt my typing

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
    targ_transit_folder = targfolder / 'transit predictions'


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
        jitter_star = satdecay_jitter(Ro)
        Arot = satdecay_rotation(Ro)
    else: # assume mid-late Ms are always quite variable
        jitter_star = jitter_saturation
        Arot = rotation_amplitude_saturation

    def star_plus_breathing_jitter(aperture):
        jitter_breathing = stis.breathing_rms[aperture]
        jitters = np.array((jitter_star, jitter_breathing))
        return utils.quadsum(jitters)


#%% lya reconstruction

    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='OverflowError converting to FloatType in column')
        lya_recon = Table.read(lya_reconstruction_file)
    lya_cases = 'low_2sig low_1sig median high_1sig high_2sig'.split()
    lya_sigs = np.arange(-2, 3)
    lya_sigma_dic = dict(zip(lya_cases, lya_sigs))
    lya_wavegrid_earth = lya_recon['wave_lya']
    lya_flux_ary = [lya_recon[f'lya_model unconvolved_{lya_case}'] for lya_case in lya_cases]
    lya_flux_ary = np.asarray(lya_flux_ary)


#%% store target-specific transit settings

    target_transit_kws = dict(
        lya_recon_wavegrid = lya_wavegrid_earth,
        rv_star = rv_star,
        rv_ism = rv_ism,
        rotation_period = Prot,
        rotation_amplitude = Arot
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

            # load in the transit models
            with h5py.File(transit_simulation_file) as f:
                transit_timegrid = f['tgrid'][:]
                transit_wavegrid_sys = f['wavgrid'][:] * 1e8
                if np.all(transit_wavegrid_sys == 0):
                    transit_wavegrid_sys = default_sim_wavgrid
                transmission_array = f['intensity'][:]
                eta_sim = f['eta'][:]
                wind_scaling_sim = f['mdot_star_scaling'][:]
                phion_scaling_sim = f['phion_scaling'][:]
                params = dict(f['system_parameters'].attrs)

            # verify that I got the right planet
            a_sim = params['semimajoraxis'] * u.cm
            a_cat = planet['pl_orbsmax'] * planets['pl_orbsmax'].unit
            if not np.isclose(a_cat, a_sim, rtol=0.1):
                raise ValueError

            # there seem to be odd "zero-point" offsets in some of the transmission vectors. correct these
            transmaxs = np.max(transmission_array, axis=2)
            offsets = 1 - transmaxs
            transmission_corrected = transmission_array + offsets[:, :, None]

            # shift the wavegrid to earth frame to match lya and spectrograph frames
            transit_vgrid_sys = lya.w2v(transit_wavegrid_sys)
            transit_vgrid_earth = transit_vgrid_sys + rv_star.to_value('km s-1')
            transit_wavegrid_earth = lya.v2w(transit_vgrid_earth)

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

            # get detection sigmas for a range of apertures
            grating = get_required_grating(target)
            tbls = []
            for aperture in apertures_to_consider[grating]:
                spec = get_spectrograph_object(grating, aperture, host)
                jitter = star_plus_breathing_jitter(aperture)
                exptimes_mod = exptime_w_peakups(aperture)

                # run snr calcs
                sigma_tbl = transit.generic_transit_snr(
                    exptimes = exptimes_mod,
                    transit_timegrid = transit_timegrid,
                    transit_wavegrid = transit_wavegrid_earth,
                    transit_transmission_ary = x_transmission,
                    lya_recon_flux_ary = x_lya_flux,
                    transit_within_rvs=search_model_transit_within_rvs,
                    spectrograph_object = spec,
                    jitter=jitter,
                    **global_transit_kws,
                    **target_transit_kws,
                    **planet_transit_kws,
                )

                # add parameters to table
                sigma_tbl['aperture'] = aperture
                sigma_tbl['lya reconstruction case'] = Column(x_lya_cases)
                sigma_tbl['eta'] = Column(x_eta, format='.2g')
                sigma_tbl['wind scaling'] = Column(x_wind_scaling, format='.2g')
                sigma_tbl['phion scaling'] = Column(x_phion_scaling, format='.2g')

                tbls.append(sigma_tbl)

            sigma_tbl = catutils.flexible_table_vstack(tbls)
            sigma_tbl_name = transit_simulation_file.name.replace('.h5', f'.{grating}-detection-sigmas.ecsv')
            sigma_tbl_path = targ_transit_folder / sigma_tbl_name
            sigma_tbl.write(sigma_tbl_path, overwrite=True)

            # diagnostic plots
            isort = np.argsort(sigma_tbl['transit sigma'])
            cases = {'max': isort[-1],
                     'median': isort[len(isort) // 2]}
            sim_lookup_indices = np.arange(len(eta_sim))
            for case, k in cases.items():
                row = sigma_tbl[k]

                sim_mask = (
                    (eta_sim == row['eta'])
                    & (wind_scaling_sim == row['wind scaling'])
                    & (phion_scaling_sim == row['phion scaling'])
                )
                isim, = sim_lookup_indices[sim_mask]
                transmission = transmission_corrected[isim]

                lya_case = row['lya reconstruction case']
                lya_flux = lya_recon[f'lya_model unconvolved_{lya_case}']

                spec = get_spectrograph_object(grating, row['aperture'], host)
                jitter = star_plus_breathing_jitter(row['aperture'])
                exptimes_mod = exptime_w_peakups(aperture)

                _, wfigs, tfigs = transit.generic_transit_snr(
                    exptimes=exptimes_mod,
                    transit_timegrid = transit_timegrid,
                    transit_wavegrid = transit_wavegrid_earth,
                    transit_transmission_ary = transmission,
                    lya_recon_flux_ary = lya_flux,
                    transit_within_rvs=search_model_transit_within_rvs,
                    spectrograph_object = spec,
                    jitter=jitter,
                    diagnostic_plots = True,
                    **global_transit_kws,
                    **target_transit_kws,
                    **planet_transit_kws,
                )

                title = f'{hostname} {letter}'

                wfig, = wfigs
                wfig.tight_layout()
                wfig.suptitle(title)
                wname_pdf = sigma_tbl_name.replace('.ecsv', f'.plot-spectra-{case}-snr.pdf')
                wname_png = sigma_tbl_name.replace('.ecsv', f'.plot-spectra-{case}-snr.png')
                wfig.savefig(targ_transit_folder / wname_pdf)
                wfig.savefig(targ_transit_folder / wname_png, dpi=300)

                tfig, = tfigs
                tfig.tight_layout()
                tfig.suptitle(title)
                tname_pdf = sigma_tbl_name.replace('.ecsv', f'.plot-lightcurve-{case}-snr.pdf')
                tname_png = sigma_tbl_name.replace('.ecsv', f'.plot-lightcurve-{case}-snr.png')
                tfig.savefig(targ_transit_folder / tname_pdf)
                tfig.savefig(targ_transit_folder / tname_png, dpi=300)

                # corner-like plot
                apertures = np.unique(sigma_tbl['aperture'])
                sigma_tbl.add_index('aperture')
                mean_snrs = [np.mean(sigma_tbl.loc[aperture]['transit sigma']) for aperture in apertures]
                ibest = np.argmax(mean_snrs)
                sigma_tbl_best_ap = sigma_tbl.loc[apertures[ibest]]

                labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],σ_Lya'.split(',')
                phion = params['phion_rate'] * sigma_tbl_best_ap['phion scaling']
                Tion = 1/phion * u.s
                lTion = np.log10(Tion.to_value('h'))
                Mdot = params['mdot_star'] * sigma_tbl_best_ap['wind scaling']
                lMdot = np.log10(Mdot)
                eta = sigma_tbl_best_ap['eta']
                leta = np.log10(eta)
                lya_sigma = [lya_sigma_dic[lcase] for lcase in sigma_tbl_best_ap['lya reconstruction case']]
                snr_vec = sigma_tbl_best_ap['transit sigma']
                param_vecs = [leta, lTion, lMdot, lya_sigma]
                cfig, _ = pu.detection_sigma_corner(param_vecs, snr_vec, labels=labels,
                                                    levels=(3,), levels_kws=dict(colors='w'))
                cfig.suptitle(title)
                cname_pdf = sigma_tbl_name.replace('.ecsv', f'.plot-snr-corner.pdf')
                cname_png = sigma_tbl_name.replace('.ecsv', f'.plot-snr_corner.png')
                cfig.savefig(targ_transit_folder / cname_pdf)
                cfig.savefig(targ_transit_folder / cname_png, dpi=300)

                plt.close('all')

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

            # get detection sigmas for a range of apertures
            grating = get_required_grating(target)
            flat_tbls = []
            for aperture in apertures_to_consider[grating]:
                spec = get_spectrograph_object(grating, aperture, host)
                jitter = star_plus_breathing_jitter(aperture)
                exptimes_mod = exptime_w_peakups(aperture)

                flat_sigma_tbl = transit.generic_transit_snr(
                    exptimes=exptimes_mod,
                    transit_timegrid = flat_tgrid,
                    transit_wavegrid = flat_wgrid,
                    transit_transmission_ary = flat_transmission,
                    lya_recon_flux_ary = flat_lya_flux,
                    transit_within_rvs = search_simple_transit_within_rvs,
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
            k_max = np.argmax(flat_sigma_tbl['transit sigma'])
            aperture = sigma_tbl['aperture'][k_max]
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
                transit_within_rvs=search_model_transit_within_rvs,
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
        sigma_tbl = Table.read(sigma_tbl_path)
        detectable = sigma_tbl['transit sigma'] > sigma_threshold
        detectability_fraction = np.sum(detectable)/len(sigma_tbl)
        planet_row[f'frac models\nw snr > {sigma_threshold}'] = detectability_fraction
        planet_row['outflow model\nmax snr'] = np.max(sigma_tbl['transit sigma'])

        # aperture that yields the best average snr
        apertures = np.unique(sigma_tbl['aperture'])
        sigma_tbl.add_index('aperture')
        mean_snrs = [np.mean(sigma_tbl.loc[aperture]['transit sigma']) for aperture in apertures]
        ibest = np.argmax(mean_snrs)
        planet_row['outflow model\nbest aperture'] = apertures[ibest]

        flat_sigma_tbl_path, = targfolder.rglob(f'*simple-opaque-tail*transit-{letter}*sigmas.ecsv')
        flat_sigma_tbl = Table.read(flat_sigma_tbl_path)
        snr = np.max(flat_sigma_tbl['transit sigma'])
        colname = f'flat transit snr\n[{simple_transit_range[0].value:.0f}, {simple_transit_range[1].value:.0f}]'
        planet_row[colname] = snr
        ibest = np.argmax(flat_sigma_tbl['transit sigma'])
        aperture = flat_sigma_tbl['aperture'][ibest]
        planet_row['flat transit\nbrest aperture'] = aperture

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
    'orbital\nperiod (d)': ''.
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
    'flat transit\nbrest aperture' : ''
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
