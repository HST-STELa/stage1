import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import h5py

from stage1_processing import preloads

from lya_prediction_tools import lya, ism, transit, stis, cos
from lya_prediction_tools.spectrograph import Spectrograph

import paths
import utilities as utils
import hst_utilities
import database_utilities as dbutils
import empirical


planet_catalog = preloads.planets
host_catalog = preloads.hosts
stela_name_tbl = preloads.stela_names


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
    },
    g130m = {
        'psa': 'etc.hst-cos-g130m.2025-08-14. 2026269.exptime900_flux1e-13_aperturepsa.csv'
    }
)


@lru_cache(maxsize=None) # enables caching for identical calls
def get_spectrograph_object(grating, aperture, host_ra, host_dec) -> Spectrograph:
    # load in spectrograph info
    usecos = grating == 'g130m'
    usestis = grating in ['g140m', 'e140m']
    assert usecos or usestis, 'Not sure what spectrograh to use for that grating.'
    folder = paths.cos if usecos else paths.stis
    etc_file = folder / stis.default_etc_filenames[grating][aperture]
    etc = hst_utilities.read_etc_output(etc_file)
    if usestis:
        lsf_name = f'lsf.hst-stis-{grating}-1200.txt'
        proxy_aperture = stis.proxy_lsf_apertures[grating].get(aperture, aperture)
        lsf_x, lsf_y = stis.read_lsf(folder / lsf_name, aperture=proxy_aperture)
    else:
        lsf_name = f'lsf.hst-cos-{grating}-1291-lp5.txt'
        lsf_x, lsf_y = cos.read_lsf(folder / lsf_name, wavelength=1215.67)

    # slim down to just around the lya line for speed
    window = lya.v2w((-500, 500))
    in_window = utils.is_in_range(etc['wavelength'], *window)
    etc = etc[in_window]

    # expand the airglow by earth's rv along the LOS as a worst case
    earth_rv_amplitude = 2*np.pi*u.AU/u.year
    earth_rv_amplitude = earth_rv_amplitude.to('km s-1')
    coord = SkyCoord(host_ra*u.deg, host_dec*u.deg)
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
    spec = Spectrograph(lsf_x, lsf_y, etc)

    return spec


def get_required_grating(target):
    targfolder = paths.target_data(target)

    # base the choice on whether the existing Lya data are e140m
    g140m_files = list(targfolder.rglob('*hst-stis-g140m.*_x1d.fits'))
    e140m_files = list(targfolder.rglob('*hst-stis-e140m.*_x1d.fits'))
    g130m_files = list(targfolder.rglob('*hst-cos-g130m.*_x1d.fits'))
    if g140m_files or g130m_files:
        grating = 'g140m'
    else:
        if e140m_files:
            grating = 'e140m'
        else:
            raise ValueError(f"No lya data for {target}. That ain't right.")

    return grating


def get_outflow_sim_letter(planet_row, planet_row_order):
    letter = planet_row['pl_letter']
    if np.ma.is_masked(letter):
        letter = 'bcdefg'[planet_row_order]
    return letter


class LyaReconstruction(object):
    case_labels = 'low_2sig low_1sig median high_1sig high_2sig'.split()
    sigmas = np.arange(-2, 3)
    lbl2sig = dict(zip(case_labels, sigmas))
    sig2lbl = dict(zip(case_labels, sigmas))

    def __init__(self, filepath):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='OverflowError converting to FloatType in column')
            tbl = Table.read(filepath)
        self.tbl = tbl
        self.fluxes = dict([(sig, self.get_case(case)) for sig,case in zip(self.sigmas, self.case_labels)])
        self.wavegrid_earth = tbl['wave_lya']

    def get_case(self, case_label):
        return self.tbl[f'lya_model unconvolved_{case_label}']

    def get_ary(self, sigma_cases):
        flux_ary = [self.fluxes[sig] for sig in sigma_cases]
        return np.asarray(flux_ary)


@lru_cache(maxsize=None) # enables caching for faster reuses of the same file
def get_lyarecon_object(filepath) -> LyaReconstruction:
    return LyaReconstruction(filepath)


_temp_simfile, = list(paths.data_targets.rglob(f'hd149026*outflow-tail-model*transit-b.h5'))
with h5py.File(_temp_simfile) as f:
    _default_sim_wavgrid = f['wavgrid'][:] * 1e8


@dataclass(frozen=True)
class Transit:
    timegrid : np.ndarray
    wavegrid : np.ndarray
    transmission : np.ndarray
    search_rvs : u.Quantity
    x_params: Optional[Mapping[str, np.ndarray]] = None


@lru_cache(maxsize=None)
def get_transit_from_simulation(host, planet):
    file, = host.folder.rglob(f'*outflow-tail-model*transit-{planet.sim_letter}.h5')
    # load in the transit models
    with h5py.File(file) as f:
        timegrid = f['tgrid'][:]
        wavegrid_sys = f['wavgrid'][:] * 1e8
        if np.all(wavegrid_sys == 0):
            wavegrid_sys = _default_sim_wavgrid
        wavegrid_sys = wavegrid_sys
        transmission_array = f['intensity'][:]
        eta = f['eta'][:]
        wind_scaling = f['mdot_star_scaling'][:]
        phion_scaling = f['phion_scaling'][:]
        params = dict(f['system_parameters'].attrs)

    # there seem to be odd "zero-point" offsets in some of the transmission vectors. correct these
    transmaxs = np.max(transmission_array, axis=2)
    offsets = 1 - transmaxs
    transmission_corrected = transmission_array + offsets[:, :, None]

    # verify that I got the right planet
    a_sim = params['semimajoraxis'] * u.cm
    a_cat = planet.params['pl_orbsmax'] * planet_catalog['pl_orbsmax'].unit
    assert np.isclose(a_cat, a_sim, rtol=0.1), "Automatically loaded simulation file doesn't match planet."

    # shift the wavegrid to earth frame to match lya and spectrograph frames
    vgrid_sys = lya.w2v(wavegrid_sys)
    vgrid_earth = vgrid_sys + host.rv.to_value('km s-1')
    wavegrid_earth = lya.v2w(vgrid_earth)

    mdot_star = wind_scaling * params['mdot_star'] * u.g/u.s
    phion = phion_scaling * params['phion']
    Tion = 1 / phion * u.s
    Tion = Tion.to('h')
    x_params = dict(eta=eta, mdot_star=mdot_star, Tion=Tion)

    transit = Transit(
        timegrid,
        wavegrid_earth,
        transmission_corrected,
        x_params
    )

    return transit


class Host(object):
    lya_reconstruction : LyaReconstruction
    def __init__(self, name):
        dbname = name
        tic_id, hostname = stela_name_tbl.loc['hostname_file', dbname][['tic_id', 'hostname']]
        self.dbname = name
        self.tic_id = tic_id
        self.hostname = hostname

        # hijinks to keep .loc from returning a row instead of a table if there is just one planet
        i_planet = planet_catalog.loc_indices[tic_id]
        i_planet = np.atleast_1d(i_planet)
        planet_rows = planet_catalog[i_planet]

        # make planet objects
        planets = []
        for i, row in enumerate(planet_rows):
            planet = Planet(row, i)
            planet.dbname = f'{self.dbname} {planet.stela_suffix}'
            planets.append(planet)
        self.planets = planets

        params = host_catalog.loc[tic_id]
        self.params = params

        self.folder = paths.target_data(dbname)
        self.transit_folder = self.folder / 'transit predictions'

        self.anticipated_grating = get_required_grating(dbname)

        # stellar age and rotation
        self.mass = Mstar = params['st_mass'] * host_catalog['st_mass'].unit
        if np.ma.is_masked(params['st_rotp']):
            # assume Prot for a 5 Gyr star of the same mass
            Minput = min(1.2 * u.Msun, Mstar)
            Minput = max(0.1 * u.Msun, Minput)
            ageunit = host_catalog['st_age'].unit
            if np.ma.is_masked(params['st_age']):
                age = 5 * u.Gyr
            else:
                if params['st_agelim'] == 0:
                    age = params['st_age'] * ageunit
                elif params['st_agelim'] == 1:
                    age = params['st_age']/2 * ageunit
                else:
                    age = 5 * u.Gyr
            Prot = empirical.Prot_from_age_johnstone21(age, Minput)
        else:
            Prot = params['st_rotp'] * host_catalog['st_rotp'].unit
        self.Prot = Prot
        self.age = age

        # radial velocities
        if np.ma.is_masked(params['st_radv']):
            rv_star = 0 * u.km/u.s
        else:
            rv_star = params['st_radv'] * host_catalog['st_radv'].unit
        rv_ism = ism.ism_velocity(params['ra']*u.deg, params['dec']*u.deg)
        self.rv = rv_star
        self.rv_ism = rv_ism

        # lya reconstruction
        lya_reconstruction_file, = self.folder.rglob('*lya-recon*')
        self.lya_reconstruction = get_lyarecon_object(lya_reconstruction_file)


def total_jitter_function(host: Host):
    def total_jitter(aperture):
        jitter_breathing = stis.breathing_rms[aperture]
        jitters = np.array((host.jitter, jitter_breathing))
        return utils.quadsum(jitters)
    return total_jitter


class Planet(object):
    def __init__(self, planet_row, planet_row_order):
        self.params = planet_row.copy()
        self.sim_letter = get_outflow_sim_letter(planet_row, planet_row_order)
        _tbl = Table(rows=[dict(planet_row)])
        self.stela_suffix, = dbutils.planet_suffixes(_tbl)
        self.optical_transit_duration = planet_row['pl_trandur'] * planet_catalog['pl_trandur'].unit
        self.in_transit_range = u.Quantity((-self.optical_transit_duration / 2, 10 * u.h))  # long egress for tails


def broadcast_lya_and_transmission(
        lyarecon_object: LyaReconstruction,
        lya_sigmas_cases: Sequence[float],
        transmission_ary: np.ndarray,
        extras: Optional[Mapping[str, np.ndarray]] = None,
):
    """
    Given N Lya cases and M transmission snapshots, returns tiled/repeated arrays so that
    you can evaluate all N*M combinations in one call to generic_transit_snr.
    """

    lya_flux_ary = lyarecon_object.get_ary(lya_sigmas_cases)

    n = np.shape(lya_flux_ary)[0]
    m = np.shape(transmission_ary)[0]

    x_lya_flux = np.repeat(np.asarray(lya_flux_ary), m, axis=0)
    x_transmission = np.tile(transmission_ary, (n, 1, 1))

    lya_case_labels = [lyarecon_object.sig2lbl[sig] for sig in lya_sigmas_cases]
    labels = dict(
        lya_cases=np.repeat(np.asarray(lya_case_labels), m),
        lya_sigs=np.repeat(np.asarray(lya_sigmas_cases), m),
    )

    x_extras = {}
    if extras:
        for k, v in extras.items():
            v = np.asarray(v)
            if v.ndim == 1 and v.shape[0] == m:
                x_extras[k] = np.tile(v, n)
            elif v.ndim == 1 and v.shape[0] == n:
                x_extras[k] = np.repeat(v, m)
            else:
                # default: assume matches transmission row dimension
                x_extras[k] = np.tile(v, (n,) + (1,) * v.ndim)

    return x_lya_flux, x_transmission, labels, x_extras


def run_snr_grid(
    planet: Planet,
    host: Host,
    time_offsets: Iterable,
    grating_apertures: Iterable[Tuple[str, str]],
    lya_flux_ary: np.ndarray,
    transit_timegrid: np.ndarray,
    transit_wavegrid: np.ndarray,
    transit_transmission_ary: np.ndarray,
    transit_search_rvs: Tuple[float, float],
    jitter_fn,
    exptime_fn,
    obstimes: u.Quantity,
    baseline_time_range: u.Quantity,
    normalization_search_rvs: u.Quantity,
    extra_cols: Optional[Mapping[str, np.ndarray]] = None,
    diagnostic_plots = False,
) -> Table:
    """
    Evaluates generic_transit_snr over all (grating, aperture) pairs for given inputs.
    Appends provenance columns and vertically stacks into a single Table.
    """
    tbls: List[Table] = []
    for time_offset in time_offsets:
        for grating, aperture in grating_apertures:
            ra, dec = host.params['ra'], host.params['dec']
            spec = get_spectrograph_object(grating, aperture, ra, dec)
            jit = jitter_fn(aperture)
            expt = exptime_fn(aperture) + time_offset

            sigma_tbl = transit.generic_transit_snr(
                obstimes=obstimes,
                exptimes=expt,
                baseline_time_range=baseline_time_range,
                in_transit_time_range=planet.in_transit_range,
                normalization_search_rvs=normalization_search_rvs,
                transit_search_rvs=transit_search_rvs,
                transit_timegrid=transit_timegrid,
                transit_wavegrid=transit_wavegrid,
                transit_transmission_ary=transit_transmission_ary,
                lya_recon_flux_ary=lya_flux_ary,
                lya_recon_wavegrid=host.lya_reconstruction.wavegrid_earth,
                spectrograph_object=spec,
                rv_star=host.rv,
                rv_ism=host.rv_ism,
                rotation_period=host.Prot,
                rotation_amplitude=host.Arot,
                jitter=jit,
                diagnostic_plots=diagnostic_plots,
            )
            sigma_tbl['time offset'] = time_offset
            sigma_tbl['grating'] = grating
            sigma_tbl['aperture'] = aperture
            if extra_cols:
                for k, v in extra_cols.items():
                    sigma_tbl[k] = v
            tbls.append(sigma_tbl)

    if not tbls:
        return Table()
    return vstack(tbls, join_type='exact')

def build_snr_sampler_fn(
        planet: Planet,
        host: Host,
        transit: Transit,
        exptime_fn,
        obstimes: u.Quantity,
        baseline_time_range: u.Quantity,
        normalization_search_rvs: u.Quantity,
):

    def get_snrs(time_offsets, grating_apertures, lya_cases, diagnostic_plots=False):
        jitter_fn = total_jitter_function(host)
        lya = host.lya_reconstruction
        extras = {**transit.extras,
                  'lya reconstruction case': [lya.sig2lbl[case] for case in lya_cases]}
        bdcst_data = broadcast_lya_and_transmission(
            host.lya_reconstruction,
            lya_cases,
            transit.transmission,
            extras
        )
        lya_flux, transmisison, lya_labels, extras = bdcst_data

        output = run_snr_grid(
            planet,
            host,
            time_offsets,
            grating_apertures,
            lya_flux,
            transit.timegrid,
            transit.wavegrid,
            transmisison,
            transit.search_rvs,
            jitter_fn,
            exptime_fn,
            obstimes,
            baseline_time_range,
            normalization_search_rvs,
            extras,
            diagnostic_plots,
        )

        return output
    return get_snrs


def best_by_mean_snr(tbl: Table, category_column: str) -> str:
    """Pick the aperture with the highest mean 'transit sigma' across rows."""
    xunq = np.unique(tbl[category_column])
    means = [np.nanmean(tbl['transit sigma'][tbl[category_column] == x]) for x in xunq]
    return xunq[int(np.nanargmax(means))]


def filter_to_obs_choices(snr_tbl):
    ap = snr_tbl.meta['best stis aperture']
    off = snr_tbl.meta['best time offset']
    mask = (snr_tbl['aperture'] == ap) & (snr_tbl['time offset'] == off)
    return snr_tbl[mask]


def make_diagnostic_plots(
    title: str,
    outprefix: str,
    case_row: Mapping,
    snr_fn
):
    """
    Re-run generic_transit_snr with diagnostic_plots=True for a single row selection;
    save spectra and lightcurve figures with sensible filenames.
    """
    from matplotlib import pyplot as plt

    grating = case_row['grating']
    aperture = case_row['aperture']
    offset = case_row['time offset']
    lya_case_lbl = case_row['lya reconstruction case']
    lya_case_sigma = LyaReconstruction.lbl2sig[lya_case_lbl]

    _, wfigs, tfigs = snr_fn(offset, [(grating, aperture)], [lya_case_sigma], diagnostic_plots=False)

    wfig, = wfigs
    wfig.tight_layout()
    wfig.suptitle(title)
    wname_pdf = f'{outprefix}.plot-spectra-snr.pdf'
    wname_png = f'{outprefix}.plot-spectra-snr.png'
    wfig.savefig(wname_pdf)
    wfig.savefig(wname_png, dpi=300)

    tfig, = tfigs
    tfig.tight_layout()
    tfig.suptitle(title)
    tname_pdf = f'{outprefix}.plot-lightcurve-snr.pdf'
    tname_png = f'{outprefix}.plot-lightcurve-snr.png'
    tfig.savefig(tname_pdf)
    tfig.savefig(tname_png, dpi=300)

    plt.close('all')