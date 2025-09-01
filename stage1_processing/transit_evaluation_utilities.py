import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from copy import copy

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, QTable
import h5py

import paths
import utilities as utils
import hst_utilities
import database_utilities as dbutils
import empirical

from stage1_processing import preloads

from lya_prediction_tools import lya, ism, transit, stis, cos, variability
from lya_prediction_tools.spectrograph import Spectrograph


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


# @lru_cache(maxsize=None) # enables caching for identical calls
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
    coord = SkyCoord(host_ra, host_dec)
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
    sig2lbl = dict(zip(sigmas, case_labels))

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


# @lru_cache(maxsize=None) # enables caching for faster reuses of the same file
def get_lyarecon_object(filepath) -> LyaReconstruction:
    return LyaReconstruction(filepath)


def read_quality_table(path, header="| Target | Quality Flag |"):
    """
    Parse a Markdown pipe table that starts with `header` from a text file
    and return it as an astropy Table with columns ['Target', 'Quality Flag'].

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Markdown/text file.
    header : str
        Exact-looking header row to locate (spacing is normalized).

    Returns
    -------
    astropy.table.Table
    """
    def _normalize(s: str) -> str:
        # collapse multiple spaces and trim
        return " ".join(s.strip().split())

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    hdr_norm = _normalize(header)
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and _normalize(line) == hdr_norm:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Header not found: {header!r}")

    # Data rows typically begin two lines after the header (skip the separator),
    # but we’ll be defensive and skip any additional separator-like rows.
    data = []
    for line in lines[start_idx + 1:]:
        s = line.strip()
        if not s.startswith("|"):
            # table ended
            break

        # Skip separator rows like |-----|-----| (with optional colons/spaces)
        sep_test = s.replace("|", "").replace(":", "").replace("-", "").strip()
        if sep_test == "":
            continue

        # Parse a data row
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue  # malformed line; ignore
        target, flag = cells[0], cells[1]
        # Stop if row appears empty
        if not target and not flag:
            break
        data.append((target, flag))

    return Table(rows=data, names=("Target", "Quality Flag"))


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

        # radial velocities
        if np.ma.is_masked(params['st_radv']):
            rv_star = 0 * u.km/u.s
        else:
            rv_star = params['st_radv']
        rv_ism = ism.ism_velocity(params['ra'], params['dec'])
        self.rv = rv_star
        self.rv_ism = rv_ism

        # lya reconstruction
        lya_reconstruction_file, = self.folder.rglob('*lya-recon*')
        self.lya_reconstruction = get_lyarecon_object(lya_reconstruction_file)

@dataclass
class VariabilityPredictor:
    """Encapsulates assumptions about how Lya variability evolves with Rossby number, and defines a function to
    predict the amplitudes of rotational and jitter variability."""
    Ro_break : float
    jitter_saturation : float
    jitter_Ro1 : float
    rotation_amplitude_saturation : float
    rotation_amplitude_Ro1 : float
    def __call__(self, Ro):
        rotation_amp = variability.saturation_decay_loglog(
            Ro,
            self.Ro_break,
            self.rotation_amplitude_saturation,
            1,
            self.rotation_amplitude_Ro1)
        jitter_amp = variability.saturation_decay_loglog(
            Ro,
            self.Ro_break,
            self.jitter_saturation,
            1,
            self.jitter_Ro1)
        return rotation_amp.to_value(''), jitter_amp.to_value('')


class HostVariability(object):
    """Calculates and stores guesses for the host star's Lya variability based on rotation, age, and Teff."""
    def __init__(self, host:Host, variability_predictor:VariabilityPredictor):
        x = host.params
        if not hasattr(x['st_rad'], 'unit'):
            raise ValueError('Host parameters, which is just a Table.Row object, has not units. To ensure units, '
                             'be sure the host and planet catalogs are transformed into ')

        # get or guess at age
        fallback_age = 5 * u.Gyr
        self.age_source = 'guess'  # true for all but one condition below
        if not np.ma.is_masked(x['st_age']):
            if x['st_agelim'] == 0:
                age = x['st_age']  # age cataloged, no limit flag
                self.age_source = 'catalog'
            elif x['st_agelim'] == 1:
                age = x['st_age'] / 2  # age cataloged, upper limit, use half as guess
            else:
                age = fallback_age  # age cataloged, lower limit, assume fallback age
        else:
            age = fallback_age  # age cataloged, lower limit, assume fallback age
        self.age = age

        # get or guess at rotation period
        if np.ma.is_masked(x['st_rotp']) or (x['st_rotplim'] != 0):
            # estimate Prot based on age
            Mstar = x['st_mass']
            mass_for_track = max(min(1.2 * u.Msun, Mstar), 0.1 * u.Msun) # mass to use for rotation tracks, which have a min and max
            Prot = empirical.Prot_from_age_johnstone21(age, mass_for_track) * u.d
            self.Prot_source = 'estimated from age'
        else:
            Prot = x['st_rotp']
            self.Prot_source = 'catalog'
        self.Prot = self.rotation_period = Prot

        # make guesses at variability based on rotation period
        if x['st_teff'] > 3500*u.K:
            Ro = variability.rossby_number(Mstar, Prot)
            self.Ro = Ro
            self.Ro_source = 'estimated from mass and rotation'
            jitter_star, Arot = variability_predictor(Ro)
        else:  # assume mid-late Ms are always quite variable -- seems to be the case in observations
            self.Ro = 0
            self.Ro_source = 'set to zero because star is mid-late M'
            jitter_star = variability_predictor.jitter_saturation
            Arot = variability_predictor.rotation_amplitude_saturation
        self.jitter = jitter_star
        self.rotation_amplitude = Arot

    def total_jitter(self, aperture):
        jitter_breathing = stis.breathing_rms[aperture]
        jitters = np.array((self.jitter, jitter_breathing))
        return utils.quadsum(jitters)


class Planet(object):
    def __init__(self, planet_row, planet_row_order):
        self.params = copy(planet_row)
        self.sim_letter = get_outflow_sim_letter(planet_row, planet_row_order)
        _tbl = Table(rows=[dict(planet_row)], masked=True)
        self.stela_suffix, = dbutils.planet_suffixes(_tbl)
        self.optical_transit_duration = planet_row['pl_trandur']
        self.in_transit_range = u.Quantity((-self.optical_transit_duration / 2, 10 * u.h))  # long egress for tails


@dataclass(frozen=True)
class Transit:
    timegrid : np.ndarray
    wavegrid : np.ndarray
    transmission : np.ndarray
    search_rvs : u.Quantity
    x_params: Optional[Mapping[str, np.ndarray]] = field(default_factory=dict)
    extras : Optional[Dict] = field(default_factory=dict)


_temp_simfile, = list(paths.data_targets.rglob(f'hd149026*outflow-tail-model*transit-b.h5'))
with h5py.File(_temp_simfile) as f:
    _default_sim_wavgrid = f['wavgrid'][:] * 1e8


# @lru_cache(maxsize=None)
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
    a_cat = planet.params['pl_orbsmax']
    assert np.isclose(a_cat, a_sim, rtol=0.1), "Automatically loaded simulation file doesn't match planet."

    # shift the wavegrid to earth frame to match lya and spectrograph frames
    vgrid_sys = lya.w2v(wavegrid_sys)
    vgrid_earth = vgrid_sys + host.rv.to_value('km s-1')
    wavegrid_earth = lya.v2w(vgrid_earth)

    mdot_star = wind_scaling * params['mdot_star'] * u.g/u.s
    phion = phion_scaling * params['phion_rate']
    Tion = 1 / phion * u.s
    Tion = Tion.to('h')
    x_params = dict(eta=eta, mdot_star=mdot_star, Tion=Tion)

    transitobj = Transit(
        timegrid,
        wavegrid_earth,
        transmission_corrected,
        x_params
    )

    return transitobj


def construct_flat_transit(
        planet: Planet,
        host: Host,
        obstimes: u.Quantity,
        exptimes: u.Quantity,
        rv_grid_span: u.Quantity,
        rv_range: u.Quantity,
        search_rvs: u.Quantity,
):
    ta = (obstimes[0] - exptimes[0] / 2)
    tb = (obstimes[-1] + exptimes[-1] / 2)
    dt = 0.25
    tgrid = np.arange(ta, tb + dt, dt) * u.h
    vgrid = np.arange(*rv_grid_span.to_value('km s-1'), 10) * u.km / u.s,
    wgrid = lya.v2w((vgrid + host.rv).to_value('km s-1'))
    transmission, depth = transit.flat_transit_transmission(
        planet.params,
        tgrid=tgrid,
        wgrid=wgrid,
        time_range=planet.in_transit_range,
        rv_star=host.rv,
        rv_range=rv_range
    )
    x_params = {'injected times': planet.in_transit_range,
                'injected rvs': rv_range,
                'depth': depth}
    transitobj = Transit(tgrid, wgrid, transmission, search_rvs)
    return transitobj


def broadcast_lya_and_transmission(
        lyarecon_object: LyaReconstruction,
        lya_sigmas_cases: Sequence[float],
        transmission_ary: np.ndarray,
        extras: Optional[Dict] = None,
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
    host_variability: HostVariability,
    time_offsets: Iterable,
    grating_apertures: Iterable[Tuple[str, str]],
    lya_flux_ary: np.ndarray,
    transit_timegrid: np.ndarray,
    transit_wavegrid: np.ndarray,
    transit_transmission_ary: np.ndarray,
    transit_search_rvs: Tuple[float, float],
    exptime_fn,
    obstimes: u.Quantity,
    baseline_time_range: u.Quantity,
    normalization_search_rvs: u.Quantity,
    extra_cols: Optional[Mapping[str, np.ndarray]] = None
) -> QTable:
    """
    Evaluates generic_transit_snr over all (grating, aperture) pairs for given inputs.
    Appends provenance columns and vertically stacks into a single Table.
    """
    tbls: List[QTable] = []
    for time_offset in time_offsets:
        for grating, aperture in grating_apertures:
            ra, dec = host.params['ra'], host.params['dec']
            spec = get_spectrograph_object(grating, aperture, ra, dec)
            jit = host_variability.total_jitter(aperture)
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
                rotation_period=host_variability.Prot,
                rotation_amplitude=host_variability.rotation_amplitude,
                jitter=jit,
            )
            sigma_tbl['time offset'] = time_offset
            sigma_tbl['grating'] = grating
            sigma_tbl['aperture'] = aperture
            if extra_cols:
                for k, v in extra_cols.items():
                    sigma_tbl[k] = v
            tbls.append(sigma_tbl)

    if not tbls:
        return QTable()
    else:
        return vstack(tbls, join_type='exact')


def build_snr_sampler_fn(
        planet: Planet,
        host: Host,
        host_variability: HostVariability,
        transit: Transit,
        exptime_fn,
        obstimes: u.Quantity,
        baseline_time_range: u.Quantity,
        normalization_search_rvs: u.Quantity,
        transit_search_rvs,
):
    def snr_single_case(offset, grating_aperture, lya_case):


    def snr_case_by_case(time_offsets, grating_apertures, lya_cases):
        lya = host.lya_reconstruction
        extra_cols = {**transit.extras,
                      'lya reconstruction case': [lya.sig2lbl[case] for case in lya_cases]}
        bdcst_data = broadcast_lya_and_transmission(
            host.lya_reconstruction,
            lya_cases,
            transit.transmission,
            extra_cols
        )
        lya_flux, transmisison, lya_labels, extra_cols = bdcst_data

        tbls: List[QTable] = []
        for time_offset in time_offsets:
            for grating, aperture in grating_apertures:
                ra, dec = host.params['ra'], host.params['dec']
                spec = get_spectrograph_object(grating, aperture, ra, dec)
                jit = host_variability.total_jitter(aperture)
                obstimes_offset = obstimes + time_offset
                expt = exptime_fn(aperture)

                sigma_tbl = transit.generic_transit_snr(
                    obstimes=obstimes,
                    exptimes=expt,
                    baseline_time_range=baseline_time_range,
                    in_transit_time_range=planet.in_transit_range,
                    normalization_search_rvs=normalization_search_rvs,
                    transit_search_rvs=transit_search_rvs,
                    transit_timegrid=transit.timegrid,
                    transit_wavegrid=transit.wavegrid,
                    transit_transmission_ary=transmisison,
                    lya_recon_flux_ary=lya_flux,
                    lya_recon_wavegrid=host.lya_reconstruction.wavegrid_earth,
                    spectrograph_object=spec,
                    rv_star=host.rv,
                    rv_ism=host.rv_ism,
                    rotation_period=host_variability.Prot,
                    rotation_amplitude=host_variability.rotation_amplitude,
                    jitter=jit,
                )
                sigma_tbl['time offset'] = time_offset
                sigma_tbl['grating'] = grating
                sigma_tbl['aperture'] = aperture
                if extra_cols:
                    for k, v in extra_cols.items():
                        sigma_tbl[k] = v
                tbls.append(sigma_tbl)

            if not tbls:
                return QTable()
            else:
                return vstack(tbls, join_type='exact')

        return




    return snr_case_by_case, snr_single_case


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

    _, wfigs, tfigs = snr_fn([offset], [(grating, aperture)], [lya_case_sigma], diagnostic_plots=True)

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