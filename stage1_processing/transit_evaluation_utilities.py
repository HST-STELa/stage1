import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Mapping, Literal
from copy import copy

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable, Row
import h5py
import pandas as pd

import paths
import utilities as utils
import hst_utilities
import database_utilities as dbutils
import catalog_utilities as catutils
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


@lru_cache(maxsize=None) # enables caching for identical calls
def get_spectrograph_object(grating, aperture, host_ra, host_dec) -> Spectrograph:
    # load in spectrograph info
    usecos = grating == 'g130m'
    usestis = grating in ['g140m', 'e140m']
    assert usecos or usestis, 'Not sure what spectrograh to use for that grating.'
    folder, spec = (paths.cos, cos) if usecos else (paths.stis, stis)
    etc_file = folder / spec.default_etc_filenames[grating][aperture]
    etc = hst_utilities.read_etc_output(etc_file)
    if usestis:
        lsf_name = f'lsf.hst-stis-{grating}-1200.txt'
        proxy_aperture = spec.proxy_lsf_apertures[grating].get(aperture, aperture)
        lsf_x, lsf_y = spec.read_lsf(folder / lsf_name, aperture=proxy_aperture)
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
    num_shifts = max(int(2*expand_dv/vres), 3)
    v_shifts = np.linspace(-expand_dv, expand_dv, num_shifts)
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


@lru_cache(maxsize=None) # enables caching for faster reuses of the same file
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
            planet = Planet(row, i, self.dbname)
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
            raise ValueError('Host parameters, which is just a Table.Row object, has no units. To ensure units, '
                             'be sure the host and planet catalogs are transformed into QTable')
        Mstar = x['st_mass'].unmasked # unmasked avoids a bug in mors when the value is a MaskedQuantity

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
            mass_for_track = max(min(1.2 * u.Msun, Mstar), 0.1 * u.Msun) # mass to use for rotation tracks, which have a min and max
            Prot = empirical.Prot_from_age_johnstone21(age, mass_for_track)
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


class Planet(object):
    def __init__(self, planet_row, planet_row_order, host_dbname):
        self.params = copy(planet_row)
        self.sim_letter = get_outflow_sim_letter(planet_row, planet_row_order)
        _tbl = planet_row.table[[planet_row.index]]
        self.stela_suffix, = dbutils.planet_suffixes(_tbl)
        self.optical_transit_duration = planet_row['pl_trandur']
        self.in_transit_range = u.Quantity((-self.optical_transit_duration / 2, 30 * u.h))  # long egress for tails
        self.dbname = f'{host_dbname}-{self.stela_suffix}'

@dataclass
class TransitModelSet:
    timegrid : np.ndarray
    wavegrid : np.ndarray
    transmission : np.ndarray
    params: Mapping[str, npt.ArrayLike]
    df: pd.Series = field(init=False)

    def __post_init__(self):
        index = pd.MultiIndex.from_arrays(list(self.params.values()),
                                          names=list(self.params.keys()))
        self.df = pd.Series(list(self.transmission), index=index)

    def loc_transmission(self, **param_value_pairs):
        # homogenize units
        for key, val in param_value_pairs.items():
            if hasattr(val, 'unit'):
                param_unit = self.params[key].unit
                param_value_pairs[key] = val.to_value(param_unit)

        # Create a tuple of keys in the order of the MultiIndex
        keys = tuple(param_value_pairs.get(name, slice(None)) for name in self.df.index.names)
        result = self.df.loc[pd.IndexSlice[keys]]
        if isinstance(result, pd.Series):
            result = np.vstack(result.to_numpy())
        return result

    def get_max_depth_times(self):
        pass


_temp_simfile, = list(paths.data_targets.rglob(f'hd149026-b*outflow-tail-model*.h5'))
with h5py.File(_temp_simfile) as f:
    _default_sim_wavgrid = f['wavgrid'][:] * 1e8


@lru_cache(maxsize=None)
def get_transit_from_simulation(host, planet):
    file, = host.folder.rglob(f'{host.dbname}-{planet.stela_suffix}.outflow-tail-model*.h5')
    # load in the transit models
    with h5py.File(file) as f:
        timegrid = f['tgrid'][:]
        wavegrid_sys = f['wavgrid'][:] * 1e8
        if np.all(wavegrid_sys == 0):
            wavegrid_sys = _default_sim_wavgrid
        wavegrid_sys = wavegrid_sys
        transmission_array = f['intensity'][:]
        eta = f['eta'][:]
        mass = f['planet_mass'][:]
        wind_scaling = f['mdot_star_scaling'][:]
        phion = f['phion_scaling'][:] # the "_scaling" is vestigial from an earlier formulation
        params = dict(f['system_parameters'].attrs)

    # verify that I got the right planet
    a_sim = params['semimajoraxis'] * u.cm
    a_cat = planet.params['pl_orbsmax']
    assert np.isclose(a_cat, a_sim, rtol=0.1), "Automatically loaded simulation file doesn't match planet."

    # shift the wavegrid to earth frame to match lya and spectrograph frames
    vgrid_sys = lya.w2v(wavegrid_sys)
    vgrid_earth = vgrid_sys + host.rv.to_value('km s-1')
    wavegrid_earth = lya.v2w(vgrid_earth)

    mdot_star = wind_scaling * params['mdot_star'] * u.g/u.s
    Tion = 1 / phion * u.s
    Tion = Tion.to('h')
    mass = mass * u.g
    x_params = dict(eta=eta, mdot_star=mdot_star, Tion=Tion, mass=mass)

    transitobj = TransitModelSet(
        timegrid,
        wavegrid_earth,
        transmission_array,
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
):
    ta = (obstimes[0] - exptimes[0] / 2)
    tb = (obstimes[-1] + exptimes[-1] / 2)
    dt = 0.25 * u.h
    tgrid = utils.qrange(ta, tb + dt, dt)
    vgrid = utils.qrange(*rv_grid_span, 10*u.km/u.s)
    wgrid = lya.v2w((vgrid + host.rv).to_value('km s-1')) * u.AA
    transmission, depth = transit.flat_transit_transmission(
        planet.params,
        tgrid=tgrid,
        wgrid=wgrid,
        time_range=planet.in_transit_range,
        rv_star=host.rv,
        rv_range=rv_range
    )
    transmission = transmission[None,...]
    to1d = lambda x: u.Quantity([x])
    x_params = {'start': to1d(planet.in_transit_range[0]),
                'stop': to1d(planet.in_transit_range[1]),
                'rv_blue': to1d(rv_range[0]),
                'rv_red': to1d(rv_range[1]),
                'depth': to1d(depth)}
    transitobj = TransitModelSet(tgrid.to_value('h'), wgrid.to_value('AA'), transmission, x_params)
    return transitobj


def build_snr_sampler_fns(
        host: Host,
        host_variability: HostVariability,
        transit_model: TransitModelSet,
        exptime_fn,
        obstimes: u.Quantity,
        baseline_exposures: slice,
        transit_exposures: slice,
        normalization_search_rvs: u.Quantity,
        transit_search_rvs,
):
    ra, dec = host.params['ra'], host.params['dec']

    def snr_single_case(offset, grating, aperture, lya_case, diagnostic_plots=False, transit_keys=None):

        if transit_keys is None:
            transmission = transit_model.transmission
        else:
            transmission = transit_model.loc_transmission(**transit_keys)

        spec = get_spectrograph_object(grating, aperture, ra, dec)

        jitter_breathing = 0 if aperture in ['psa', 'boa'] else stis.breathing_rms[aperture]
        jitters = np.array((host_variability.jitter, jitter_breathing))
        jitter = utils.quadsum(jitters)

        obstimes_offset = obstimes + offset
        expt = exptime_fn(aperture)
        lya_flux = host.lya_reconstruction.fluxes[lya_case]

        obs_starts = obstimes_offset - expt/2
        obs_ends = obstimes_offset + expt/2
        baseline_time_range = (obs_starts[baseline_exposures][0], obs_ends[baseline_exposures][-1])
        transit_time_range = (obs_starts[transit_exposures][0], obs_ends[transit_exposures][-1])
        baseline_time_range, transit_time_range = map(u.Quantity, (baseline_time_range, transit_time_range))

        result = transit.generic_transit_snr(
            obstimes=obstimes_offset,
            exptimes=expt,
            baseline_time_range=baseline_time_range,
            in_transit_time_range=transit_time_range,
            normalization_search_rvs=normalization_search_rvs,
            transit_search_rvs=transit_search_rvs,
            transit_timegrid=transit_model.timegrid,
            transit_wavegrid=transit_model.wavegrid,
            transit_transmission_ary=transmission,
            lya_recon_flux=lya_flux,
            lya_recon_wavegrid=host.lya_reconstruction.wavegrid_earth,
            spectrograph_object=spec,
            rv_star=host.rv,
            rv_ism=host.rv_ism,
            rotation_period=host_variability.Prot,
            rotation_amplitude=host_variability.rotation_amplitude,
            jitter=jitter,
            diagnostic_plots=diagnostic_plots
        )
        sigma_tbl = result[0] if diagnostic_plots else result

        # add time range info
        sigma_tbl.meta['baseline time range'] = baseline_time_range
        sigma_tbl.meta['transit time range'] = transit_time_range

        # add transit parameter info if a full set was run
        if transit_keys is None:
            for k, v in transit_model.params.items():
                sigma_tbl[k] = v

        return result

    def snr_case_by_case(time_offsets, grating_aperture_pairs, lya_cases):
        tbls: List[QTable] = []
        for time_offset in time_offsets:
            for grating, aperture in grating_aperture_pairs:
                for lya_case in lya_cases:
                    sigma_tbl = snr_single_case(time_offset, grating, aperture, lya_case, False)
                    sigma_tbl['time offset'] = time_offset
                    sigma_tbl['grating'] = grating
                    sigma_tbl['aperture'] = aperture
                    sigma_tbl['lya reconstruction case'] = host.lya_reconstruction.sig2lbl[lya_case]
                    tbls.append(sigma_tbl)
        if not tbls:
            return QTable()
        else:
            return catutils.table_vstack_flexible_shapes(tbls)

    return snr_case_by_case, snr_single_case


def best_by_mean_snr(tbl: Table, category_column: str) -> str:
    """Pick the category value with the highest mean 'transit sigma' across rows."""
    xunq = np.unique(tbl[category_column])
    means = [np.nanmean(tbl['transit sigma'][tbl[category_column] == x]) for x in xunq]
    return xunq[int(np.nanargmax(means))]


def filter_to_obs_choices(snr_tbl, aperture, offset):
    filters = {'aperture': aperture,
               'time offset': offset}
    result = catutils.filter_table(snr_tbl, filters)
    return result


def make_diagnostic_plots(
        planet: Planet,
        transit: TransitModelSet,
        snr_fn,
        case_row: Row,
):
    """
    Re-run generic_transit_snr with diagnostic_plots=True for a single row selection;
    save spectra and lightcurve figures with sensible filenames.
    """
    from matplotlib import pyplot as plt

    # keys for the observation case
    grating = case_row['grating']
    aperture = case_row['aperture']
    offset = case_row['time offset']
    lya_case_lbl = case_row['lya reconstruction case']
    lya_case_sigma = LyaReconstruction.lbl2sig[lya_case_lbl]

    # keys for the transit case
    transit_param_cols = list(transit.params.keys())
    transit_keys = {key:case_row[key] for key in transit_param_cols}

    _, wfigs, tfigs = snr_fn(
        offset, grating, aperture, lya_case_sigma,
        diagnostic_plots=True, transit_keys=transit_keys
    )

    title = planet.dbname

    wfig, = wfigs
    wfig.suptitle(title)
    wfig.tight_layout()

    tfig, = tfigs
    tfig.suptitle(title)
    tfig.tight_layout()

    return wfig, tfig


class FileNamer(object):
    descriptors = {
        'model': 'outflow-tail-model',
        'flat': 'simple-opaque-tail'
    }

    def __init__(self, transit_type: Literal['model', 'flat'], planet):
        descriptor = self.descriptors[transit_type]
        self.snr_tbl = f'{planet.dbname}.{descriptor}.detection-sigmas.ecsv'
        self.corner_basename = self.snr_tbl.replace('.ecsv', f'.plot-snr-corner')

    def diagnostic_plot_basename(
            self,
            type: Literal['spectra', 'lightcurve'],
            snr_case: Literal['max', 'median']):
        plot_prefix = self.snr_tbl.replace('detection-sigmas', f'detection-sigmas-{snr_case}')
        return f'{plot_prefix}.plot-{type}'


def save_diagnostic_plots(wfig, tfig, snr_case, host, filenamer: FileNamer):
    get_path = lambda type: host.transit_folder / filenamer.diagnostic_plot_basename(type, snr_case)
    utils.save_pdf_png(wfig, get_path('spectra'))
    utils.save_pdf_png(tfig, get_path('lightcurve'))


def clip_transit_set(
        transit_object: TransitModelSet,
        n: int=3):
    """Reduces the size of a transit set to somethign small that is convenient for quick debugging of
    downstream issues without having to wait for operations to iterate through a huge set."""
    m = transit_object.transmission.shape[0]
    stride = m // (n-1)
    new_transit = copy(transit_object)
    new_transit.transmission = new_transit.transmission[::stride]
    for key in new_transit.params.keys():
        new_transit.params[key] = new_transit.params[key][::stride]
    return new_transit


def best_offset(snr_table, max_offset=np.inf, slctn_fn=best_by_mean_snr):
    mask = snr_table['time offset'] <= max_offset
    snr_table = snr_table[mask]
    best_safe_offset = slctn_fn(snr_table, 'time offset')
    return best_safe_offset