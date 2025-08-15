
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

import utilities as utils
from lya_prediction_tools import lya, transit, stis, cos  # type: ignore
from lya_prediction_tools.spectrograph import Spectrograph  # type: ignore


# -----------------------------
# Context containers
# -----------------------------

@dataclass(frozen=True)
class TargetContext:
    target: str
    tic_id: int
    hostname: str
    host_row: Mapping
    rv_star: u.Quantity
    rv_ism: float  # km/s scalar (barycentric frame)
    Prot: u.Quantity
    Arot: float
    anticipated_grating: str
    lya_wavegrid_earth: np.ndarray
    lya_flux_cases: Dict[str, np.ndarray]  # {case_name: flux array}
    targfolder: object
    transit_folder: object
    planet_catalog: object
    host_catalog: object
    stela_name_tbl: object


@dataclass(frozen=True)
class PlanetContext:
    letter: str
    in_transit_range: u.Quantity   # (t_start, t_end) Quantity in hours or seconds
    transit_simfile: Optional[str] # path or None for flat case
    params: Optional[Mapping]      # contents loaded from HDF5 or equivalent


# -----------------------------
# Small utilities
# -----------------------------

def star_plus_breathing_jitter_fn(jitter_star: float):
    """
    Returns a thin-closure that, given an aperture, combines stellar jitter with STIS breathing
    (quadrature sum). Keeps your call sites simple: jitter(aperture).
    """
    def _fn(aperture: str) -> float:
        jb = stis.breathing_rms[aperture]
        return utils.quadsum(np.array((jitter_star, jb)))
    return _fn


def exptime_with_peakups_fn(exptimes, peakup_overhead, peakup_num_exposures):
    """
    Returns a thin-closure that adjusts exposure times when a given aperture needs peak-ups.
    Assumes exptimes is a 1D Quantity array like [pre, in, post, ...].
    """
    exptimes = exptimes.copy()

    def _fn(aperture: str):
        if aperture in peakup_overhead:
            # Guess acquisitions and subtract their overhead from the time budget
            guess_acq = 5 * u.s
            total = peakup_overhead[aperture] + peakup_num_exposures[aperture] * guess_acq
            emod = exptimes.copy()
            # convention: subtract from "science" phases [0] and [2] if present
            idxs = [i for i in (0, 2) if i < len(emod)]
            for i in idxs:
                emod[i] = np.maximum(0 * emod[i].unit, emod[i] - total)
            return emod
        return exptimes

    return _fn


def broadcast_lya_and_transmission(
    lya_flux_ary: np.ndarray,
    lya_cases: Sequence[str],
    lya_sigs: Sequence[float],
    transmission_ary: np.ndarray,
    extras: Optional[Mapping[str, np.ndarray]] = None,
):
    """
    Given N Lya cases and M transmission snapshots, returns tiled/repeated arrays so that
    you can evaluate all N*M combinations in one call to generic_transit_snr.

    Returns
    -------
    x_lya_flux : (N*M, ...) array
    x_transmission : (N*M, ...) array
    labels : dict of broadcasted labels, including lya_cases and lya_sigs
    x_extras : dict of any extras broadcasted to (N*M, ...)
    """
    n = np.shape(lya_flux_ary)[0]
    m = np.shape(transmission_ary)[0]

    x_lya_flux = np.repeat(np.asarray(lya_flux_ary), m, axis=0)
    x_transmission = np.tile(transmission_ary, (n, 1, 1))

    labels = dict(
        lya_cases=np.repeat(np.asarray(lya_cases), m),
        lya_sigs=np.repeat(np.asarray(lya_sigs), m),
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


# -----------------------------
# Spectrograph/ETC factory with caching
# -----------------------------

@lru_cache(maxsize=None)
def _lsf_for(grating: str, aperture: str, use_stis: bool = True):
    """Load/choose LSF once per (grating, aperture, instrument-type)."""
    if use_stis:
        proxy_ap = stis.proxy_lsf_apertures[grating].get(aperture, aperture)
        lsf_x, lsf_y = stis.read_lsf(f'LSF_{grating.upper()}_1200.txt', aperture=proxy_ap)
    else:
        lsf_x, lsf_y = cos.read_lsf(f'LSF_{grating.upper()}_1200.txt', wavelength=1215.67)
    return lsf_x, lsf_y


def _expand_airglow_to_worst_case(etc: Table, host_row):
    """
    Expand the ETC sky spectrum by scanning plausible airglow velocity shifts and
    keeping the worst-case (maximum) sky counts at each wavelength bin.
    """
    earth_rv_amp = (2 * np.pi * u.AU / u.year).to('km s-1').value
    coord = SkyCoord(host_row['ra'] * u.deg, host_row['dec'] * u.deg).barycentrictrueecliptic
    unit = coord.cartesian.xyz.value
    # component in ecliptic plane
    parallel_fraction = np.linalg.norm(unit - unit[2] * np.array([0, 0, 1]))
    expand_dv = earth_rv_amp * parallel_fraction

    w = etc['wavelength']
    v = lya.w2v(w)
    y = etc['sky_counts']
    vres = (v[1] - v[0]) / 3
    nshift = max(1, int(2 * expand_dv / vres))
    v_shifts = np.linspace(-expand_dv, expand_dv, nshift)
    # Interpolate each shifted model onto v, then take max envelope
    yy = np.array([np.interp(v, v + dv, y, left=y[0], right=y[-1]) for dv in v_shifts])
    etc['sky_counts'] = np.max(yy, axis=0)
    return etc


def make_spec_factory(
    etc_reader,
    use_stis: bool,
    host_row,
    etc_file_for: Mapping[Tuple[str, str], str],
):
    """
    Returns a thin-closure that builds (and memorizes) a Spectrograph object for a given (grating, aperture).
    - etc_reader: function that reads an ETC file path -> Table with 'wavelength' and 'sky_counts'
    - use_stis: True for STIS LSF/behavior, False for COS
    - host_row: host star row (needs 'ra','dec' for airglow expansion)
    - etc_file_for: mapping from (grating, aperture) -> etc file path
    """
    @lru_cache(maxsize=None)
    def _factory(grating: str, aperture: str) -> Spectrograph:
        etc_file = etc_file_for[(grating, aperture)]
        etc = etc_reader(etc_file)
        # crop to relevant Lyα window
        wmin, wmax = lya.v2w((-500, 500))
        mask = (etc['wavelength'] >= wmin) & (etc['wavelength'] <= wmax)
        etc = etc[mask]
        etc = _expand_airglow_to_worst_case(etc, host_row)
        lsf_x, lsf_y = _lsf_for(grating, aperture, use_stis=use_stis)
        return Spectrograph(lsf_x, lsf_y, etc)

    return _factory


# -----------------------------
# Core execution helper
# -----------------------------

def run_snr_grid(
    grating_apertures: Iterable[Tuple[str, str]],
    spec_factory,
    jitter_fn,
    exptime_fn,
    transit_timegrid: np.ndarray,
    transit_wavegrid: np.ndarray,
    transit_transmission_ary: np.ndarray,
    lya_flux_ary: np.ndarray,
    transit_within_rvs: Tuple[float, float],
    global_kws: Mapping,
    target_kws: Mapping,
    planet_kws: Mapping,
    extra_cols: Optional[Mapping[str, np.ndarray]] = None,
) -> Table:
    """
    Evaluates generic_transit_snr over all (grating, aperture) pairs for given inputs.
    Appends provenance columns and vertically stacks into a single Table.
    """
    tbls: List[Table] = []
    for grating, aperture in grating_apertures:
        spec = spec_factory(grating, aperture)
        jit = jitter_fn(aperture)
        expt = exptime_fn(aperture)

        sigma_tbl = transit.generic_transit_snr(
            exptimes=expt,
            transit_timegrid=transit_timegrid,
            transit_wavegrid=transit_wavegrid,
            transit_transmission_ary=transit_transmission_ary,
            lya_recon_flux_ary=lya_flux_ary,
            transit_within_rvs=transit_within_rvs,
            spectrograph_object=spec,
            jitter=jit,
            **global_kws,
            **target_kws,
            **planet_kws,
        )
        sigma_tbl['grating'] = grating
        sigma_tbl['aperture'] = aperture
        if extra_cols:
            for k, v in extra_cols.items():
                sigma_tbl[k] = v
        tbls.append(sigma_tbl)

    if not tbls:
        return Table()
    return vstack(tbls, join_type='exact')


def best_aperture_by_mean_snr(tbl: Table) -> str:
    """Pick the aperture with the highest mean 'transit sigma' across rows."""
    aps = np.unique(tbl['aperture'])
    means = [np.nanmean(tbl['transit sigma'][tbl['aperture'] == a]) for a in aps]
    return aps[int(np.nanargmax(means))]


def make_diagnostic_plots(
    title: str,
    outprefix: str,
    spec_factory,
    jitter_fn,
    exptime_fn,
    case_row: Mapping,
    transit_timegrid: np.ndarray,
    transit_wavegrid: np.ndarray,
    transmission: np.ndarray,
    lya_flux: np.ndarray,
    transit_within_rvs: Tuple[float, float],
    global_kws: Mapping,
    target_kws: Mapping,
    planet_kws: Mapping,
):
    """
    Re-run generic_transit_snr with diagnostic_plots=True for a single row selection;
    save spectra and lightcurve figures with sensible filenames.
    """
    grating = case_row['grating']
    aperture = case_row['aperture']
    spec = spec_factory(grating, aperture)
    jit = jitter_fn(aperture)
    expt = exptime_fn(aperture)

    _, wfigs, tfigs = transit.generic_transit_snr(
        exptimes=expt,
        transit_timegrid=transit_timegrid,
        transit_wavegrid=transit_wavegrid,
        transit_transmission_ary=transmission,
        lya_recon_flux_ary=lya_flux,
        transit_within_rvs=transit_within_rvs,
        spectrograph_object=spec,
        jitter=jit,
        diagnostic_plots=True,
        **global_kws,
        **target_kws,
        **planet_kws,
    )

    import matplotlib.pyplot as plt

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


# -----------------------------
# Rotation/jitter helper (optional)
# -----------------------------

def compute_rotation_and_jitter(
    host_row: Mapping,
    host_catalog,
    Ro_break: float,
    jit_sat: float,
    jit_Ro1: float,
    rot_sat: float,
    rot_Ro1: float,
):
    """
    Mirror of your in-script logic to compute Prot and jitter scaling, returning
    (Prot, Arot, jitter_star). Keeps the main script cleaner.
    """
    # Prot inference
    if np.ma.is_masked(host_row['st_rotp']):
        Mstar = host_row['st_mass'] * host_catalog['st_mass'].unit
        Minput = min(1.2 * u.Msun, max(0.1 * u.Msun, Mstar))
        ageunit = host_catalog['st_age'].unit
        if np.ma.is_masked(host_row['st_age']):
            age = 5 * u.Gyr
        else:
            if host_row.get('st_agelim', 0) == 0:
                age = host_row['st_age'] * ageunit
            elif host_row.get('st_agelim', 0) == 1:
                age = host_row['st_age'] / 2 * ageunit
            else:
                age = 5 * u.Gyr

        from empirical import Prot_from_age_johnstone21  # local import to avoid hard dep here
        Prot = Prot_from_age_johnstone21(age, Minput)
    else:
        Prot = host_row['st_rotp'] * host_catalog['st_rotp'].unit

    # Jitter / rotational amplitude via Rossby scaling
    from lya_prediction_tools import variability  # type: ignore
    if host_row['st_teff'] > 3500:
        Ro = variability.rossby_number(host_row['st_mass'] * host_catalog['st_mass'].unit, Prot)
        jitter_star = variability.saturation_decay_loglog(Ro, Ro_break, jit_sat, 1, jit_Ro1).to_value('')
        Arot = variability.saturation_decay_loglog(Ro, Ro_break, rot_sat, 1, rot_Ro1).to_value('')
    else:
        jitter_star = jit_sat
        Arot = rot_sat

    return Prot, Arot, jitter_star
