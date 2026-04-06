import re
import warnings
from pathlib import Path
from typing import NamedTuple, Optional
import io
import contextlib

import numpy as np
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.mast import MastMissions
from astropy.io import fits
from astropy import time, table
from astropy import units as u
from astropy.coordinates import SkyCoord

import database_utilities as dbutils
import utilities as utils
from stage1_processing.observation_table import notes_menu, reasons_menu

hst_database = MastMissions(mission='hst')

def locate_nearby_acquisitions(path, radius, additional_files=()):
    max_visit_length = 10*u.h
    h = fits.open(path)
    hdr = h[0].header + h[1].header
    pieces = dbutils.parse_filename(path)
    ra = h[0].header['ra_targ']
    dec = h[0].header['dec_targ']
    coords = SkyCoord(ra*u.deg, dec*u.deg)

    # a gotcha is that sometimes the acquisitions are labeled as a separate visit
    # (see DS Tuc A visits OE8T01 and OE8TA1)
    # so instead I will base the search on looking for any exposures within a visit length of time that have ACQ in mode
    expstart = time.Time(hdr['expstart'], format='mjd')
    searchstart = expstart - max_visit_length
    date_search_str = f'{searchstart.iso}..{expstart.iso}'

    # make sure observations are from the same program
    id = pieces['id']
    id_searchstr = id[:4] + '*' # all files with this root will be from the same observation set
    results = hst_database.query_region(
        coords,
        radius=radius,
        sci_instrume=h[0].header['instrume'],
        sci_data_set_name=id_searchstr,
        sci_start_time=date_search_str,
        sci_operating_mode='*ACQ*',
        select_cols=[
            'sci_operating_mode',
            'sci_start_time',
            'sci_targname',
            'sci_instrume'
        ]
    )
    if len(results) == 0:
        warnings.warn(f'No acquitions found for {path.name}')
        return []
    datasets = hst_database.get_unique_product_list(results)
    filtered = hst_database.filter_products(datasets, file_suffix=['RAW', 'RAWACQ'] + list(additional_files))

    results.add_index('sci_data_set_name')
    filtered['obsmode'] = results.loc[filtered['dataset']]['sci_operating_mode']
    filtered['start'] = results.loc[filtered['dataset']]['sci_start_time']
    filtered['inst'] = results.loc[filtered['dataset']]['sci_instrume']

    return filtered


def infer_associated_acquisitions(path, hst_database_table):
    dbtbl = hst_database_table
    acq_mask = np.char.count(dbtbl['obsmode'].filled(''), 'ACQ') > 0
    dbtbl = dbtbl[acq_mask]
    dbtbl = hst_database.filter_products(dbtbl, file_suffix=['RAW', 'RAWACQ'])

    expstart = fits.getval(path, 'expstart', 1)
    expstart = time.Time(expstart, format='mjd')

    instrument, = set(dbtbl['instrument_name'])
    instrument = instrument.strip()

    times = time.Time(dbtbl['start'])
    dt = expstart - times
    assert np.all(dt.to_value('h') > 0)

    # work backwards in time keeping the closest acquisition of each type
    # discarding any that are out of order
    # for STIS, the sequence goes acq and then a single peakd or a peakd and peakxd
    # confusingly, these both will be marked just as acq/peak
    # for COS, it is acq/search, /image, /peakxd, and /peakd
    i_backwards = np.argsort(dt)
    if instrument == 'STIS':
        acq = []
        peaks = []
        for row in dbtbl[i_backwards]:
            mode = row['obsmode']
            if mode == 'ACQ':
                acq = [row]
                break
            elif mode == 'ACQ/PEAK':
                if len(peaks) == 0:
                    peaks.append(row)
                elif len(peaks) == 1:
                    # might be a peakd+peakxd pair. if they're adjacent in time, keep them both
                    dt = time.Time(peaks[0]['start']) - time.Time(row['start'])
                    if dt < 45*u.min:
                        peaks.append(row)
                # if n >= 2, do nothing, you can't have more peakups than two for a given science exposure
        acq_rows = acq + peaks
    elif instrument == 'COS':
        mode_sequence = ['ACQ/SEARCH', 'ACQ/IMAGE', 'ACQ/PEAKXD', 'ACQ/PEAKD']
        acq_rows = []
        i_sequence = 5
        for row in dbtbl[i_backwards]:
            mode = row['obsmode']
            j = mode_sequence.index(mode)
            if j < i_sequence:
                acq_rows.append(row)
                i_sequence = j
    else:
        raise NotImplementedError(f'{instrument} not implemented.')

    acqtbl = table.Table(rows=acq_rows, names=dbtbl.colnames)
    acqtbl.sort('start')
    return acqtbl


def is_key_science_file(file):
    file = Path(file)
    pieces = dbutils.parse_filename(file.name)
    if 'hst-cos' in pieces['config']:
        if pieces['type'] == 'x1d':
            return True
        else:
            return False
    if 'hst-stis' in pieces['config']:
        if 'tag' in pieces['type']:
            return True
        if 'raw' in pieces['type']:
            mode = fits.getval(file, 'obsmode')
            if mode == 'ACCUM':
                return True
            else:
                return False
    else:
        return False


# Gaussian σ equivalent for MAD (median absolute deviation about the median)
_MAD_SCALE_NORMAL = 1.482602218505602


def spectral_comparison_band_definitions(science_config):
    """
    Bandpasses for chi²-style spectral comparison, in priority order.

    Lyα is only included for STIS G140M / E140M. STIS G140L uses wider windows for
    the UV lines (no Lyα step).
    """
    cfg = science_config.lower()
    g140l = 'g140l' in cfg
    out = []
    if not g140l and ('stis-g140m' in cfg or 'stis-e140m' in cfg):
        out.append(('Lyα', 1210.0, 1220.0))
    if g140l:
        out.extend(
            [
                ('C II', 1330.0, 1340.0),
                ('C IV', 1545.0, 1555.0),
                ('C III', 1170.0, 1182.0),
                ('He II', 1635.0, 1645.0),
            ]
        )
    else:
        out.extend(
            [
                ('C II', 1333.0, 1337.0),
                ('C IV', 1547.0, 1553.0),
                ('C III', 1173.0, 1178.0),
                ('He II', 1638.0, 1643.0),
            ]
        )
    return out


def select_spectral_comparison_band(wavelength, science_config, min_pixels=10):
    """
    Choose the first priority band that has enough samples on ``wavelength``.

    Parameters
    ----------
    wavelength : ndarray
        1-D wavelength grid (Å), same length as flux/median/mad arrays.
    science_config : str
        Row ``science config`` string (e.g. ``hst-stis-g140m``).
    min_pixels : int
        Minimum number of grid points inside the band.

    Returns
    -------
    tuple (str, ndarray) or None
        ``(band_name, mask)`` where ``mask`` is a boolean array on the grid, or
        ``None`` if no band qualifies.
    """
    wave = np.asarray(wavelength, dtype=float)
    for name, wlo, whi in spectral_comparison_band_definitions(science_config):
        mask = (wave >= wlo) & (wave <= whi) & np.isfinite(wave)
        if np.count_nonzero(mask) >= min_pixels:
            return name, mask
    return None


def band_chi2_sigma_vs_median_and_zero(
    flux,
    median,
    mad,
    *,
    mad_scale=_MAD_SCALE_NORMAL,
    zero_mad_percentile=5.0,
):
    """
    Chi²-style deviation of one spectrum from the ensemble median vs. from zero.

    Uses scaled MAD (``mad * mad_scale``) per pixel as σ for comparison to the median.
    For comparison to zero, σ is constant: the given percentile of positive scaled
    MAD values in the band (robust floor so near-zero MAD pixels do not dominate).

    Each χ² is divided by ``sqrt(2 * dof)`` with ``dof`` the number of pixels used
    (finite flux, median, mad, and strictly positive scaled MAD).

    Parameters
    ----------
    flux, median, mad : ndarray
        Aligned 1-D arrays for a single wavelength band.

    Returns
    -------
    sigma_vs_median, sigma_vs_zero : float
        ``nan`` if the band cannot be evaluated (no usable pixels or invalid σ₀).
    """
    flux = np.asarray(flux, dtype=float)
    median = np.asarray(median, dtype=float)
    mad = np.asarray(mad, dtype=float)
    sigma_robust = mad * mad_scale
    base = np.isfinite(flux) & np.isfinite(median) & np.isfinite(mad)
    use = base & (sigma_robust > 0)
    dof = int(np.count_nonzero(use))
    if dof == 0:
        return np.nan, np.nan

    resid = (flux - median)[use]
    sig = sigma_robust[use]
    chi2_median = float(np.sum((resid / sig) ** 2))
    sigma_vs_median = chi2_median / np.sqrt(2.0 * dof)

    sigma0 = float(np.percentile(sig, zero_mad_percentile))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return sigma_vs_median, np.nan

    chi2_zero = float(np.sum((flux[use] / sigma0) ** 2))
    sigma_vs_zero = chi2_zero / np.sqrt(2.0 * dof)
    return sigma_vs_median, sigma_vs_zero


def spectral_band_chi2_sigmas(
    wavelength,
    flux,
    median,
    mad,
    science_config,
    *,
    min_pixels=3,
    zero_mad_percentile=5.0,
):
    """
    Select a priority band and return normalized χ² sigmas vs. median and vs. zero.

    Returns
    -------
    band_name : str or None
    sigma_vs_median, sigma_vs_zero : float
    """
    picked = select_spectral_comparison_band(wavelength, science_config, min_pixels=min_pixels)
    if picked is None:
        return None, np.nan, np.nan
    name, mask = picked
    sm, sz = band_chi2_sigma_vs_median_and_zero(
        flux[mask],
        median[mask],
        mad[mask],
        zero_mad_percentile=zero_mad_percentile,
    )
    return name, sm, sz


def central_chunk_prominence_sigma(image, n, *, tile_reduce=np.sum):
    """
    Split a 2D image into an ``n×n`` grid of equal tiles (``n`` odd), sum each tile,
    and measure how many Gaussian-equivalent σ the **central** tile’s sum lies from the
    median of the **other** tiles, using the MAD of those other tiles as a robust scale.

    The grid stays centered on the image: if height or width is not a multiple of ``n``,
    the crop uses the largest size divisible by ``n``, dropping the same number of
    pixels from the top and bottom (and from left and right). If one extra pixel must
    be removed, it is taken from the bottom and/or right so the grid stays centered.

    Parameters
    ----------
    image : array_like
        Two-dimensional array.
    n : int
        Odd number of tiles along each axis; must be ≥ 3.
    tile_reduce : callable, optional
        Ufunc-like ``f(array, axis=...)`` over tile pixels; default ``numpy.sum``.

    Returns
    -------
    float
        (central chunk sum − reference median) / (MAD × normal Gaussian scale). Signed.
        If MAD is 0, returns ``±inf`` when the central sum differs from the reference
        median, else ``0.0``.

    Notes
    -----
    Median and MAD are computed from the ``n² − 1`` non-central tile sums so the
    central tile does not inflate the scatter estimate. Robust scale is
    ``MAD × 1.4826…`` (normal consistency).
    """
    if n % 2 == 0 or n < 3:
        raise ValueError('n must be an odd integer >= 3.')

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError('image must be a 2D array.')

    h, w = arr.shape
    h_trim = (h // n) * n
    w_trim = (w // n) * n
    if h_trim < n or w_trim < n:
        raise ValueError(
            f'After centering, image must span at least {n} pixels per axis; got trim shape {(h_trim, w_trim)}.'
        )

    dh, dw = h - h_trim, w - w_trim
    row_start = dh // 2
    row_end = row_start + h_trim
    col_start = dw // 2
    col_end = col_start + w_trim

    sub = arr[row_start:row_end, col_start:col_end]
    chunk_h, chunk_w = h_trim // n, w_trim // n

    # (n, chunk_h, n, chunk_w) → sum over tile pixels → (n, n)
    tiles = sub.reshape(n, chunk_h, n, chunk_w)
    chunk_sums = tile_reduce(tiles, axis=(1, 3))
    if chunk_sums.shape != (n, n):
        raise RuntimeError('internal shape error in chunk_sums')

    ci = cj = n // 2
    central_sum = float(chunk_sums[ci, cj])

    mask = np.ones((n, n), dtype=bool)
    mask[ci, cj] = False
    peripheral = chunk_sums[mask]
    median_reference = float(np.median(peripheral))
    mad = float(np.median(np.abs(peripheral - median_reference)))
    robust_sigma = mad * _MAD_SCALE_NORMAL

    if robust_sigma > 0:
        sigma_prominence = (central_sum - median_reference) / robust_sigma
    elif central_sum == median_reference:
        sigma_prominence = 0.0
    else:
        sigma_prominence = np.inf if central_sum > median_reference else -np.inf

    return float(sigma_prominence)


def read_etc_output(etc_output_file):
    try:
        pattern = r'etc.hst-\w+-(.*?)\.(.*?)\..*exptime(.*?)_flux(.*?)_aperture(.*?)\.csv'
        result, = re.findall(pattern, etc_output_file.name)
        grating, date, expt, flux, aperture = result
    except ValueError as e:
        if 'too many values' in str(e) or 'not enough values' in str(e):
            raise(ValueError('Nonstandard ETC output file name. Example standard name: '
                             'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv'))
        else:
            raise
    etc = table.Table.read(etc_output_file)
    etc.meta['date of etc run'] = date
    etc.meta['aperture'] = aperture
    etc.meta['grating'] = grating
    etc.meta['exptime'] = float(expt)
    etc.meta['flux'] = float(flux)
    etc['flux2cps'] = etc['target_counts'] / etc.meta['exptime'] / etc.meta['flux']
    etc['bkgnd_cps'] = (etc['total_counts'] - etc['target_counts']) / etc.meta['exptime']
    return etc


def get_extraction_strip_ratio(x1d_spec_hdu):
    """ratio of trace extraction width to sum of background extraction strip widths"""
    return x1d_spec_hdu['extrsize'] / (x1d_spec_hdu['bk1size'] + x1d_spec_hdu['bk2size'])


def get_flux_factor(x1d_data):
    z = x1d_data['net'] == 0
    if np.any(z):
        raise NotImplementedError
    return x1d_data['flux'] / x1d_data['net']


def get_background_flux(x1d_data):
    size_scale = get_extraction_strip_ratio(x1d_data)
    flux_factor = get_flux_factor(x1d_data)
    return x1d_data['background'] * flux_factor * size_scale


def construct_error_predictor(x1d_hdu, data_mask=None, order=None):
    x1d_data = x1d_hdu[1].data[order]

    if data_mask is None:
        data_mask = np.ones_like(x1d_data['flux'], bool)

    fluxfac = get_flux_factor(x1d_data)
    T = x1d_hdu[1].header['exptime']

    bk_cntrate = x1d_data['background']
    bk_cnts = bk_cntrate*T

    bk_cnts = bk_cnts[data_mask]
    fluxfac = fluxfac[data_mask]

    def predict_errors_fn(flux_model):
        mod_cntrate = flux_model / fluxfac
        mod_cnts = mod_cntrate * T

        tot_cnts = mod_cnts + bk_cnts
        poiss_err_cnts = np.sqrt(tot_cnts)
        poiss_err_flux = poiss_err_cnts / T * fluxfac

        return poiss_err_flux

    return predict_errors_fn


def mitigate_error_inflation(f, e, n_var=10, n_window=50):
    warnings.warn("The mitigate_error_inflation function is experimental and does not perform"
                  "as well as I'd like.")

    # get rid of "floor"
    var_no_floor = utils.shift_floor_to_zero(e**2, n_window)

    # add a new floor based on min variance within a window
    ## first compute variances in a small sliding window
    vars = utils.apply_across_window(f, np.var, n_var)
    ## then take median of these across the wider window
    var_smooth = utils.apply_across_window(vars, np.median, n_window)
    ## now add the new floor
    var_mod = var_no_floor + var_smooth
    e_mod = np.sqrt(var_mod)

    # in areas where flux is changing rapidly over the entire window, like the edges of lines,
    # variance can exceed pipeline error. use pipeline error there.
    e_mod = np.clip(e_mod, 0, e)

    return e_mod


def _acq_msg_print(msgs, verbosity, tastis_output=None):
    if verbosity == 1:
        if msgs:
            print('\n\n'.join(msgs))
        else:
            print('No issues identified.')
    if verbosity == 2:
        print(tastis_output)


def auto_validate_cos_acq_peakxd(fits_object, verbosity=1):
    """return true/false based on whether slew was roughly equal to centroid offset"""
    h = fits_object
    msgs = []

    plate_scale_xd_dic = dict(G130M=100 / 1000, G160M=90 / 1000, G140L=90 / 1000, G230L=24 / 1000,
                              MIRRORA=23.5 / 1000, MIRRORB=23.5 / 1000)
    plate_scale_xd = plate_scale_xd_dic[h[0].header['opt_elem']]

    counts = h[1].data['counts']
    if counts == 0:
        msgs.append(notes_menu['peakxd zeros'])

    centroid_offset = (h[0].header['acqmeasy'] - h[0].header['acqprefy']) * plate_scale_xd # arcsec
    slew = h[0].header['ACQSLEWY'] # arcsec

    atol = 0.2 # arcsec
    slew_diff = np.abs(centroid_offset- slew)

    if verbosity == 2:
        print(f'centroid offset {centroid_offset:.2f} | slew {slew:.2f} | difference {slew_diff:.2f}')

    if not slew_diff > atol:
        msgs.append(
            notes_menu['peakxd big slew'].format(slew_diff=slew_diff, atol=atol)
        )

    _acq_msg_print(msgs, verbosity)

    return msgs


def auto_validate_cos_acq_peakd(fits_object, verbosity=1):
    """return True/False if the ACQ/PEAKD is reasonable/suspect based on whether the slew is between the top two
    dwell points in erms of counts"""
    msgs = []
    h = fits_object

    offsets = h[1].data['DISP_OFFSET']  # arcsec
    counts = h[1].data['counts']

    lo_cts_threshold = 100
    if np.all(counts == 0):
        msgs.append(notes_menu['peakd zeros'])
    elif np.all(counts <= lo_cts_threshold):
        msgs.append(
            notes_menu['peakd lo cts'].format(lo_cts_threshold, lo_cts_threshold)
        )

    if not np.all(counts == 0):
        slew = h[0].header['ACQSLEWX']  # arcsec
        wgtd_offset = np.sum(offsets*counts)/np.sum(counts) # arcsec
        atol = 0.1 # arcsec
        slew_diff = np.abs(wgtd_offset - slew)

        if verbosity == 2:
            print(f'count-weighted offset {wgtd_offset:.2f} | slew {slew:.2f} | difference {slew_diff:.2f}')

        if not slew_diff > atol:
            msgs.append(
                notes_menu['peakd big slew'].format(slew_diff=slew_diff, atol=atol)
            )

    _acq_msg_print(msgs, verbosity)

    return msgs


def auto_validate_stis_acq(acq_path, verbosity=1, return_full_output=False):
    """Run the tastis acquistion checking tool and return any warnings that result. Optionally print
    the final tastis synopsis (verbosity=1), the full output (2), or nothing (0)."""

    import stistools as stis # import here so that hst_utilities works even if stistools not in env

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stis.tastis.tastis(str(acq_path))  # full printout from this is noisy, but helpful if you want to inspect closely
    output = buf.getvalue()
    pieces = output.split('-------------------------------------------------------------------------------')
    synopsis = pieces[-1]

    msgs = []
    for msg in synopsis.split('\n\n'):
        msg = msg.replace('\n', ' ')
        if not (
            msg.strip() == ''
            or 'succe' in msg
            or '===========' in msg
        ):
            msgs.append(msg)

    _acq_msg_print(msgs, verbosity, output)

    if return_full_output:
        return msgs, output
    return msgs


def acq_image_eval(test_image, n_chunks, sigma_threshold):
    prom = central_chunk_prominence_sigma(test_image, n_chunks)
    note = notes_menu['acq target flux'].format(n=n_chunks, sigma=prom)
    passes = np.isfinite(prom) and prom >= sigma_threshold
    return note, passes


class KeyScienceDataQualityAssessment(NamedTuple):
    """Result of header/data checks on key science files for one observation row."""

    reject: bool
    reason: str
    odd_expflag: Optional[str]
    check_zero_exptime_repair: bool


def assess_key_science_files_data_quality(scifiles, shortnames):
    """
    Evaluate TAG/RAW science data and primary-header flags for unusable conditions.

    Parameters
    ----------
    scifiles : sequence of path-like
        Resolved paths to science FITS (same order as shortnames).
    shortnames : sequence of str
        Basenames used in log messages (must align with scifiles).

    Returns
    -------
    KeyScienceDataQualityAssessment
    """
    if len(scifiles) == 0:
        return KeyScienceDataQualityAssessment(
            reject=True,
            reason=reasons_menu['no data'],
            odd_expflag=None,
            check_zero_exptime_repair=False,
        )

    pieces = dbutils.parse_filename(scifiles[0])
    reject = False
    reason = ''

    if 'tag' in pieces['type']:
        counts = 0
        for file_info in scifiles:
            if 'x1d' in Path(file_info).name:
                continue
            with fits.open(file_info) as h:
                counts += len(h[1].data['time'])
            if counts <= 100:
                reject = True
                reason = reasons_menu['no data']
    elif 'raw' in pieces['type']:
        exptimes = [fits.getval(f, 'exptime', 1) for f in scifiles]
        if np.all(np.array(exptimes) == 0):
            reject = True
            reason = reasons_menu['no data']

    odd_expflag = None
    with fits.open(scifiles[0]) as h:
        hdr = h[0].header + h[1].header
        if hdr['FGSLOCK'] != 'FINE':
            reject = True
            reason = reasons_menu['no gs lock']
        if hdr['expflag'] == 'NO DATA':
            reject = True
            reason = reasons_menu['no data']
        elif hdr['expflag'] == 'SHUTTER CLOSED':
            reject = True
            reason = reasons_menu['shutter closed']
        elif hdr['expflag'] != 'NORMAL':
            odd_expflag = hdr['expflag']

        check_zero_exptime_repair = hdr['exptime'] == 0 and not reject

    return KeyScienceDataQualityAssessment(
        reject=reject,
        reason=reason,
        odd_expflag=odd_expflag,
        check_zero_exptime_repair=check_zero_exptime_repair,
    )


def repair_zero_exptime_from_photon_times(scifiles, shortnames):
    """
    When primary EXPTIME is zero but extension 1 has photon times, update EXPTIME/TEXPTIME
    and GTI extension 2 from first/last times. Returns a note string for the observation table.
    """
    note = ''
    for shortname, file_info in zip(shortnames, scifiles):
        with fits.open(file_info, mode='update') as h:
            if len(h[2].data['start']):
                raise NotImplementedError
            if h[1].header['exptime'] == 0:
                start, stop = h[1].data['time'][[0, -1]]
                data = np.recarray((1,), dtype=[('START', 'f8'), ('STOP', 'f8')])
                data['START'] = start
                data['STOP'] = stop
                h[2].data = data
                h[1].header['EXPTIME'] = stop - start
                h[0].header['TEXPTIME'] = stop - start
                h.flush()
                note += (
                    f'{shortname} had data but header set to zero exposure time. '
                    'Manually replaced GTIs based on first and last photon count.'
                )
            else:
                raise ValueError('Weird. Look into this.')
    return note


def plot_acq_image(fits_handle, object_coords, figure, subplot_spec, zoom_region=None):
    h = fits_handle

    newobstime = Time(h.header['expstart'], format='mjd')
    coords_at_obs = object_coords.apply_space_motion(newobstime)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="'datfix'")
        wcs = WCS(h.header)
    ax = figure.add_subplot(*subplot_spec, projection=wcs)

    ax.imshow(h.data, origin='lower')
    ax.coords.grid(True, color='white', ls=':', lw=0.5)

    if zoom_region is not None:
        ra, dec = coords_at_obs.ra, coords_at_obs.dec
        coord1 = SkyCoord(ra - zoom_region, dec - zoom_region)
        coord2 = SkyCoord(ra + zoom_region, dec + zoom_region)

        # Convert to pixel coordinates
        (x1, y1) = wcs.world_to_pixel(coord1)
        (x2, y2) = wcs.world_to_pixel(coord2)

        # avoid reversed pixel coords
        xlo = min(x1, x2)
        xhi = max(x1, x2)
        ylo = min(y1, y2)
        yhi = max(y1, y2)

        # avoid skinny images
        dx = xhi - xlo
        dy = yhi - ylo
        dmx = max(dx, dy)
        if dx < dmx/2:
            xlo -= dmx/2
            xhi += dmx/2
        if dy < dmx/2:
            ylo -= dmx/2
            yhi += dmx/2

        # Set the limits using pixel coordinates
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    # get just the displayed array to return
    j0, j1 = [int(round(x)) for x in ax.get_xlim()]
    i0, i1 = [int(round(x)) for x in ax.get_ylim()]
    ary = h.data[int(i0):i1, j0:j1]

    return ax, coords_at_obs, ary
