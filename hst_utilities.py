import re
import warnings
from pathlib import Path
from typing import NamedTuple, Optional
import io
import contextlib

import numpy as np
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


def auto_validate_stis_acq(acq_path, verbosity=1):
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

    return msgs


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