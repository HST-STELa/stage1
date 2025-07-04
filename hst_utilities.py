import warnings
from pathlib import Path

import numpy as np
from astroquery.mast import MastMissions
from astropy.io import fits
from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord

import database_utilities as dbutils
import utilities as utils

hst_database = MastMissions(mission='hst')

def locate_associated_acquisitions(path, additional_files=()):
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
    before = time.Time(hdr['expstart'], format='mjd')
    after = before - max_visit_length
    date_search_str = f'{after.iso}..{before.iso}'

    # make sure observations are from the same program
    id = pieces['id']
    id_searchstr = id[:6] + '*' # all files with this root will be from the same observation set
    results = hst_database.query_region(
        coords,
        radius=0.1,
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

def is_raw_science(file):
    file = Path(file)
    pieces = dbutils.parse_filename(file.name)
    if 'tag' in pieces['type']:
        return True
    if 'raw' in pieces['type']:
        if 'hst-cos' in pieces['config']:
            return False
        mode = fits.getval(file, 'obsmode')
        if mode == 'ACCUM':
            return True
        else:
            return False
    else:
        return False



def infer_count2flux_factor(flux, window_size=50, fuzz_factor=0.1):
    """
    Uses the quantization of the fluxes to estimate what the scaling factor is over a sliding window.

    A gotcha is that cos shifts the floor down to give zero average background, so this will give negative counts
    in many spots. I'm still not sure it really works.
    """
    estimated_counts = np.zeros_like(flux)

    from numpy.lib.stride_tricks import sliding_window_view

    # get sets of differences in sorted flux values along sliding windows
    # these will be appx quantized, enabling me to estimate qunatization scale or "step"
    windows = sliding_window_view(flux, window_shape=(window_size,1))
    windows = windows.squeeze()
    sorted = np.sort(windows, axis=2)
    diffs = np.diff(sorted, axis=2)
    max_diff = np.max(diffs, axis=2)
    min_diff= np.min(diffs, axis=2)
    if np.any(max_diff*fuzz_factor < min_diff):
        n_bad_windows = np.sum(max_diff*fuzz_factor < min_diff)
        raise ValueError(f'Fluxes in {n_bad_windows} do not appear quantized. This may indicate a position in '
                         f'the flux array has a window that countains fluxes which are all appx the same value. '
                         f'Consider a larger window size.')

    # infer the step size based on the median of values that are above a floor
    diff_floor = max_diff*fuzz_factor
    above_floor = diffs >= diff_floor[:,:,None]
    masked_diffs = np.ma.array(diffs, mask=~above_floor)
    step_estimates = np.ma.median(masked_diffs, axis=-1)

    # fill missing values with interpolation
    # I considered fitting a polynomial to handle any missing or anomalous bad step estimates and extrapolate to ends
    # but this does not work well for coadded spectra where the number of couns depends on varying exposure time
    # also grid wires and such
    stepsizes = []
    ivec = np.arange(step_estimates.shape[0])
    for i in range(step_estimates.shape[-1]):
        stepvec = step_estimates[:,i]
        missing = stepvec.mask | (stepvec <= 0) # the zero check handles cos detector gaps
        if np.any(missing):
            fillvalues = np.interp(ivec[missing], ivec[~missing], stepvec[~missing])
            stepvec[missing] = fillvalues
        stepsizes.append(stepvec)
    stepsizes = np.array(stepsizes).T

    # put in array with masked values at the ends
    counts2flux_factor = np.ma.masked_all_like(flux)
    center_slice = utils.sliding_center_slice(flux[:,0], window_size)
    counts2flux_factor[center_slice,:] = stepsizes

    return counts2flux_factor
