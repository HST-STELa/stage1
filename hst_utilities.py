import warnings

from astroquery.mast import MastMissions
from astropy.io import fits
from astropy import time
from astropy import units as u

import database_utilities as dbutils

hst_database = MastMissions(mission='hst')

def locate_associated_acquisitions(path, additional_files=()):
    max_visit_length = 10*u.h
    h = fits.open(path)
    hdr = h[0].header + h[1].header
    pieces = dbutils.parse_filename(path)

    # a gotcha is that sometimes the acquisitions are labeled as a separate visit
    # (see DS Tuc A visits OE8T01 and OE8TA1)
    # so instead I will base the search on looking for any exposures within a visit length of time that have ACQ in mode
    before = time.Time(hdr['expstart'], format='mjd')
    after = before - max_visit_length
    date_search_str = f'{after.iso}..{before.iso}'

    # make sure observations are from the same program
    id = pieces['id']
    id_searchstr = id[:4] + '*' # all files with this root will be from the same visit
    results = hst_database.query_criteria(sci_data_set_name=id_searchstr,
                                          sci_start_time=date_search_str,
                                          sci_operating_mode='*ACQ*',
                                          select_cols=['sci_operating_mode', 'sci_start_time'])
    if len(results) == 0:
        warnings.warn(f'No acquitions found for {path.name}')
        return []
    datasets = hst_database.get_unique_product_list(results)
    filtered = hst_database.filter_products(datasets, file_suffix=['RAW', 'RAWACQ'] + list(additional_files))

    results.add_index('sci_data_set_name')
    filtered['obsmode'] = results.loc[filtered['dataset']]['sci_operating_mode']
    filtered['start'] = results.loc[filtered['dataset']]['sci_start_time']

    return filtered