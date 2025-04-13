import warnings

import numpy as np
import pyvo as vo
from astroquery.simbad import Simbad
from astroquery.exceptions import BlankResponseWarning
from astropy import coordinates as astrocoord
from astropy import units as u
from astropy import table

import catalog_utilities as catutils


def get_simbad_from_names(names, extra_cols=(), suppress_blank_response_warnings=False):
    """
    For info on finding available columns, see
    https://astroquery.readthedocs.io/en/stable/simbad/simbad.html#specifying-which-votable-fields-to-include-in-the-result
    """
    simquery = Simbad()
    extra_cols = ['typed_id'] + list(extra_cols) # typed_id essential or else a full list will not be returned
    extra_cols = list(set(extra_cols)) # remove duplicates
    simquery.add_votable_fields(*extra_cols)
    with warnings.catch_warnings():
        ignore = (UserWarning, BlankResponseWarning) if suppress_blank_response_warnings else UserWarning
        warnings.simplefilter('ignore', ignore)
        simbad = simquery.query_objects(list(names))

    # mask values in rows where there was no object found and add a flag column for whether found
    mask = simbad['MAIN_ID'] == ''
    for col in simbad.columns.values():
        if col.name != 'TYPED_ID':
            col.mask = mask
    simbad['simbad_match'] = table.MaskedColumn(~mask, mask=False)

    ra, dec = catutils.convert_simbad_coords(simbad)
    simbad['ra'] = ra
    simbad['dec'] = dec
    return simbad


def get_simbad_from_tic_id(tic_id_column, extra_cols=()):
    names = tic_id_column.astype('str').filled('?')
    names = np.char.add('TIC ', names)
    result = get_simbad_from_names(names=names, extra_cols=extra_cols)
    return result


def get_simbad_by_id(id_list, cols=()):
    """
    This will raise an error if one of the IDs is not found, so it isn't very helpful at the moment.
    Keeping it because it might be helpful later for collecting Teff or other properties.

    For available columns see
    https://simbad.cds.unistra.fr/simbad/tap/tapsearch.html
    This function will access anything from basic, mesFe_H, and mesDistance.
    """
    # cols = ['basic.ra', 'basic.dec', 'otype', 'main_id'] # for troubleshooting
    select_str = ', '.join(cols)

    id_elements = [f"('{id}')" for id in id_list]
    id_str = ', '.join(id_elements)

    service = vo.dal.TAPService('https://simbad.cds.unistra.fr/simbad/sim-tap/')
    querystr = f"""SELECT {select_str}
                 FROM basic 
                 JOIN ident ON ident.oidref = oid
                 LEFT JOIN mesDistance ON mesDistance.oidref = oid
                 LEFT JOIN mesFe_H ON mesFe_H.oidref = oid
                 WHERE id IN ({id_str});"""
    resultset = service.search(querystr)
    simbad = resultset.to_table()
    return simbad

def get_simbad_3d(catalog, search_radius, cols=()):
    """
    For available columns see
    https://simbad.cds.unistra.fr/simbad/tap/tapsearch.html
    This function will access anything from basic, mesFe_H, and mesDistance.
    """
    query_cols = ['basic.ra', 'basic.dec', 'mesDistance.dist'] + list(cols)
    # query_cols = ['basic.ra', 'basic.dec', 'otype'] # for troubleshooting
    query_cols = list(set(query_cols))
    select_str = ', '.join(query_cols)

    coords_cat2000 = catutils.J2000_positions(catalog)
    coords_tbl = table.Table((coords_cat2000.ra, coords_cat2000.dec),
                             names=['ra', 'dec'])

    service = vo.dal.TAPService('https://simbad.cds.unistra.fr/simbad/sim-tap/')
    querystr = f"""SELECT {select_str}
                FROM basic 
                LEFT JOIN mesDistance ON mesDistance.oidref = oid
                LEFT JOIN mesFe_H ON mesFe_H.oidref = oid, 
                TAP_UPLOAD.cat as cat
                WHERE CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', cat.ra, cat.dec, {search_radius.to_value('deg')})) = 1
                  AND otype = 'star..';"""
    resultset = service.search(querystr, uploads=dict(cat=coords_tbl))
    simbad = resultset.to_table()

    # keep only results with a distance
    has_distance = ~simbad['dist'].mask
    simbad = simbad[has_distance]

    dist_simbad = simbad['dist'].filled(np.nan) * u.pc
    pos_simbad = astrocoord.SkyCoord(simbad['ra'], simbad['dec'], distance=dist_simbad)
    pos_cat2000 = astrocoord.SkyCoord(coords_cat2000, distance=catalog['sy_dist'])
    idx, _, dist  = pos_cat2000.match_to_catalog_3d(pos_simbad)

    matched_simbad = simbad[idx]
    return matched_simbad, dist


def get_TIC_by_id(catalog, cols=('ID', 'GAIA')):
    select_str = ', '.join(cols)
    service = vo.dal.TAPService('https://mast.stsci.edu/vo-tap/api/v0.1/tic/')

    # query in chunks or else query gets too large
    chunksize = 300
    a, b = 0, chunksize
    result_tables = []
    while b < len(catalog):
        print(f"Pulling TIC info for entries {a}:{b} of {len(catalog)}")
        tic_ids = catalog['tic_id'][a:b].tolist()
        # tic_ids = ','.join(map(str, tic_ids))
        tic_ids = [f'({id})' for id in tic_ids]
        tic_ids_str = ', '.join(tic_ids)
        # I tried uploading a table of tic ids, but couldn't get it to work.
        # I think maybe the TIC TAP service does not allow for uploads
        querystr = f"""SELECT {select_str}
                    FROM dbo.CatalogRecord as tic
                    WHERE ID IN ({tic_ids_str})
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            resultset = service.search(querystr)
        result_table = resultset.to_table()
        result_tables.append(result_table)

        a = b
        b += chunksize
    result_table = table.vstack(result_tables)

    # put in the same order as the input catalog
    result_table.add_index('ID')
    result_table = result_table.loc[catalog['tic_id']]
    result_table.remove_indices('ID')

    return result_table


def filter_available_exoarchive_columns(table_name, columns, add_err_and_lim=True):
    service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    all_columns = service.search(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name LIKE '{table_name}'")
    # select only columns from the columns.retrieve list that are available for the indicated table
    available_columns = []
    for col in columns:
        if add_err_and_lim:
            col_err_lim = [col, col + 'err1', col + 'err2', col + 'lim']
            for name in col_err_lim:
                if name in all_columns['column_name']:
                    available_columns.append(name)
        else:
            if col in all_columns:
                available_columns.append(col)
    return available_columns


def pull_exoarchive_catalog(table_name, columns):
    service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    column_string = ','.join(columns)
    resultset = service.search(f'SELECT {column_string} FROM {table_name}')
    catalog = resultset.to_table()
    return catalog