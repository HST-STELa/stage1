import warnings
import re

import requests
from urllib.parse import quote_plus
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
    extra_cols = list(extra_cols)
    extra_cols = list(set(extra_cols)) # remove duplicates
    simquery.add_votable_fields(*extra_cols)
    with warnings.catch_warnings():
        ignore = (UserWarning, BlankResponseWarning) if suppress_blank_response_warnings else UserWarning
        warnings.simplefilter('ignore', ignore)
        simbad = simquery.query_objects(list(names))

    # mask values in rows where there was no object found and add a flag column for whether found
    mask = simbad['main_id'] == ''
    for col in simbad.columns.values():
        if col.name != 'user_specified_id':
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


def query_tic_catalog_using_tic_ids(catalog, cols=('ID', 'GAIA')):
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


def query_simbad_for_tic_ids(names):
    """Get TIC IDs for list of target names. Returns an array of strings with just the TIC ID numbers.
    Note a single target can have multiple TIC IDs. If so, they will be spearated by spaces, such as '111111 111112'
    If no IDs are found, the string for that target will be empty ('')

    Helpful tip: to locate a target in the list based on its tic id, use something like
    `np.char.count(targets_id, tic_ids) > 0`
    """
    names = np.asarray(names)
    simbad = get_simbad_from_names(names, extra_cols=['ids'])
    if np.any(~simbad['simbad_match']):
        names_not_found = names[~simbad['simbad_match']]
        raise ValueError(f'No SIMBAD matches for these names: \n\t{'\n\t'.join(names_not_found)}')

    tic_ids_lst = [] # some targets have multiple tic IDs, annoyingly, so need to account for that
    for ids in simbad['ids']:
        result = re.findall(r'TIC (\d+)', ids)
        if result:
            tic_ids_lst.append(' '.join(result))
        else:
            tic_ids_lst.append('')
    tic_ids_ary = np.asarray(tic_ids_lst)

    return tic_ids_ary


def get_exoarchive_parameters_from_all_sources(pl_names, cols='*'):
    service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    column_string = ','.join(cols)
    pieces = [f"pl_name = '{name}'" for name in pl_names]
    planets_string =  ' or '.join(pieces)
    resultset = service.search(f'SELECT {column_string} FROM ps WHERE {planets_string}')
    catalog = resultset.to_table()
    return catalog


def get_default_exoarchive_name(objname: str) -> str:
    BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py?objname="
    """Return the default (canonical) name from NASA Exoplanet Archive System Aliases."""
    url = BASE + quote_plus(objname)
    data = requests.get(url, timeout=30).json()
    return data.get("manifest", {}).get("resolved_name")


def get_exofop_ephemerides(toi_pfx: str | int):
    """
    Return latest (best-available) transit ephemeris for a TOI from ExoFOP.
    Output: dict with period [d], t0_bjd_tdb [BJD_TDB], notes, and source URL.
    """
    toi_str = str(toi_pfx).replace("TOI", "").replace(" ", "")
    # normalize like "1066.01"
    if not re.match(r"^\d+(\.\d+)?$", toi_str):
        raise ValueError("TOI must look like '1066.01' or '1066'.")

    # ExoFOP will resolve by TOI number:
    base = f"https://exofop.ipac.caltech.edu/tess/target.php?toi={toi_str}"
    # The page exposes a JSON export; appending '&json' returns the page data as JSON.
    url = base + "&json"
    j = requests.get(url, timeout=30).json()

    pp_rows  = j.get("planet_parameters", [])

    ephemerides = []
    for row in pp_rows:
        name = row.get('name', '')
        toi = row.get('toi', '')
        pdate = row.get('pdate', '')
        puser = row.get('puser', '')
        if not name and not toi:
            continue
        values = [row.get(key, None) for key in 'epoch epoch_e per per_e'.split()]
        if any(value is None for value in values):
            continue
        ephemerides.append((name, toi, *map(float, values), puser, pdate))

    if not ephemerides:
        raise RuntimeError("No period/epoch found on the ExoFOP page JSON for this TOI.")

    ephemeris_table = table.Table(rows=ephemerides, names='name toi epoch epoch_e per per_e puser pdate'.split())
    return ephemeris_table