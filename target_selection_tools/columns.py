import re

from astropy import table
from astropy import units as u

# list columns to retrieve.
retrieve = [
    # in pscomppars and maybe toi
    'pl_name',
    'hostname',
    'pl_letter',
    'hd_name',
    'hip_name',
    'tic_id',
    'gaia_dr2_id',
    'gaia_dr3_id',
    'cb_flag',
    'discoverymethod',
    'disc_year',
    'disc_refname',
    'disc_pubdate',
    'rv_flag',
    'tran_flag',
    'pl_controv_flag',
    'pl_orbper',
    'pl_orbsmax',
    'pl_orbeccen',
    'pl_rade',
    'pl_bmasse',
    'pl_bmassprov',
    'pl_dens',
    'pl_insol',
    'pl_eqt',
    'pl_tranmid',
    'pl_tranmid_systemref',
    'pl_imppar',
    'pl_trandep',
    'pl_trandur',
    'pl_ratdor',
    'pl_ratror',
    'pl_rvamp',
    'st_spectype',
    'st_teff',
    'st_rad',
    'st_mass',
    'st_met',
    'st_metratio',
    'st_lum',
    'st_logg',
    'st_age',
    'st_dens',
    'st_vsin',
    'st_rotp',
    'st_radv',
    'rastr',
    'ra',
    'decstr',
    'dec',
    'sy_dist',
    'sy_plx',
    'sy_pnum',
    'sy_pmra',
    'sy_pmdec',
    'sy_bmag',
    'sy_vmag',
    'sy_jmag',
    'sy_hmag',
    'sy_kmag',
    'sy_umag',
    'sy_gmag',
    'sy_rmag',
    'sy_imag',
    'sy_zmag',
    'sy_w1mag',
    'sy_w2mag',
    'sy_w3mag',
    'sy_w4mag',
    'sy_gaiamag',
    'sy_icmag',
    'sy_tmag',
    'sy_kepmag',
    'pl_nnotes',
    # in toi only
    'toi',
    'toipfx',
    'tid',
    'ctoi_alias',
    'tfopwg_disp',
    'toi_created',
    'rowupdate',
    'st_dist',
    'st_pmra',
    'st_pmdec',
    'st_tmag',
    'pl_trandurh',
    'pl_pnum'
]


# list columns essential to the target selection and observation planning process
essential = ['ra',
             'dec',
             'tic_id',
             'id',
             'pl_orbper',
             'pl_rade',
             'st_teff',
             'sy_dist'
             ]


# map a few columns that exist in both the comppars table and toi table
toi2comp_map = {
    'st_dist': 'sy_dist',
    'st_pmra': 'sy_pmra',
    'st_pmdec': 'sy_pmdec',
    'st_tmag': 'sy_tmag',
    'pl_trandurh': 'pl_trandur',
    'pl_pnum': 'sy_pnum',
    'tid': 'tic_id'
}

comp2toi_map = {v: k for k, v in toi2comp_map.items()}

dtype_map = {
    'discoverymethod' : 'object',

}

units_map = {
    'Earth Radius' : u.Rearth,
    'R_Earth' : u.Rearth,
    'Earth Mass' : u.Mearth,
    'M_Earth' : u.Mearth,
    'Earth flux' : u.def_unit('Searth', 1361*u.W/u.m**2),
    'Earth Flux' : u.def_unit('Searth', 1361*u.W/u.m**2),
    'BJD' : u.d,
    'days' : u.d,
    'hours' : u.h,
    'Solar Radius' : u.Rsun,
    'R_Sun' : u.Rsun,
    'log10(cm/s**2)' : u.dex(u.cm/u.s**2),
    'log(Solar)' : u.dex(u.Lsun),
    'Solar mass' : u.Msun
}

def basename(name):
    base = re.sub('(sym)?err[12]?', '', name)
    base = base.replace('reflink', '')
    base = base.replace('lim', '')
    base = base.replace('src', '')
    return base


def rename_columns(catalog, name_map):
    for basename in name_map.keys():
        matches = filter(lambda s: basename in s, catalog.colnames)
        for fullname in matches:
            newname = fullname.replace(basename, name_map[basename])
            catalog.rename_column(fullname, newname)


def fix_units(catalog, unit_map):
    for name in catalog.colnames:
        oldunit = str(catalog[name].unit)
        if oldunit in units_map:
            catalog[name].unit = unit_map[oldunit]


def fix_dtypes(catalog, dtype_map):
    for name in catalog.colnames:
        if name in dtype_map:
            newtype = dtype_map[name]
            catalog[name] = catalog[name].astype(newtype)


def compare_columns(left_catalog, right_catalog):
    '''
    Compares the column names and data types between two tables.

    Function takes two tables as input and produces a new table where each row corresponds
    to a column name that was in either of input tables. The types of corresponding columns
    are also provided if the column name exists in the respective input table.

    Parameters
    ----------
    left_catalog : astropy.table.Table
        The left catalog to compare.
    right_catalog : astropy.table.Table
        The right catalog to compare.

    Returns
    -------
    meta_table : astropy.table.Table
        Output table with column names and their corresponding data types in the input tables.
    '''

    # Combine the column names from both catalogs
    colnames = set(left_catalog.colnames + right_catalog.colnames)
    col1 = table.Column(data=list(colnames), name='colname')

    # Create masked columns for storing the data types of the corresponding columns in the input tables
    kws = dict(dtype='object', mask=True, length=len(col1))
    col2 = table.MaskedColumn(name='left_type', **kws)
    col3 = table.MaskedColumn(name='right_type', **kws)
    col4 = table.MaskedColumn(name='left_unit', **kws)
    col5 = table.MaskedColumn(name='right_unit', **kws)

    # Create a new table to store the column names and their respective types
    meta_table = table.Table([col1, col2, col3, col4, col5])

    # Iterate through the combined list of column names
    for i, colname in enumerate(col1):
        # If column is in the left catalog, store its type in the 'left_type' column of the meta_table
        if colname in left_catalog.colnames:
            left_type = str(left_catalog[colname].dtype)
            meta_table['left_type'][i] = left_type
            left_unit = str(left_catalog[colname].unit)
            meta_table['left_unit'][i] = left_unit
        # If column is in the right catalog, store its type in the 'right_type' column of the meta_table
        if colname in right_catalog.colnames:
            right_type = str(right_catalog[colname].dtype)
            meta_table['right_type'][i] = right_type
            right_unit = str(right_catalog[colname].unit)
            meta_table['right_unit'][i] = right_unit
    # Return the metadata table
    return meta_table


def add_missing_columns(target_table, source_table):
    t = target_table
    s = source_table
    n = len(t)
    missing_cols = list(set(s.colnames) - set(t.colnames))
    for name in missing_cols:
        emptycol = empty_column_like(source_table[name], length=n)
        t.add_column(emptycol)


def empty_column_like(col, length=None):
    n = len(col) if length is None else length
    emptycol = table.MaskedColumn(name=col.name, length=n,
                                  dtype=col.dtype, unit=col.unit,
                                  mask=True)
    return emptycol


def operate_on_suffixes(table, name, operation_function):
    for suffix in ['', 'err1', 'err2', 'lim']:
        fullname = name + suffix
        if fullname in table.colnames:
            table[fullname] = operation_function(table[fullname])
