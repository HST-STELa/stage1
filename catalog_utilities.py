import re
import warnings
from math import nan
from contextlib import contextmanager

import numpy as np
from astropy import table, coordinates as coord, time, units as u
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
from tqdm import tqdm
from scipy.spatial import KDTree
import pandas as pd

import paths

limit_int2str = {-1:'>', 0:'', 1:'<'}

def isnull(table_value, null_value):
    if hasattr(table_value, 'mask') and table_value.mask:
        return True
    else:
        return table_value == null_value


def is_positive_real(catalog, colname):
    if hasattr(catalog[colname], 'mask'):
        return ~catalog[colname].mask & (catalog[colname].filled(-999) > 0)
    else:
        return catalog[colname] > 0


def mask_unphysical(catalog, colname, must_be_positive=False):
    nancol = catalog[colname].filled(np.nan)
    unphysical = ~np.isfinite(nancol)
    if must_be_positive:
        unphysical = unphysical | (nancol <= 0)
    catalog[colname].mask = unphysical


def filter_bad_values(catalog, colname, fill_value, filter_func):
    vals = catalog[colname].filled(fill_value)
    remove = filter_func(vals)
    iremove, = np.nonzero(remove)
    catalog.remove_rows(iremove)


def tqdm_rows(tab):
    i_row_pairs = list(enumerate(tab))
    return tqdm(i_row_pairs)


def load_and_mask_ecsv(path):
    """
    Solves the problem of columns with no masked values being loaded with no mask attribute.

    Note that object data types *do* round trip, so if you're experiencing a column having dtype string when it
    should have dtype object, it's not because of saving and loading a table.
    """
    catalog = table.Table.read(path)
    add_masks(catalog)
    return catalog


def read_hand_checked_planets(keys=('remove', 'vetted')):
    filepaths = []
    for key in keys:
        newpaths = paths.checked.glob(f'*{key}*.txt')
        filepaths.extend(newpaths)
    ids = []
    for filepath in filepaths:
        _ids = np.loadtxt(filepath, dtype=object, delimiter=',')
        if _ids.ndim == 0:
            _ids = _ids[None]
        ids.extend(_ids.tolist())
    return np.asarray(ids)


def read_requested_targets(path):
    with open(path) as f:
        lines = f.readlines()
    name, = re.findall('name:? ?(.+)\n', lines[0])
    description, = re.findall('description:? ?(.+)\n', lines[1])
    description = f'{path.name}: {description}'
    target_names = [re.sub('(.+ ?-> ?)?(.+)\n?', r'\2', s) for s in lines[2:]]
    col = table.Column(name=name, description=description, data=target_names, dtype=object)
    return col


def requested_target_lists_loop(operation_function):
    list_paths = list(paths.requested.glob('*.txt'))
    results = []
    for path in list_paths:
        targets = read_requested_targets(path)
        results.append(operation_function(targets))
    return results


def safe_interp_table(xnew, oldxcol, oldycol, table):
    usable = np.ones(len(table), bool)
    for col in [oldxcol, oldycol]:
        if hasattr(table[col], 'mask'):
            usable = usable & ~table[col].mask
    slim = table[usable]
    slim.sort(oldxcol)
    return np.interp(xnew, slim[oldxcol], slim[oldycol], left=np.nan, right=np.nan)


def has_index(catalog, index_name):
    names = [x.columns[0].name for x in catalog.indices]
    return index_name in names


def add_index(catalog, index_name):
    if not has_index(catalog, index_name):
        catalog.add_index(index_name)


def match_by_tic_id(catalog0, catalog1):
    # these will be used to fill entries without tic_ids to avoid errors and ensure they don't match to anything
    unmatchable_ids = (n for n in range(-10, -10000, -10))

    kds = []
    for cat in [catalog0, catalog1]:
        tic_ids = cat['tic_id'].astype('float')

        # fill with made up, unmatchable values if needed
        if hasattr(tic_ids, 'filled'):
            tic_ids = tic_ids.filled(np.nan)
        bad = ~np.isfinite(tic_ids)
        tic_ids[bad] = [next(unmatchable_ids) for _ in range(sum(bad))]

        tic_ids = tic_ids.data.reshape((-1, 1))
        kd = KDTree(tic_ids)
        kds.append(kd)
    return kds[1].query_ball_tree(kds[0], 0.5)


def convert_simbad_coords(simbad):
    radec = coord.SkyCoord(simbad['ra'], simbad['dec'], unit=('hourangle', 'deg'))
    return radec.ra, radec.dec


def scrub_indices(cat):
    name_sets = []
    for index in cat.indices:
        names = [col.name for col in index.columns]
        cat.remove_indices(*names)
        name_sets.extend(names)
    return name_sets


def set_index(cat, name):
    scrub_indices(cat)
    cat.add_index(name)


def add_masked_row(catalog):
    catalog.add_row()
    for name in catalog.colnames:
        catalog[name].mask[-1] = True


def add_filled_masked_column(catalog, colname, fill_value, **kws):
    catalog[colname] = table.MaskedColumn(length=len(catalog), **kws)
    catalog[colname] = fill_value


def add_masked_columns(catalog, basename, dtype, suffixes=('', 'err1', 'err2', 'lim'), **kws):
    for suffix in suffixes:
        colname = basename + suffix
        catalog[colname] = table.MaskedColumn(name=colname, length=len(catalog), dtype=dtype, mask=True, **kws)


def transfer_values(targettbl, targetname, srctable, srcname, targ_mask=None, src_mask=None,
                    targ_suffixes=('', 'err1', 'err2', 'lim'), src_suffixes=('',)):
    if targ_mask is None and src_mask is not None:
        raise ValueError('Either both or neither targ_mask and src_mask need to be specified.')
    if src_mask is None and targ_mask is not None:
        raise ValueError('Either both or neither targ_mask and src_mask need to be specified.')
    if targ_mask is None:
        tname = targetname + targ_suffixes[0]
        targ_bad = targettbl[tname].mask
        sname = srcname + src_suffixes[0]
        src_good = ~srctable[sname].mask
        transfer = targ_bad & src_good
        targ_mask = src_mask = transfer
    for targx, srcx in zip(targ_suffixes, src_suffixes):
        tname = targetname + targx
        sname = srcname + srcx
        targettbl[tname][targ_mask] = srctable[sname][src_mask]


def make_a_cut(catalog, mask_col, keepers=('stage1_locked_target', 'requested_target', 'flag_multiplanet')):
    """
    Cuts targets according to the flags in cut_col while keeping community targets and printing stats.
    """
    passing = catalog[mask_col]
    assert not np.any(passing.mask)
    passing = passing.filled(False)
    keeper_mask = np.zeros(len(catalog), bool)
    for name in keepers:
        _mask = catalog[name]
        assert not np.any(_mask.mask)
        keeper_mask = keeper_mask | _mask.filled(False)
    either = passing | keeper_mask
    print(f'{np.sum(passing)}/{len(catalog)} targets meet the cut criteria.')
    keeping = keeper_mask & ~passing
    print(f'{np.sum(keeping)} additional targets kept because of a True value in {keepers}.')
    return catalog[either]


def J2000_positions(catalog):
    ra, dec = catalog['ra'].filled(nan), catalog['dec'].filled(nan)
    pmra, pmdec = catalog['sy_pmra'].filled(0), catalog['sy_pmdec'].filled(0)
    obstime_cat = time.Time('J2015.5')
    # as of 2024-09-26 the coordinates and PMs in the confirmed planets and TOI catalogs are from GAIA DR2 which uses J2015.5
    coords_cat = coord.SkyCoord(ra, dec, pm_ra_cosdec=pmra, pm_dec=pmdec, obstime=obstime_cat)

    # correct positions to J2000 for simbad search
    obstime_simbad = time.Time('J2000')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        coords_cat2000 = coords_cat.apply_space_motion(obstime_simbad)
    assert np.all(np.isfinite(coords_cat2000.ra))

    return coords_cat2000


def flag_cut(catalog, mask, decision_str, colname='stage1'):
    cut_mask = mask & catalog[colname].filled(True)
    n_mask = np.sum(mask)
    n_cut = np.sum(cut_mask)
    catalog['decision'][cut_mask] = decision_str
    catalog[colname][cut_mask] = False
    print(f"Regarding {decision_str}"
          f"\n\t{n_mask} entries elgible to be cut. "
          f"\n\t{n_mask - n_cut} already flagged to cut for other reasons. "
          f"\n\t{n_cut} new entries flagged for cutting."
          f"\n\t{sum(catalog[colname])} entries remain eligible.")


def find_duplicates(catalog, distlimit=0.2*u.pc):
    fake_distance = (1000 + catalog['pl_orbper'].data) * u.pc
    positions = coord.SkyCoord(catalog['ra'], catalog['dec'], fake_distance)
    i_match, _, _, _ = positions.search_around_3d(positions, distlimit=distlimit)
    _, i_match_unique, match_count = np.unique(i_match, return_index=True, return_counts=True)
    duplicated = match_count > 1
    return positions, duplicated


def unique_star_indices(cat):
    assert not np.any(cat['tic_id'].mask)
    _, i = np.unique(cat['tic_id'], return_index=True)
    return sorted(i)


def pick_planet_parameters(cat, colname, picking_function=np.max, hostcolname='default'):
    """adds a column with the name '{col}_host' that gives the parameters of only one planet"""
    assert not np.any(cat['tic_id'].mask)
    set_index(cat, 'tic_id')
    hostcol = f'{colname}_host' if hostcolname == 'default' else hostcolname
    oldcol = cat[colname]
    cat[hostcol] = table.MaskedColumn(length=len(cat), dtype=cat[colname].dtype, mask=True)
    tic_ids = np.unique(cat['tic_id'].filled(0))
    for tic_id in tic_ids:
        i_planets = cat.loc_indices[tic_id]
        top_score = picking_function(oldcol[i_planets])
        cat[hostcol][i_planets] = top_score


def planets2hosts(cat):
    cols = [col for col in cat.colnames if not col.startswith('pl_')]
    return cat[cols][unique_star_indices(cat)]


def empty_table(length, names, dtypes):
    MC = lambda dtype: table.MaskedColumn(length=length, dtype=dtype, mask=True)
    cols = list(map(MC, dtypes))
    return table.Table(cols, names=names)


def unmatched_names(names_to_match, names_matching_into):
    matches = np.in1d(names_to_match, names_matching_into)
    umatched_names = names_to_match[~matches]
    return umatched_names


def match_by_position(ra1, dec1, ra2, dec2):
    coord1 = coord.SkyCoord(ra1, dec1)
    coord2 = coord.SkyCoord(ra2, dec2)
    i1, i2, _, _ = coord2.search_around_sky(coord1)
    return i1, i2


def read_excel(path, *args, **kwargs):
    tbl = pd.read_excel(path, *args, keep_default_na=False, **kwargs)
    tbl = table.Table.from_pandas(tbl)
    return tbl


def add_masks(catalog, inplace=True):
    if inplace:
        for name in catalog.colnames:
            if not hasattr(catalog[name], 'mask'):
                catalog[name] = table.MaskedColumn(catalog[name])
    else:
        cols = [table.MaskedColumn(catalog[name]) for name in catalog.colnames]
        return table.Table(cols, meta=catalog.meta)


def loc_indices_and_unmatched(catalog, values):
    index_column = catalog.indices[0].columns[0]
    has_match = np.in1d(values, index_column)
    i_matched, = np.nonzero(has_match)
    i_unmatched, = np.nonzero(~has_match)
    idx = catalog.loc_indices[values[has_match]]
    return idx, i_matched, i_unmatched


def get_value_or_col_filled(key, tbl_or_row, fillvalue=nan):
    isrow = isinstance(tbl_or_row, table.Row)
    x = tbl_or_row[key]
    if isrow:
        if np.ma.is_masked(x):
            return fillvalue
        else:
            return x
    else: # it's a table
        return x.filled(fillvalue).copy()


def get_quantity_flexible(key, tbl_or_row, tbl=None, fill=False, fillvalue=nan):
    isrow = isinstance(tbl_or_row, table.Row)
    if fill:
        x = get_value_or_col_filled(key, tbl_or_row, fillvalue)
    else:
        x = tbl_or_row[key].copy()
    if isrow:
        if np.ma.is_masked(x):
                return x
        if hasattr(x, 'unit'):
            return u.Quantity(float(x.value), x.unit)
        else:
            if tbl is None:
                raise ValueError('If you want to get a value with units from a row, either supply the source table'
                                 'so the units can be determined from the table or use a row from an astropy QTable'
                                 'instead of a Table.')
            return x * tbl[key].unit
    else:
        if isinstance(x, u.Quantity):
            return x
        elif hasattr(x, 'quantity'):
            return x.quantity
        else:
            raise ValueError('Column neither is a quantity or has a quantity attribute.')


def merge_tables_with_update(old, new, key):
    """
    Merge `new` into `old`, updating existing rows, adding new rows and columns. Generated by ChatGPT.

    Parameters
    ----------
    old : astropy.table.Table
        Original table.
    new : astropy.table.Table
        New table with possible updates and additions.
    key : str
        Name of the unique key column used for matching rows.

    Returns
    -------
    merged : astropy.table.Table
        Combined table with updates applied.
    """
    # Ensure both tables have the key and set it as the primary key
    if key not in old.colnames or key not in new.colnames:
        raise ValueError(f"Both tables must contain the key column '{key}'")

    # Identify new columns
    all_columns = set(old.colnames) | set(new.colnames)
    new_columns_only = set(new.colnames) - set(old.colnames)
    old_columns_only = set(old.colnames) - set(new.colnames)

    # Add missing columns to each table, masking values
    for col in new_columns_only:
        old[col] = table.MaskedColumn([None] * len(old), mask=[True] * len(old))

    for col in old_columns_only:
        new[col] = table.MaskedColumn([None] * len(new), mask=[True] * len(new))

    # Ensure same column order
    new = new[old.colnames]

    # Use 'join' to update common rows; defaults to using the key
    updated = table.join(old, new, keys=key, join_type='outer', table_names=('old', 'new'), uniq_col_name='{col}_{table}')

    # Prepare merged result by preferring new values when available
    merged = table.Table(masked=True)
    for col in old.colnames:
        col_old = f'{col}_old'
        col_new = f'{col}_new'
        if col_new in updated.colnames and col_old in updated.colnames:
            values = updated[col_new].copy()
            mask = updated[col_new].mask if hasattr(updated[col_new], 'mask') else [False] * len(values)

            # Where new is masked, fall back to old
            for i in range(len(values)):
                if mask[i]:
                    values[i] = updated[col_old][i]
                    mask[i] = updated[col_old].mask[i] if hasattr(updated[col_old], 'mask') else False
            merged[col] = table.MaskedColumn(values, mask=mask)
        elif col_old in updated.colnames:
            merged[col] = updated[col_old]
        else:
            merged[col] = updated[col_new]

    return merged


def harmonize_dtypes_for_merge(tables, copy=True):
    """
    Given a sequence of astropy Tables that you intend to merge,
    coerce common-name columns to a common dtype to avoid TableMergeError
    from incompatible column dtypes.

    - Uses the least-general numeric dtype that can hold all values
      (via numpy's type promotion rules).
    - For mixed or non-numeric types, falls back to a common Unicode
      string dtype large enough to hold all stringified values.

    Parameters
    ----------
    tables : sequence of astropy.table.Table
        Tables to harmonize.
    copy : bool, optional
        If True (default), work on copies and return new tables.
        If False, modify the input tables in-place.

    Returns
    -------
    list of astropy.table.Table
        Dtype-harmonized tables (either new or the originals).
    """
    if copy:
        tables = [t.copy() for t in tables]

    if not tables:
        return tables

    # all columns
    all_cols = set(tables[0].colnames)
    for t in tables[1:]:
        all_cols |= set(t.colnames)

    for name in all_cols:
        relevant_tables = [t for t in tables if name in t.colnames]
        cols = [t[name] for t in relevant_tables]
        dtypes = [c.dtype for c in cols]

        # If they're already all identical, nothing to do
        if all(dt == dtypes[0] for dt in dtypes):
            continue

        kinds = {dt.kind for dt in dtypes}

        # any object -> use object
        if 'O' in kinds:
            target_dtype = 'object'

        # All numeric (ints/floats/bools) -> use numpy's promotion
        elif kinds <= set("iufb"):
            target_dtype = np.result_type(*dtypes)

        # all strings -> use numpy's promotion
        elif kinds == set("U"):
            target_dtype = np.result_type(*dtypes)

        # fall back to object
        else:
            target_dtype = 'object'

        # Apply conversion
        for t in tables:
            t[name] = t[name].astype(target_dtype)

    return tables


def table_vstack_flexible_shapes(tables, join_type='outer', copy=True):
    """Stacks tables while avoiding shape and type mismatch errors, at the cost of possibly creating frankentables."""

    tables = harmonize_dtypes_for_merge(tables, copy=copy)

    allnames = set()
    for tbl in tables:
        allnames |= set(tbl.colnames)

    for name in allnames:
        shapes = [tbl[name].shape[1:] for tbl in tables if name in tbl.colnames]
        # the shape[1:] is because the first dim is just the length of the table, and it's fine if those vary

        # is some tables have conflicting shapes recast as object type with no shape restriction
        if len(set(shapes)) > 1:
            for tbl in tables:
                if name in tbl.colnames:
                    x = table.Column(list(tbl[name]) + [nan], dtype='object')
                    tbl[name] = x[:-1]

    return table.vstack(tables, join_type=join_type)

@contextmanager
def catch_QTable_unit_warnings():
    """
    Suppress the noisy Astropy warning:
      '... has a unit but is kept as a MaskedColumn ...'
    that can be emitted when constructing QTables.

    Usage:
        with catch_QTable_unit_warnings():
            planets = table.QTable(planets)
            hosts = table.QTable(hosts)
    """
    with warnings.catch_warnings():
        # Start from default filters inside this context
        warnings.simplefilter("default")

        # Limit to astropy.table module & Astropy warnings to avoid hiding unrelated issues
        shared_kws = dict(
            action="ignore",
            message=r".*has a unit but is kept as a (?:Masked)?Column"
            )
        warnings.filterwarnings(**shared_kws, category=AstropyWarning)
        warnings.filterwarnings(**shared_kws, category=AstropyUserWarning)

        yield


def download_escape_catalog():
    cat = read_excel(paths.escape_catalog_google_sheet_xlsx_export, header=2)
    return cat


def escape_catalog_merge_targets(cat='download'):
    if cat == 'download':
        cat = download_escape_catalog()
    grouped = cat.group_by('Planet Name')

    id_cols = grouped[['Target Star', 'Planet Letter', 'Planet Name', 'TIC ID']]
    agg_id = id_cols.groups.aggregate(min)

    det_cols = grouped[['He* Detected?', 'H-alpha detected?', 'Lya detected?']]
    def detection_label_joiner(x):
        xx = set(x)
        xx = xx - {'', ' '}
        result = ', '.join(xx)
        return result
    agg_det = det_cols.groups.aggregate(detection_label_joiner)

    result = table.hstack((agg_id, agg_det))
    return result


def filter_table(tbl, column_value_dictionary, copy=True):
    """Filter a table to select only those rows with the provided values in each column as specified by the
    column_value_dictionary."""
    mask = np.ones(len(tbl), bool)
    for name, val in column_value_dictionary.items():
        newmask = tbl[name] == val
        mask &= newmask
    filtered = tbl[mask]
    if copy:
        filtered = filtered.copy()
    return filtered