import re
import warnings
from math import nan

import numpy as np
from astropy import table, coordinates as coord, time, units as u
from tqdm import tqdm
from scipy.spatial import KDTree
import pandas as pd

import paths


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

        tic_ids = tic_ids.data.reshape((-1,1))
        kd = KDTree(tic_ids)
        kds.append(kd)
    return kds[1].query_ball_tree(kds[0], 0.5)


def convert_simbad_coords(simbad):
    radec = coord.SkyCoord(simbad['RA'], simbad['DEC'], unit=('hourangle', 'deg'))
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


def read_excel(path):
    tbl = pd.read_excel(path, keep_default_na=False)
    tbl = table.Table.from_pandas(tbl)
    return tbl


def add_masks(catalog):
    for name in catalog.colnames:
        if not hasattr(catalog[name], 'mask'):
            catalog[name] = table.MaskedColumn(catalog[name])


def loc_indices_and_unmatched(catalog, values):
    index_column = catalog.indices[0].columns[0]
    has_match = np.in1d(values, index_column)
    i_matched, = np.nonzero(has_match)
    i_unmatched, = np.nonzero(~has_match)
    idx = catalog.loc_indices[values[has_match]]
    return idx, i_matched, i_unmatched