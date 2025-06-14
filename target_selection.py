#%% imports and constants
import string
import sys
import re
import warnings
from math import pi, nan
from datetime import datetime

from astropy import table
from astropy import units as u
from astropy import constants as const
from astropy import coordinates as coord
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import catalog_utilities
import paths
import catalog_utilities as catutils
from lya_prediction_tools import transit, lya
from target_selection_tools import galex_query
from target_selection_tools import duplication_checking as dc
from target_selection_tools import reference_tables as ref
from target_selection_tools import query, columns, apt, empirical
import database_utilities as dbutils

erg_s_cm2 = u.Unit('erg s-1 cm-2')


#%% settings and toggles

allocated_orbits = 204

# if true, pulls the latest versions of these files from the progress review folder:
# - visit list from status xml
# - manually udpated observation status excel sheet
# then makes sure all planned visits are kept and lemons are dropped
# whether an fuv visit is kept or dropped depends on the "pass to stage 1b?" column in the excel sheet
toggle_mid_cycle_update = True
# use thes to specify visits we are going to add back in or remove for whatever reason
# for example, in the May update I wanted to remove the K2-72 fuv visit Z2 because of an error in the clearance sheet
# and add in NB for TOI-2015 due to revised pass through cut
hand_remove_visits = 'J7'.split()
hand_add_visits = 'W1 W6 OO OP'.split()

# adds this many orbits-worth of backup targets if desired so you can get ahead on vetting them
backup_orbits = 5
# backup_orbits = int(round(0.2 * 204))

toggle_plots = True

toggle_save_outputs = True
toggle_save_galex = True
toggle_save_difftbl = True
toggle_save_visit_labels = True

toggle_redo_all_galex = False # only needed if galex search methodology modified
toggle_remake_filtered_hst_archive = True  # only needed if archive file redownloaded

diff_label = 'target-backfill-2025-06'
toggle_checkpoint_saves = True
toggle_target_removal_test = False # removes targets to see if sort order changes as a test for bugs
assumed_transit_range = [-100, 50] # based on typical ranges from actual transit observations
default_sys_rv = "ism"


#%% load old merged table (if available)

"""This is needed in scattered places even if you skip the block that creates it, so it is wise to load it."""
merged_path = paths.selection_intermediates / 'chkpt1__merged-confirmed-tois-community.ecsv'
if merged_path.exists():
    merged = catutils.load_and_mask_ecsv(merged_path)


#%% Pull the composite and toi catalogs.

compcols = query.filter_available_exoarchive_columns('pscomppars', columns.retrieve, add_err_and_lim=True)
confirmed = query.pull_exoarchive_catalog('pscomppars', compcols)

toicols = query.filter_available_exoarchive_columns('toi', columns.retrieve, add_err_and_lim=True)
tois = query.pull_exoarchive_catalog('toi', toicols)

#%% save catalogs

tois_uncut = tois.copy()
tois_uncut.write(paths.ipac / 'tois_uncut.ecsv', overwrite=True)
confirmed.write(paths.ipac / 'confirmed_uncut.ecsv', overwrite=True)


#%% Sanity cuts
"""
Basic sanity cuts. 
"""

# rename toi columns to match confirmed columns using a custom utility function
columns.rename_columns(tois, columns.toi2comp_map)
tois['tran_flag'] = 1
tois['discoverymethod'] = 'Transit'

# Ensure all columns have masks
for cat in [confirmed, tois]:
    catalog_utilities.add_masks(cat)

# all planets must have known periods
for cat in [confirmed, tois]:
    catutils.filter_bad_values(cat, 'pl_orbper', fill_value=-1, filter_func=lambda x: x <= 0)

# keep only the transiting planets from confirmed
transit_mask = confirmed['tran_flag'].filled(0) == 1
confirmed = confirmed[transit_mask]

# keep only planets not known to be false positives from the toi table
candidate_mask = (tois['tfopwg_disp'] != 'FA') & (tois['tfopwg_disp'] != 'FP')
tois = tois[candidate_mask]

# some error and lim columns in the toi table have only masked values. delete these.
removed_toi_columns = [] # just so we can look at what was removed, if desired
for name in tois.colnames:
    col = tois[name]
    if sum(col.mask) == len(col):
        tois.remove_column(name)
        removed_toi_columns.append(name)


#%% Unit and dtype housekeeping
"""
Fix units and dtypes.
"""

# fix dtypes
columns.fix_dtypes(confirmed, columns.dtype_map)
columns.fix_dtypes(tois, columns.dtype_map)

# fix units
columns.fix_units(confirmed, columns.units_map)
columns.fix_units(tois, columns.units_map)

# some by hand unit fixes
confirmed['pl_orbsmax'].unit = u.au

# stellar logg for tois should be a dex unit
def fix_logg_unit(col):
    col.unit = u.dex(u.cm/u.s**2)
    return col
columns.operate_on_suffixes(tois, 'st_logg', fix_logg_unit)

def ppm_to_percent(col):
    newcol = col/1e6*100
    newcol.unit = '%'
    return newcol
columns.operate_on_suffixes(tois, 'pl_trandep', ppm_to_percent)

def fix_inaccurate_trandur_unit(col):
    col.unit = u.Unit('h')
    newcol = col.to('h')
    return newcol
columns.operate_on_suffixes(confirmed, 'pl_trandur', fix_inaccurate_trandur_unit)


#%% Save - confirmed & tois
"""
Save tables here. 
"""

confirmed.write(paths.selection_intermediates / 'confirmed_planets.ecsv', overwrite=True)
tois.write(paths.selection_intermediates / 'tois.ecsv', overwrite=True)


#%% Load - confirmed & tois
"""
Load tables if restarting from this position.
"""

confirmed = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'confirmed_planets.ecsv')
tois = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'tois.ecsv')


#%% define period match tolerance
"""
Set relative tolerance for matching by orbital period for use in the next two cells. 
"""
period_relative_tolerance = 0.1


#%% tic id match
"""
Identify matches in the confirmed table based on TIC ID and orbital period. 

I wish we could use the astropy.table.join function to merge tables
but I have been unable to find a way to use it that enables me to prioritize the comnfirmed values
adding in values for extra columns from the TOI table for the matches as well as adding TOI rows that do not match
so I am doing this by brute force more or less
"""

# make tic id cols consistent
if hasattr(confirmed['tic_id'][0], '__len__'): # checks if confirmed col is a string so this only runs if it is (helpful if rerunning cell)
    old_tic_id = confirmed['tic_id']
    confirmed['tic_id'] = columns.empty_column_like(tois['tic_id'], len(confirmed))
    for i_toi, val in enumerate(old_tic_id):
        if not catutils.isnull(val, ''):
            confirmed['tic_id'][i_toi] = int(val.replace('TIC ', ''))

assert not any(tois['tic_id'].mask)

i_toi_matches_tic_id_only = catutils.match_by_tic_id(tois, confirmed)
i_toi_matches_tic_id = [[] for _ in range(len(confirmed))]
Pconfirmed = confirmed['pl_orbper'].filled(np.nan)
Ptois = tois['pl_orbper'].filled(np.nan)
for i_confirmed, i_matches in enumerate(i_toi_matches_tic_id_only):
    if len(i_matches) > 0: # at least one match
        # note! even if there is only one match, that doesn't mean it is the right planet
        # there are some systems with both a confirmed planet and a candidate
        # so we must match the period also to be sure that we aren't selecting the candidate
        period_match = np.isclose(Pconfirmed[i_confirmed], Ptois[i_matches], period_relative_tolerance)
        m = sum(period_match)
        if m == 0:
            # this might be a new planet in the system
            continue
        elif m == 1:
            i_tic_match = np.asarray(i_matches)
            i_match, = i_tic_match[period_match]
        else:
            raise ValueError
        i_toi_matches_tic_id[i_confirmed].append(i_match)

n_matches_tic_id = np.asarray([len(x) for x in i_toi_matches_tic_id])
# verify no multiple matches
assert not np.any(n_matches_tic_id > 1)
N_matches_tic_id = sum(n_matches_tic_id)
print(f'{N_matches_tic_id} tois matched to confirmed planets based on TIC ID and period.')


#%% position match
"""
Even after the tic_id matching, there remain ~20 tois that are flagged as known planets. 
Presumaby these should all match to planets in the exoplanets catalog.
Let's try matching by position next to see if it captures these. 
(I still want to keep the initial tic_id matching because I think it might be more accurate.)

Note that this adds to the existing i_toi_matches map. 
"""
# cross match by position first
# I checked and even for the target that matches based on tic id with the largest proper motion, the coordinates
# between the toi and confirmed catalogs are the same, so I believe they are specified at the same epoch
# (except that the confirmed catalog uses one more sig fig)
confirmed_coords = coord.SkyCoord(confirmed['ra'], confirmed['dec'])
toi_coords = coord.SkyCoord(tois['ra'], tois['dec'])
# allow for a generous position error to accommodate any epoch mismatch between the catalogs
# that might result in offsets to target proper motion
position_matches = toi_coords.search_around_sky(confirmed_coords, 0.01*u.deg)

# now check for period matches and see if things work out against the tic_id match
i_toi_matches_position = [[] for _ in range(len(confirmed))]
for i_confirmed, i_toi, d, _ in zip(*position_matches):
    Pm = confirmed['pl_orbper'][i_confirmed]
    Pt = tois['pl_orbper'][i_toi]
    if catutils.isnull(Pm, np.nan) or catutils.isnull(Pt, np.nan):
        continue
    if np.isclose(Pm, Pt, period_relative_tolerance):
        i_toi_matches_position[i_confirmed].append(i_toi)

n_matches_position = np.asarray([len(x) for x in i_toi_matches_position])
# verify no multiple matches
assert not np.any(n_matches_position > 1)
N_matches_position = sum(n_matches_position)
print(f'{N_matches_position} tois matched to confirmed planets based on position and period.')


#%% compare
"""
compare the two cross matches
"""
xmatch_compare = [x == y for x,y in zip(i_toi_matches_tic_id, i_toi_matches_position)]
if not all(xmatch_compare):
    raise NotImplementedError('The TIC ID and position cross matches do not agree. Revisions required.')

i_toi_matches = i_toi_matches_tic_id


#%% extra KPs
"""
There remain ~20 planets listed as known planets in the TOIs that don't match to a confirmed planet. Inspect.

So far as I can tell, these are not actually known planets. 
We should look into these more if any make it into our final sample. 

As of 2025-01-15 build, none of these appear in the final table, so I have not made any effort to address them.
There are only two that match to a system in the confirmed table based on tic_id or period (and they match using both):
Kepler-37 (164652245) and K2-79 (435339558) and in both cases the TOI catalog period is twice the period of one of
the planets in the confirmed system. 

"""
print_cols = 'ra dec tic_id toi pl_orbper pl_rade'.split()
unmatched_mask = np.ones(len(tois), bool)
i_matched = sum(i_toi_matches, [])
unmatched_mask[i_matched] = False
KP_mask = tois['tfopwg_disp'] == 'KP'
unmatched_KP_mask = unmatched_mask & KP_mask
tois[unmatched_KP_mask][print_cols].pprint(-1, -1)

# adding a column to track these
tois['warning1'] = table.MaskedColumn(unmatched_KP_mask)
tois['warning1'].description = ("Planets from the TOI catalog that were listed as known planets but that don't appear"
                                "in the confirmed planets catalog.")


#%% merge

"""
Merge confirmed planets and TOIs. 

This involves a lot of legwork to deal with column inconsistencies and so on. 
"""
# create a new table for the merge, starting with the confirmed table
cat = confirmed.copy()

Nboth = len(confirmed) + len(tois)
print(f'Initial number of targets in both the confirmed and toi tables: {Nboth}')

# add extra columns from the TOI table not already in the merge
columns.add_missing_columns(cat, tois)

# helpful if you need to examine differences in the columns available in each table
# especially for making data types consistent
# meta = columns.compare_columns(confirmed, tois)

# ensure all columns have a mask attribute for consistency
for name in cat.colnames:
    if not hasattr(cat[name], 'mask'):
        cat[name] = table.MaskedColumn(cat[name])

# columns to merge in from the tois table for matches
toi_merge_cols = 'toi toipfx ctoi_alias tfopwg_disp toi_created rowupdate'.split()

# transfer info from matched tois into the merged table
for i_cat, i_matches in enumerate(i_toi_matches_tic_id):
    if len(i_matches) > 0:
        i_toi, = i_matches
        for name in toi_merge_cols:
            cat[name][i_cat] = tois[name][i_toi]

# remove the matched tois from the tois table
i_remove = sum(i_toi_matches_tic_id, [])
unique_tois = tois.copy()
unique_tois.remove_rows(i_remove)

# add a hostname column to the toi table (used much later)
# be sure to use the hostname in the merged table if there is one to avoid confusion later
catutils.add_filled_masked_column(unique_tois, 'hostname','', dtype='object', mask=True)
in_cat = np.in1d(unique_tois['tic_id'], cat['tic_id'])
tic_ids_in_cat = unique_tois['tic_id'][in_cat]
tic_ids_in_cat = np.unique(tic_ids_in_cat.data)
catutils.set_index(cat, 'tic_id')
catutils.set_index(unique_tois, 'tic_id')
for tic_id_ in tic_ids_in_cat:
    # used a loop because these hosts appear different numbers of times in the two tables
    # but maybe there is a faster way?
    n_id_matches = np.sum(cat['tic_id'] == tic_id_)
    cat_hostnames = cat.loc[tic_id_]['hostname']
    if n_id_matches > 1:
        cat_hostname, = set(cat_hostnames)
    else:
        cat_hostname = cat_hostnames
    unique_tois.loc[tic_id_]['hostname'] = cat_hostname
toi_names = np.char.add('TOI-', unique_tois['toipfx'].astype(str))
unique_tois['hostname'][~in_cat] = toi_names[~in_cat]
assert not np.any(unique_tois['hostname'].mask)

# stack the tables
# when you do this, you will get a some warnings about units and descriptions not matching.
# this led to me correcting all* units after pulling the tables.
# there are still some weird units I kept like %. and I made a new unit for Earth insolation, Searth.
cat = table.vstack((cat, unique_tois), join_type='outer')

print(f'Final number of targets in the merged table {len(cat)}')

# make sure every column still has a mask
assert all(hasattr(col, 'mask') for col in cat.itercols())

print('\nMake sure the merge warnings from the table stack operation are not dangerous.')

#%% define unique planet ids

"""
Create a unique ID for each planet and index the table by that.
"""

cat['id'] = cat['pl_name']
noname = cat['pl_name'].mask
cat['id'][noname] = cat['toi'][noname]

assert not any(cat['id'].mask)
assert len(np.unique(cat['id'])) == len(cat)

#%% community targets simbad
"""
Check community target lists and add targets that are not present. Start by getting their parameters from SIMBAD.
"""

# some shenanigans to make just a single query to simbad or else it might lock us out as a bot
alltargets = []
def concatenate(targets):
    alltargets.extend(targets.tolist())
_ = catutils.requested_target_lists_loop(concatenate)
extra_cols = 'ids distance rv_value parallax'.split()
simbad = query.get_simbad_from_names(alltargets, extra_cols=extra_cols)
# it seems like you should be able to add "measurements" to the query, but this returns an error
# maybe try again in the future bc this could be a good way to get Teff
assert len(simbad) == len(alltargets)


#%% merge community targets
"""
Now match with targets in the merged catalog.

List those that didn't match and were appended to the catalog and any critical params that should be entered by 
hand for them. 
"""

catutils.add_filled_masked_column(cat,'requested_target', False, dtype=bool)
catutils.add_filled_masked_column(cat,'manually_added', False, dtype=bool)
catutils.scrub_indices(cat) # scrub any residual indices or else add_row() will raise an error
i_simbad = 0
def check_and_mark(targets):
    global i_simbad  # simbad shenanigans to track location in the all targets simbad results
    flag_col_name = f'requested_{targets.name}'
    catutils.add_filled_masked_column(cat, flag_col_name, 0, dtype=int) # column to identify targets (and their order) present in given list
    tic_ids = []
    for ids in simbad['IDS']:
        # some targets have multiple tic ids in simbad, so gotta be careful
        result = re.findall(r'\bTIC (\d+)\|', ids)
        host_tic_ids = np.array(list(map(int, result)))
        mask = np.isin(host_tic_ids, cat['tic_id'])
        if sum(mask) == 0:
            tic_ids.append(host_tic_ids[0])
        else:
            tic_ids.append(host_tic_ids[mask][0])
    simbad['tic_id'] = tic_ids
    simbad_slice = slice(i_simbad, i_simbad + len(targets))
    i_match_in_cat = catutils.match_by_tic_id(cat, simbad[simbad_slice])
    for i, i_matches in enumerate(i_match_in_cat):
        cat[flag_col_name][i_matches] = i + 1
        cat['requested_target'][i_matches] = True
        if len(i_matches) == 0:
            catutils.add_masked_row(cat)
            added_cols = 'id hostname ra dec tic_id st_radv sy_dist sy_plx'.split()
            cat['manually_added'][-1] = True
            cat['requested_target'][-1] = True
            cat[flag_col_name][-1] = i + 1
            cat['id'][-1] = targets[i] + ' b'
            cat['hostname'][-1] = targets[i]
            cat['ra'][-1] = simbad['ra'][i_simbad]
            cat['dec'][-1] = simbad['dec'][i_simbad]
            cat['tic_id'][-1] = simbad['tic_id'][i_simbad]
            cat['st_radv'][-1] = simbad['RV_VALUE'][i_simbad]
            cat['sy_dist'][-1] = simbad['Distance_distance'][i_simbad]
            cat['sy_plx'][-1] = simbad['PLX_VALUE'][i_simbad]
            needed_cols = set(columns.essential) - set(added_cols)
            print(f'{targets.name} request {targets[i]} not matched. '
                  f'Added {added_cols} from Simbad. '
                  f'YOU MUST manually add {needed_cols} in the next code cell.')
        i_simbad += 1 # simbad tracking, very tricksy sorry

_ = catutils.requested_target_lists_loop(check_and_mark)


#%% clean duplicates

# TODO at some point we might start to see this program's observations show up in the duplication checking table
# need to modify things so they are excluded in the search

# I'm mainly concerned about some planets showing up in both the confirmed and TOI tables,
# though this didnt't seem to have happened when I first ran this.

# using a bit of a hack by giving the periods as a distance and then doing a 3D position match based on that
distlimit = 0.2*u.pc
merge_positions, duplicated = catutils.find_duplicates(cat, distlimit)

# select just one planet to keep
viewcols = 'pl_name toi ra dec pl_orbper sy_dist'.split()
i_dups, = np.nonzero(duplicated)
i_dups = list(i_dups)
to_discard = []
while len(i_dups) > 0:
    i_dup = i_dups[0]
    match_mask = merge_positions.separation_3d(merge_positions[i_dup]) < distlimit
    i_matches, = np.nonzero(match_mask)
    i_discard = list(i_matches)
    # keep the first planet in the confirmed table, if one is available
    has_name = ~cat[match_mask]['pl_name'].mask
    if np.any(has_name):
        i_keep = i_matches[has_name][0]
    else:
        has_dist = ~cat[match_mask]['sy_dist'].mask
        if np.any(has_dist):
            i_keep = i_matches[has_dist][0]
        else:
            i_keep = i_matches[0]
    i_discard.remove(i_keep)
    to_discard.extend(i_discard)
    cat[match_mask][viewcols].pprint(-1,-1)
    kept_id = cat[i_keep]['id']
    print(f'\tKeeping {kept_id}')
    print('')
    [i_dups.remove(_) for _ in i_matches]
cat.remove_rows(to_discard)

# verify there are no more duplicates
_, still_duplicated = catutils.find_duplicates(cat, distlimit)
assert not np.any(still_duplicated)

#%% new target parameters

"""
HUMAN HAND NEEDED. Manually add key parameters for targets that are not present and other paremeters useful later.

Make sure that all these planets transit, on the off chance someone suggested a planet that does not.
"""

catutils.set_index(cat, 'id')

cat.loc['HD 60779 b']['st_teff'] = 5860 # not sure where I got this. vizier, maybe
cat.loc['HD 60779 b']['pl_orbper'] = 30 # FIXME made up
cat.loc['HD 60779 b']['pl_rade'] = 3 # FIXME made up

# check that all the manually added targets have their key values present
manual_mask = cat['manually_added'].filled(False)
for row in cat[manual_mask]:
    value_masked = [catutils.isnull(row[name], np.nan) for name in columns.essential]
    assert not any(value_masked)


# activity indicators for targets that would otherwise require buffer dumps or violate brightness limits
catutils.set_index(cat, 'hostname')
cat.loc['TOI-4364']['st_rotp'] = 10.57
cat.loc['HD 60779']['st_age'] = 7.6
cat.loc['HD 60779']['st_ageerr1'] = 10.8 - 7.6
cat.loc['HD 60779']['st_ageerr2'] = 7.6 - 4.6
cat.loc['TOI-2443']['st_age'] = 10**0.6 # Yee+ 2017 https://iopscience.iop.org/article/10.3847/1538-4357/836/1/77
cat.loc['TOI-2443']['st_ageerr1'] = 10**(0.6+0.52) - 10**0.6
cat.loc['TOI-2443']['st_ageerr1'] = 10**0.6 - 10**(0.6-0.52)
cat.loc['TOI-5789']['st_age'] = 5. # based on vsin < 1 km s-1, ExoFOP
cat.loc['TOI-1231']['st_age'] = 5. # based on lack of activity, Burt+ 2016



#%% flag distance and period cuts

"""
Add a column to track what happens to targets and make a distance cut. 
"""
catutils.add_filled_masked_column(cat, 'stage1', True, dtype=bool)
cat['decision'] = table.MaskedColumn(length=len(cat), mask=True, dtype='O')

# distant targets
# assume if they have no catalogged distance they probably are bad targets
# TODO see what happens if I just don't make this cut and rely on Lya flux estimates later
max_dist = 200
distance_cut = cat['sy_dist'].filled(2*max_dist) > max_dist
dist_cut_str = f'Removed due to missing distance or distance > {max_dist} pc.'
catutils.flag_cut(cat, distance_cut, dist_cut_str)

# long period targets
max_period = 90
period_cut = cat['stage1'].filled(True) & (cat['pl_orbper'].filled(2*max_period) > max_period)
period_cut_str = f'Removed due to missing orbital period or period > {max_period} d.'
catutils.flag_cut(cat, period_cut, period_cut_str)


#%% checkpoint
"""This may take several minutes."""

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt1__merged-confirmed-tois-community.ecsv', overwrite=True)
    merged = cat.copy() # used later to check that various targets were considered

#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt1__merged-confirmed-tois-community.ecsv')


#%% PRUNE

keepers = ('requested_target',)
cat = catutils.make_a_cut(cat, 'stage1', keepers=keepers)


#%% target removal test

"""This allows for a test where a number of promising targets are removed. The idea is to
 test for bugs that might cause values to be computed and then assigned to the wrong
targets. If so, the removal of some targets could cause the ordering of others in the list
of selected targets to change. If the order does not change, the test is passed.

The test isn't automatic -- you have to look at the before and after target selections
 to see if it was passed."""

if toggle_target_removal_test:
    names_to_remove = 'TOI-454,TOI-2540,LP 791-18,TOI-2443,TOI-2422,HD 207897,TOI-1710,HD 17156,HR 858,TOI-1764'.split(',')
    catutils.set_index(cat, 'hostname')
    i = cat.loc_indices[names_to_remove]
    cat.remove_rows(i)


#%% SIMBAD ID match
"""
Successively try all the different names in the catalog to try to get matches. 
"""
_ = catutils.scrub_indices(cat) # these are slowing down table operations. I'll index again later

with warnings.catch_warnings():
    warnings.simplefilter('ignore', query.BlankResponseWarning)
    extra_cols = ['rv_value', 'flux(J)']
    simbad = query.get_simbad_from_tic_id(cat['tic_id'], extra_cols=extra_cols)
    assert len(simbad) == len(cat)
    print(f"{np.sum(simbad['simbad_match'])}/{len(cat)} targets matched by TIC ID. "
          f"{np.sum(~simbad['simbad_match'])} unmatched.")

    other_name_cols = ['gaia_id', 'hostname', 'hd_name', 'hip_name']
    for colname in other_name_cols:
        names = cat[colname]
        query_mask = ~simbad['simbad_match'] & ~names.mask
        i_query, = np.nonzero(query_mask)
        query_names = names[query_mask].filled('?').tolist()
        new_simbad = query.get_simbad_from_names(names=query_names, extra_cols=extra_cols)
        print(f"{np.sum(new_simbad['simbad_match'])} additional targets matched by {colname}.")
        simbad[i_query] = new_simbad

    print(f"{np.sum(simbad['simbad_match'])}/{len(cat)} targets matched by one of the cataloged identifiers. "
          f"{np.sum(~simbad['simbad_match'])} unmatched.")

# keep only the columns we want to transfer later
keepcols = ['MAIN_ID', 'RV_VALUE', 'FLUX_J', 'simbad_match']
simbad = simbad[keepcols]


#%% SIMBAD 3D position match
"""
Now try a positional match on remaining targets. This uses pyvo and so the columns end up being different, annoyingly.

Note that using the XMatch service for this doesn't work well because it returns planets as well as stars and does 
not give radial velocity or fluxes for the planets. 
"""
extra_cols = ['main_id', 'rvz_radvel'] # unfortunately I haven't been able to figure out how to get J fluxes
query_mask = ~simbad['simbad_match']
i_query, = np.nonzero(query_mask)
simbad_3d, dist = query.get_simbad_3d(cat[query_mask], search_radius=20 * u.arcsec, cols=extra_cols)

# If this query times out, consider using an asynchronous query
# https://pyvo.readthedocs.io/en/latest/dal/index.html#synchronous-vs-asynchronous-query

print(f'Maximum distance between a target and its match is {dist.max()}')
dist_frac = dist / cat['sy_dist'][query_mask]
if toggle_plots:
    plt.figure()
    _ = plt.hist(dist_frac.to_value(''), 200)
    # plt.xlabel('3D object distances from the SIMBAD cross match (pc)')
    plt.xlabel('fractional 3D distance offset in SIMBAD cross match')

# impose a cut
match_cut_3d = 0.01
good_match = dist_frac < match_cut_3d
print(f"{np.sum(good_match)} additional targets matched based on 3D positions with {match_cut_3d*100}% tolerance.")
i_transfer = i_query[good_match]
simbad['MAIN_ID'][i_transfer] = simbad_3d['main_id'][good_match]
simbad['RV_VALUE'][i_transfer] = simbad_3d['rvz_radvel'][good_match]
simbad['simbad_match'][i_transfer] = True

print(f"{np.sum(simbad['simbad_match'])}/{len(cat)} targets matched by identifier or 3D position. "
      f"{np.sum(~simbad['simbad_match'])} unmatched.")


#%% transfer simbad radv, id
"""
We will want ot use the simbad id for future queries. 
"""

transfer_rv = cat['st_radv'].mask & ~simbad['RV_VALUE'].mask & np.isfinite(simbad['RV_VALUE'].filled(0))
cat['st_radv'][transfer_rv] = simbad['RV_VALUE'][transfer_rv]

transfer_jmag = cat['sy_jmag'].mask & np.isfinite(simbad['FLUX_J'].filled(nan))
cat['sy_jmag'][transfer_jmag] = simbad['FLUX_J'][transfer_jmag]

cat['simbad_match'] = simbad['simbad_match']
cat['simbad_id'] = simbad['MAIN_ID']

assert np.all(np.isfinite(cat['st_radv'].filled(0)))


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt2__cut-dist-period__add-simbad-rvs-names.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt2__cut-dist-period__add-simbad-rvs-names.ecsv')


#%% filter HST observation table (only run if new catalog downloaded)

"""get the latest catalog of all existing and planned observations at https://archive.stsci.edu/pub/catalogs/, paec-7-present.cat
be careful not to download the _ss_ catalog as I think it is solar system objects"""

if toggle_remake_filtered_hst_archive:
    hst_observations_path = paths.hst_observations / 'paec_7-present.cat'
    hst_filtered = dc.filter_hst_observations(hst_observations_path)
    hst_filtered.write(paths.hst_observations / 'hst_observations_filtered.ecsv', overwrite=True)


#%% load observation tables

hst_filtered = table.Table.read(paths.hst_observations / 'hst_observations_filtered.ecsv')
our_observations = hst_filtered['prop'] == 17804
hst_filtered = hst_filtered[~our_observations]
verified = table.Table.read(paths.checked / 'verified_external_observations.csv')


#%% mark observations

"""we should keep these in the table so that we can go back later and hand check targets that are good candidates
to be sure that the lya and FUV observations did not fail"""

dc.flag_duplicates(cat, hst_filtered)
dc.merge_verified(cat, verified)


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt3__add-archival_obs_counts.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt3__add-archival_obs_counts.ecsv')


#%% fill key missing params

"""
Radii needed to compute transit depth.
Mass needed to compute planetary hill sphere.
Teff needed to estimate intrinsic Lya.
Lbol needed to catch overly high Lya estimates.

orbsmax needed to compute hill sphere.
mass neeeded to compute hill sphere.
eqt needed for later cuts.
"""

def hist_compare(key, x, title, bins=np.linspace(0,2,100)):
    y = cat[key].filled(nan)
    plt.figure()
    plt.title(title)
    _ = plt.hist(x/y, bins)
    plt.xlabel('esimated/cataloged values')

### stellar params ###

# make sure any unphysical or nan values are masked in key columns
key_columns = 'st_rad st_mass st_teff'.split()
for name in key_columns:
    catutils.mask_unphysical(cat, name, must_be_positive=True)

n_bad_st_rad = np.sum(cat['st_rad'].mask)
n_bad_st_mass = np.sum(cat['st_mass'].mask)
n_bad_st_teff = np.sum(cat['st_teff'].mask)
n_bad_st_lum = np.sum(cat['st_lum'].mask)

# R from Teff
R = catutils.safe_interp_table(cat['st_teff'].filled(0), 'Teff', 'R_Rsun', ref.mamajek) * u.Rsun
transfer = (R > 0) & cat['st_rad'].mask
print(f'{np.sum(transfer)} of {n_bad_st_rad} bad stellar radius values filled based on Teff.')
if toggle_plots:
    hist_compare('st_rad', R, 'Rstar from Teff')
cat['st_rad'][transfer] = R[transfer]

# M from Teff
M = catutils.safe_interp_table(cat['st_teff'].filled(0), 'Teff', 'Msun', ref.mamajek) * u.Msun
transfer = (M > 0) & cat['st_mass'].mask
print(f'{np.sum(transfer)} of {n_bad_st_mass} bad stellar mass values filled based on Teff.')
if toggle_plots:
    hist_compare('st_mass', M, 'Mstar from Teff')
cat['st_mass'][transfer] = M[transfer]

# M from R
M = catutils.safe_interp_table(cat['st_rad'].filled(0), 'R_Rsun', 'Msun', ref.mamajek) * u.Msun
transfer = (M > 0) & cat['st_mass'].mask
print(f'{np.sum(transfer)} of {n_bad_st_mass} bad stellar mass values filled based on stellar radius.')
if toggle_plots:
    hist_compare('st_mass', M, 'Mstar from R')
cat['st_mass'][transfer] = M[transfer]

# Teff from R
T = catutils.safe_interp_table(cat['st_rad'].filled(0), 'R_Rsun', 'Teff', ref.mamajek) * u.K
transfer = (T > 0) & cat['st_teff'].mask
print(f'{np.sum(transfer)} of {n_bad_st_teff} bad stellar teff values filled based on radius.')
if toggle_plots:
    hist_compare('st_teff', T, 'Mstar from Teff')
cat['st_teff'][transfer] = T[transfer]

# Lbol from Teff
logL = catutils.safe_interp_table(cat['st_teff'].filled(0), 'Teff', 'logL', ref.mamajek)
transfer = np.isfinite(logL) & cat['st_lum'].mask
print(f'{np.sum(transfer)} of {n_bad_st_lum} bad stellar luminosity values filled based on radius.')
if toggle_plots:
    hist_compare('st_lum', logL, 'logL from Teff')
cat['st_lum'][transfer] = logL[transfer]


### planet params ###


# make sure any unphysical or nan values are masked in key columns
key_columns = 'st_teff st_rad st_mass pl_eqt pl_orbsmax pl_orbper'.split()
for name in key_columns:
    catutils.mask_unphysical(cat, name, must_be_positive=True)
catutils.mask_unphysical(cat, 'st_lum', must_be_positive=False)

isgood = lambda colname: catutils.is_positive_real(cat, colname)
isbad = lambda colname: ~isgood(colname)
get_nanfilled = lambda colname: cat[colname].filled(np.nan).quantity

n_bad_orbsmax = np.sum(isbad('pl_orbsmax'))
n_bad_pl_mass = np.sum(isbad('pl_bmasse'))
n_bad_eqt = np.sum(isbad('pl_eqt'))

# a using period and stellar mass
transfer = isgood('st_mass') & isgood('pl_orbper') & isbad('pl_orbsmax')
P, M = map(get_nanfilled, ('pl_orbper', 'st_mass'))
G = const.G
a = (G*M/4/pi**2 * P**2)**(1/3)
a = a.to('AU')
print(f'{np.sum(transfer)} of {n_bad_orbsmax} bad planet orbsmax values filled based on orbital period.')
assert np.all(a[transfer] > 0)
if toggle_plots:
    hist_compare('pl_orbsmax', a, 'a from period', bins=100)
cat['pl_orbsmax'][transfer] = a[transfer]

# a using stellar Teff, rad, and planet Teq
transfer = isgood('st_teff') & isgood('st_rad') & isgood('pl_eqt') & isbad('pl_orbsmax')
# albedo=0.3 is a typical assumption, and based on a comparison of estimates of a made using period vs Teq,
# this seems to be the assumption of the TOI catalog
albedo = 0.3
Tstar, Tplanet, Rstar = map(get_nanfilled, ('st_teff', 'pl_eqt', 'st_rad'))
a = (Tstar / Tplanet) ** 2 * Rstar / 2 * np.sqrt(1 - albedo)
a = a.to('AU')
assert np.all(a[transfer] > 0)
print(f'{np.sum(transfer)} of {n_bad_orbsmax} bad planet orbsmax values filled based on equilibrium temperature.')
if toggle_plots:
    hist_compare('pl_orbsmax', a, 'a from Teq')
cat['pl_orbsmax'][transfer] = a[transfer]

# Teq using Lstar and a
goodLum = np.isfinite(cat['st_lum'].filled(np.nan))
transfer = goodLum & isgood('pl_orbsmax') & isbad('pl_eqt')
logL, a = map(get_nanfilled, ('st_lum', 'pl_orbsmax'))
L = logL.physical
sb = const.sigma_sb
Teq = (L / 16 / pi / sb / a ** 2) ** (1 / 4)
Teq = Teq.to('K')
assert np.all(Teq[transfer] > 0)
print(f'{np.sum(transfer)} of {n_bad_orbsmax} bad planet Teq values filled based on smax and luminosity.')
if toggle_plots:
    hist_compare('pl_eqt', Teq, 'Teq from Lstar and a')
cat['pl_eqt'][transfer] = Teq[transfer]

# Teq using Teff and Rstar and a
transfer = isgood('st_teff') & isgood('st_rad') & isgood('pl_orbsmax') & isbad('pl_eqt')
Teff, Rstar, a = map(get_nanfilled, ('st_teff', 'st_rad', 'pl_orbsmax'))
Teq = np.sqrt(Rstar / 2 / a) * Teff
Teq = Teq.to('K')
assert np.all(Teq[transfer] > 0)
print(f'{np.sum(transfer)} of {n_bad_orbsmax} bad planet Teq values filled based on smax, stellar radius, and Teff.')
if toggle_plots:
    hist_compare('pl_eqt', Teq, 'Teq from Teff, Rstar, and a')
cat['pl_eqt'][transfer] = Teq[transfer]

## Mplanet from Rplanet
# note that whenever there is a massj there is a masse -- I checked
n_bad_mass = np.sum(~isgood('pl_bmasse'))
transfer = isgood('pl_rade') & ~isgood('pl_bmasse')
Rp = get_nanfilled('pl_rade')
Rp = Rp.to_value('Rearth')
small = Rp < 1.23
medium = ~small & (Rp < 14.26)
large = Rp >= 14.26
M = np.zeros_like(Rp)
M[small] = 0.9718 * Rp[small] ** 3.58
M[medium] = 1.436 * Rp[medium] ** 1.7
M[large] = 150 # after the breakpoint, increases in mass hardly change radius. we'll assume a lowish mass.
assert np.all(M[transfer] > 0)
print(f'{np.sum(transfer)} of {n_bad_mass} bad planet mass values filled based on radius in Chen and Kipping 2017 relationships.'
      f'\nBEWARE for gas giants a mass of 150 Mearth was assumed. After stage 1, only measured masses should be used.')
if toggle_plots:
    hist_compare('pl_bmasse', M, 'Mp from Rp')
    med_compare = M[medium]/cat['pl_bmasse'][medium].filled(nan)
    _ = plt.hist(med_compare, np.linspace(0.7,1.3,100))
cat['pl_bmasse'][transfer] = M[transfer]

# J values from Teff
transfer = isgood('st_teff') & isgood('sy_dist') & cat['sy_jmag'].mask
Mj = catutils.safe_interp_table(cat['st_teff'], 'Teff', 'M_J', ref.mamajek)
d = cat['sy_dist'].filled(nan)
J = Mj + 5*np.log10(d) - 5
if toggle_plots:
    hist_compare('sy_jmag', J, 'J mag from Teff')
cat['sy_jmag'][transfer] = J[transfer]


#%% flag missing or suspect params to cut

pass
# stellar params
print(f"{np.sum(cat['st_rad'].mask)} stellar radius values remain missing.")
print(f"{np.sum(cat['st_mass'].mask)} stellar mass values remain missing.")
print(f"{np.sum(cat['st_teff'].mask)} stellar Teff values remain missing.")

underconstrained_star = cat['st_rad'].mask & cat['st_mass'].mask & cat['st_teff'].mask
underconst_star_str = ('Cut because the star does not have at least one of '
                       'radius, mass, or Teff in the catalog.')
catutils.flag_cut(cat, underconstrained_star, underconst_star_str)

# missing planetary params
no_smax = cat['pl_orbsmax'].mask
no_eqt = cat['pl_eqt'].mask
no_mass = cat['pl_bmasse'].mask
print(f"{np.sum(no_smax)} planetary smax values remain missing.")
print(f"{np.sum(no_eqt)} planet Teq values remain missing.")
print(f"{np.sum(no_mass)} planet mass values remain missing.")

smax_cut_str = ('Cut because planet has no cataloged orbital semi-major axis and '
                'one could not be inferred from other parameters.')
catutils.flag_cut(cat, no_smax, smax_cut_str)

mass_cut_str = ('Cut because planet has no cataloged mass and one could not be '
                'inferred from other parameters')
catutils.flag_cut(cat, no_mass, mass_cut_str)

# suspect planetary params
suspect_b = cat['pl_imppar'].filled(0) >= 1
suspect_b_str = f'Cut because impact parameter >= 1'
catutils.flag_cut(cat, suspect_b, suspect_b_str)


#%% flag hot & evolved stars to cut
""""""
# evolved stars
min_logg = 3.5
max_logg = 5.5
logg = cat['st_logg'].filled(4)
spTs = cat['st_spectype'].filled('V').astype('str')
off_MS = ((logg > max_logg) | (logg < min_logg)
          | (np.char.count(spTs, 'I') > 0))
off_MS_str = (f'Cut because star assumed not on main sequence due to logg outside of [{min_logg}, {max_logg}] range or'
              f'SpT had an "I" in it.')
catutils.flag_cut(cat, off_MS, off_MS_str)

# hot stars
max_Teff = 6500
hotstar = cat['st_teff'].filled(5000) > max_Teff
hotstar_str = f'Cut because star assumed too hot to have Lya emission due to Teff > {max_Teff} K.'
catutils.flag_cut(cat, hotstar, hotstar_str)


#%% flag no H/He atmosphere for cutting

Rgap = empirical.ho_gap_lowlim(cat['pl_orbper'], cat['st_mass'])
has_HHe = cat['pl_rade'] > Rgap
cat['flag_gaseous'] = has_HHe

mask_no_HHe = ~has_HHe
remove_no_HHe = ("Removed because planet candidate not likely to have an H/He atm "
                 "(size known to be below the radius gap).")
catutils.flag_cut(cat, mask_no_HHe.filled(False), remove_no_HHe)


#%% flag young systems
young = (cat['st_age'].filled(5) < 1) & (cat['st_agelim'].filled(0) != -1)
# note that lim == 1 corresponds to < the value in the table, -1 is >
cat['flag_young'] = table.MaskedColumn(young)


#%% flag multis to be sure they're kept

# match by position to include candidates in the count
positions1 = coord.SkyCoord(cat['ra'], cat['dec'])
i_multis1, _, _, _ = positions1.search_around_sky(positions1, seplimit=0.000001 * u.deg)
_, multi_count1 = np.unique(i_multis1, return_counts=True)
flag_multi = multi_count1 > 1
cat['flag_multiplanet'] = table.MaskedColumn(flag_multi, fill_value=False,
                                               description='multiple confirmed or candidate planets in system')


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt4__fill-basic_properties.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt4__fill-basic_properties.ecsv')


#%% PRUNE

cat = catutils.make_a_cut(cat, 'stage1', keepers=('requested_target', 'flag_multiplanet', 'external_lya'))


#%% load and match existing galex mags

"""
These are helpful for improved Lya flux estimates and estimating buffer times for G140L spectra. 

This is slowww because handling proper motions requires a target-by-target query. Expect 1 h per 1000 targets.
Results saved will be reloaded and matched in in order to save time. 
"""

# try loading and cross-matching galex first
filepath = paths.selection_intermediates / 'galex_magntiudes.ecsv'
if filepath.exists():
    galex_tbl = catutils.load_and_mask_ecsv(filepath)
    cat = table.join(cat, galex_tbl, 'id', join_type='left')


#%% new/missing galex

pass
# create empty columns for galex mags if needed
for band in ['fuv', 'nuv']:
    basename = f'sy_{band}mag'
    if basename not in cat.colnames:
        catutils.add_masked_columns(cat, basename, 'float', suffixes=('', 'err1', 'err2', 'lim'))
    obsname = basename + 'obs'
    if obsname not in cat.colnames:
        cat[obsname] = table.MaskedColumn(length=len(cat), mask=True, dtype=bool)

def add_mag(cps, cpserr, band, i):
    if cps == -999: # either lower limit or no data
        if cpserr == -999: # no data
            cat[f'sy_{band}magobs'][i] = False
        else: # lower limit, limit cps given in cpserr
            result = galex_query.galex_cps2mag(cpserr, band)
            assert np.isfinite(result)
            cat[f'sy_{band}mag'][i] = result
            cat[f'sy_{band}maglim'][i] = -1
            cat[f'sy_{band}magobs'][i] = True
    elif cps > 0:
        result = galex_query.galex_cps2mag(cps, band)
        assert np.isfinite(result)
        cat[f'sy_{band}mag'][i] = result
        magerr = galex_query.cps2magerr(cps, cpserr)
        cat[f'sy_{band}magerr1'][i] = magerr
        cat[f'sy_{band}magerr2'][i] = magerr
        cat[f'sy_{band}maglim'][i] = 0
        cat[f'sy_{band}magobs'][i] = True
    elif np.isnan(cps):
        # no data, this is returned when one bad was observed and the other was not
        cat[f'sy_{band}magobs'][i] = False
    else:
        raise ValueError

pos = catutils.J2000_positions(cat)
missing_values = cat['sy_nuvmagobs'].mask | cat['sy_fuvmagobs'].mask
while any(missing_values):
    i_failed, = np.nonzero(missing_values)
    for i in tqdm(i_failed):
        try:
            assert np.isfinite(pos.ra[i])
            posi = pos[i]
            ra= posi.ra.to_value('deg')
            dec = posi.dec.to_value('deg')
            pm_ra = posi.pm_ra_cosdec.to_value('mas yr-1')
            pm_dec = posi.pm_dec.to_value('mas yr-1')
            result = galex_query.extract_and_coadd(ra, dec, pm_ra, pm_dec, match_radius=16. / 3600, query_timeout=30.)
            (nuv, nerr), (fuv, ferr) = result
            add_mag(nuv, nerr, 'nuv', i)
            add_mag(fuv, ferr, 'fuv', i)
            missing_values[i] = False
        except ValueError as e:
            msg = str(e)
            if 'Request not filled due to connection errors' in msg:
                continue
            if 'Response ended prematurely' in msg:
                continue

assert np.all(np.isfinite(cat['sy_fuvmag']))


#%% save galex

if toggle_save_galex:
    assert not np.any(cat['id'].mask)
    galex_cols = ['id']
    for band in ['nuv', 'fuv']:
        for suffix in ['', 'err1', 'err2', 'lim', 'obs']:
            name = f'sy_{band}mag{suffix}'
            galex_cols.append(name)
    galex_tbl = cat[galex_cols]
    galex_tbl.write(paths.selection_intermediates / 'galex_magntiudes.ecsv', overwrite=True)


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt5__cut-planet_host_types__add-galex.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt5__cut-planet_host_types__add-galex.ecsv')


#%% estimate Lya based on galex

for band in ['nuv', 'fuv']:
    # values must not be masked or an upper or lower limit
    mag = cat[f'sy_{band}mag'].filled(nan)
    lim_flag = cat[f'sy_{band}maglim'].filled(nan) != 0
    MK = cat['st_teff'].filled(1e4) < 4500. # because schneider GALEX-Lya relationship is for late Ks and Ms only
    d = cat['sy_dist'].filled(nan).quantity
    Flya_1AU = lya.Lya_from_galex_schneider19(mag, d.to_value('pc'), band=band)
    Flya_1AU = Flya_1AU * erg_s_cm2
    bad_galex_lya = ~np.isfinite(Flya_1AU) | lim_flag | ~MK
    cat[f'Flya_1AU_{band}'] = table.MaskedColumn(Flya_1AU, mask=bad_galex_lya)


#%% estimate Lya based on Teff, distance

keepers = cat['stage1'].filled(True)
Teff = cat['st_teff'].filled(nan)

# assume stars without measured period or known youth fall on the "normal" track from Linsky+ 2013
Prot_adopt = cat['st_rotp'].filled(nan)
noP = ~(Prot_adopt > 0)
young = cat['flag_young']
Prot_adopt[young & noP] = 1
Prot_adopt[~young & noP] = 20

Flya_linsky = lya.Lya_from_Teff_linsky13(Teff, Prot_adopt) * erg_s_cm2
valid_lya = Flya_linsky > 0
assert np.all(valid_lya[keepers]) # sanity check
cat['Flya_1AU_Teff_linsky'] = table.MaskedColumn(Flya_linsky, mask=~valid_lya,
                                                   description="Lya flux estimated based on Teff and Prot using Lisnky+ 2013.")


Flya_schneider = lya.Lya_from_Teff_schneider19(Teff) * erg_s_cm2
valid_lya = Flya_schneider > 0
assert np.all(valid_lya[keepers])
cat['Flya_1AU_Teff_schneider'] = table.MaskedColumn(Flya_schneider, mask=~valid_lya,
                                                      description="Lya flux estimated based on Teff using Schneider+ 2019.")



#%% pick best Lya estimate

catutils.add_masked_columns(cat, 'Flya_1AU_adopted', 'float', suffixes=('',), unit='erg s-1 cm-2')
F = cat['Flya_1AU_adopted'] # changes to F will appear in the cat table
fuv = cat['sy_fuvmag'].filled(nan)
fuvlim = cat['sy_fuvmaglim'].filled(nan)
nuv = cat['sy_nuvmag'].filled(nan)
nuvlim = cat['sy_nuvmaglim'].filled(nan)
Teff = cat['st_teff'].filled(nan)
Prot = cat['st_rotp'].filled(nan)

# based on Pineda 2021 for M star saturated activity level
# expected to apply to all given the similarity in saturation levels across
# spectral types for other activity generated flux
max_plausible_Llya_Lbol = 10**-3.0
max_plausible_Llya = max_plausible_Llya_Lbol * 10**cat['st_lum'] * u.Lsun
max_plausible_Flya_1AU = max_plausible_Llya / (4 * np.pi * (1*u.AU)**2)
max_plausible_Flya_1AU = max_plausible_Flya_1AU.to('erg s-1 cm-2')

use_fuv = (np.isfinite(fuv) & (fuvlim == 0) & (cat['Flya_1AU_fuv'].filled(nan) > 0)
           & (cat['Flya_1AU_fuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using FUV-estimated Lya (Schneider+ 19) for {np.sum(use_fuv)} targets.")
F[use_fuv] = cat['Flya_1AU_fuv'][use_fuv]

use_nuv = (F.mask & np.isfinite(nuv) & (nuvlim == 0) & (cat['Flya_1AU_nuv'].filled(nan) > 0)
           & (cat['Flya_1AU_nuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using NUV-estimated Lya (Schneider+ 19) for {np.sum(use_nuv)} targets.")
F[use_nuv] = cat['Flya_1AU_nuv'][use_nuv]

use_Teff1 = F.mask & (Teff > 0) & (Prot > 0) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff+Prot-estimated Lya (Linsky+ 13) for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]

use_Teff1 = F.mask & (Teff > 0) & ~(Prot > 0) & (cat['flag_young'].filled(False)) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff estimated Lya from Linsky+ 13 assuming fast rotation due to known youth for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]

use_Teff2 = F.mask & (cat['Flya_1AU_Teff_schneider'].filled(nan) > 0)
print(f"Using Teff-estimated Lya (Schneider+ 19) for remaining {np.sum(use_Teff2)} targets.")
F[use_Teff2] = cat['Flya_1AU_Teff_schneider'][use_Teff2]

# make sure there aren't any masked or bad values left in F
assert np.all(F.filled(nan) > 0)


#%% flag Lya flux cut, pre-ISM

"""
In the future, if we update to a 3D ISM model, this should be updated to cut without considering the ISM. 
"""

# scale to earth
d = cat['sy_dist'].filled(np.nan)
Flya_earth_no_ISM = cat['Flya_1AU_adopted'].quantity * (1*u.AU/d)**2
Flya_earth_no_ISM = Flya_earth_no_ISM.to('erg s-1 cm-2')
valid_lya_earth = Flya_earth_no_ISM > 0

# check values are all good
keepers = cat['stage1'].filled(True)
valid_lya_earth = Flya_earth_no_ISM > 0
assert np.all(valid_lya_earth[keepers])

cat['Flya_earth_no_ISM'] = table.MaskedColumn(Flya_earth_no_ISM, mask=~valid_lya_earth,
                                                description='Flya_1AU_adopted scaled to earth distance, no ISM abosprtion.')

Flya_earth_min = 1e-15 * erg_s_cm2 # about 10x fainter than the K2-18 spectrum
FLya_basic_too_faint = cat['Flya_earth_no_ISM'].filled(0) < Flya_earth_min
FLya_basic_too_faint_str = (f'Cut because a basic, no ISM estimate of the Lya flux at earth is below {Flya_earth_min}'
                            f'erg s-1 cm-2.')
catutils.flag_cut(cat, FLya_basic_too_faint, FLya_basic_too_faint_str)


#%% transit SNR estimates

"""
This "crude" estimate takes an ad hoc transit profile, scales it to the a maximum depth equivalent to
an opaque rectangle representing an outflow tail that fills the planet's hill sphere vertically and spans
the diameter of the star in length, and shifts it blueward.

There is obvious optimism in an estimate like this. However, it has a lower 
false negative rate than a more sophisticated estimate based on an outflow model in predicting the 
transit detectability for existing known transits. Because we want to encourage detections, we make it the
basis of our target selection.
"""

params = dict(expt_out=3500, expt_in=6000, default_rv=default_sys_rv, transit_range=assumed_transit_range, integrate_range='best', show_progress=True)
SNR_optimistic = transit.blue_wing_occulting_tail_SNR(cat, n_H_percentile=16, lya_percentile=84, **params)
assert np.all(SNR_optimistic > 0)
cat['transit_snr_optimistic'] = table.MaskedColumn(SNR_optimistic)

SNR_nominal = transit.blue_wing_occulting_tail_SNR(cat, n_H_percentile=50, lya_percentile=50, **params)
assert np.all(SNR_nominal > 0)
cat['transit_snr_nominal'] = table.MaskedColumn(SNR_nominal)


#%% basic transit SNR cut

min_SNR = 2
low_SNR = cat['transit_snr_optimistic'].filled(-1) < min_SNR
low_SNR_str = f'Cut because an optimistic estimate of the transit SNR is below {min_SNR}.'
catutils.flag_cut(cat, low_SNR, low_SNR_str)


#%% remove manually IDed bad targets

"""
note that you will need the confirmed and toi tables for this so that the code can check that each target was
initially present and was subsequently cut if it is not found to be sure that if the planet isn't found it isn't
because its name has changed (or some other problematic reason)
"""

# first make sure the ids in the hand checked lists are all accounted for in the initial input tables
# to be certain none will be missed because of a mismatched id
if 'tois_uncut' not in locals():
    tois_uncut = catutils.load_and_mask_ecsv(paths.ipac / 'tois_uncut.ecsv')
if 'confirmed' not in locals():
    confirmed = catutils.load_and_mask_ecsv(paths.ipac / 'confirmed_uncut.ecsv')
all_checked_ids = catutils.read_hand_checked_planets(('remove', 'vetted', 'no-exofop'))
in_tois = np.in1d(all_checked_ids, tois_uncut['toi'].astype(str))
in_confirmed = np.in1d(all_checked_ids, confirmed['pl_name'])
in_input_tables = in_tois | in_confirmed
assert np.all(in_input_tables)
# if this raises an error, figure out why some targets weren't matched and fix it!

ids_to_remove = catutils.read_hand_checked_planets(('remove', 'no-exofop'))
mask_remove = np.in1d(cat['id'], ids_to_remove) | np.in1d(cat['toi'], ids_to_remove)
remove_str = "Removed because planet candidate failed manual vetting."
catutils.flag_cut(cat, mask_remove, remove_str)


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt6__add-lya_transit_snr.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt6__add-lya_transit_snr.ecsv')


#%% PRUNE

cat = catutils.make_a_cut(cat, 'stage1', keepers=('requested_target', 'flag_multiplanet', 'external_lya'))


#%% estimate EUV

Flya_1au = cat['Flya_1AU_adopted'].filled(nan).data
Teff = cat['st_teff'].filled(nan).data
Feuv_1au = lya.EUV_Linsky14(Flya_1au, Teff)
assert np.all(Feuv_1au > 0)
Feuv_1au *= erg_s_cm2
cat['Feuv_1AU'] = table.MaskedColumn(Feuv_1au)

a = cat['pl_orbsmax'].filled(nan).quantity
Feuv_at_planet = Feuv_1au * (1*u.AU/a)**2
assert np.all(Feuv_at_planet > 0)
cat['Feuv_at_planet'] = table.MaskedColumn(Feuv_at_planet)


#%% flag "interesting" systems

"""planets near gap"""
Rgap = empirical.ho_gap_lowlim(cat['pl_orbper'], cat['st_mass'])
dR_gap_dex = np.log10(cat['pl_rade'] / Rgap)
cat['flag_gap_upper_cusp'] = (dR_gap_dex > 0) & (dR_gap_dex < 0.1)

# water worlds
# consider them water world compatible if they are within 1 sigma of the Luque & Palle 2022 water world line"""
lp22 = ref.lp22
pp = lp22['m'] * u.Mearth / (4 / 3 * np.pi * (lp22['r'] * u.Rearth) ** 3)
pp = pp.to_value('g cm-3')
p_water = np.interp(cat['pl_bmasse'], lp22['m'], pp, left=np.nan, right=np.nan)
rho_diff_sig = np.abs((cat['pl_dens'] - p_water) / cat['pl_denserr1'])
water_world = (rho_diff_sig < 1) & (cat['pl_bmasse'] < 30) & (cat['pl_rade'] < 3)
cat['flag_water_world'] = water_world


# flag escape on/off multis
above_euv_hydro_threshold = cat['Feuv_at_planet'] > 1 # lower xuv threshold to catch stars where code likely underpredicts xuv flux
has_HHe = cat['flag_gaseous']
has_outflow = has_HHe & above_euv_hydro_threshold
no_outflow = has_HHe & ~above_euv_hydro_threshold
print(f'{sum(no_outflow)} planets in catalog are gaseous but too weakly irradiated to support an outflow.')
cat['flag_outflow'] = table.MaskedColumn(has_outflow)

positions_cut3 = coord.SkyCoord(cat['ra'], cat['dec'])
positions_outflow = positions_cut3[has_outflow.filled(False)]
positions_no_outflow = positions_cut3[no_outflow.filled(False)]
i_matches_no_outflow, i_matches_outflow, _, _ = positions_outflow.search_around_sky(positions_no_outflow, seplimit=0.000001*u.deg)
i_outflow, = np.nonzero(has_outflow.filled(False))
i_no_outflow, = np.nonzero(no_outflow.filled(False))
on_off_multis = np.hstack((i_outflow[i_matches_outflow], i_no_outflow[i_matches_no_outflow]))
on_off_multis = np.unique(on_off_multis)
_description = 'systems contains planets with and without H/He envelopes'
cat['flag_outflow_and_not_in_sys'] = table.MaskedColumn([False] * len(cat), fill_value=False,
                                                          description=_description)
cat['flag_outflow_and_not_in_sys'][on_off_multis] = True


# HHe + no HHe multis
positions_cut3 = coord.SkyCoord(cat['ra'], cat['dec'])
has_HHe = cat['flag_gaseous']
positions_HHe = positions_cut3[has_HHe.filled(False)]
positions_noHHe = positions_cut3[(~has_HHe).filled(False)]
i_matches_noHHe, i_matches_HHe, _, _ = positions_HHe.search_around_sky(positions_noHHe, seplimit=0.000001*u.deg)
i_HHe, = np.nonzero(has_HHe.filled(False))
i_noHHe, = np.nonzero((~has_HHe).filled(False))
H_no_H_multis = np.hstack((i_HHe[i_matches_HHe], i_noHHe[i_matches_noHHe]))
H_no_H_multis = np.unique(H_no_H_multis)
_description = 'systems contains planets with and without H/He envelopes'
cat['flag_gas_and_rocky_in_sys'] = table.MaskedColumn([False] * len(cat), fill_value=False,
                                                        description=_description)
cat['flag_gas_and_rocky_in_sys'][H_no_H_multis] = True


#  super puffs
p_sp = 0.3  # based on Libby-Roberts 2020
super_puff = (cat['pl_dens'] < p_sp) & (cat['pl_bmasse'] < 30)
cat['flag_super_puff'] = super_puff


# high TSM

Rp = cat['pl_rade']#.quantity.to_value('Rearth')
Teq = cat['pl_eqt']#.quantity.to_value('K')
Mp = cat['pl_bmasse']#.quantity.to_value('Mearth')
Rs = cat['st_rad']#.quantity.to_value('Rsun')
J = cat['sy_jmag']#.quantity.to_value('')
R1 = (Rp < 1.5)
R2 = (Rp >= 1.5) & (Rp < 2.)
R3 = (Rp >= 2) & (Rp < 2.75)
R4 = (Rp >= 2.75) & (Rp < 4)
R5 = (Rp >= 4) & (Rp < 10.)
TSM_raw = Rp ** 3 * Teq / Mp / Rs ** 2 * 10 ** (-J / 5)
cat['TSM'] = TSM_raw
cat['TSM'][R1] = TSM_raw[R1] * 0.19
cat['TSM'][R2] = TSM_raw[R2] * 1.26 * 2.3 / 18
cat['TSM'][R3] = TSM_raw[R3] * 1.26
cat['TSM'][R4] = TSM_raw[R4] * 1.28
cat['TSM'][R5] = TSM_raw[R5] * 1.15
cat['flag_high_TSM'] = (((cat['pl_rade'] <= 1.5) & (cat['TSM'] > 10))
                          | ((cat['pl_rade'] > 1.5) & (cat['TSM'] > 90)))

# flag measured mass

cat['flag_measured_mass'] = cat['pl_bmassprov'] == 'Mass'


# flag Lya or FUV already

cat['flag_has_FUV_or_Lya_already'] = (cat['external_lya'].astype(float) + cat['external_fuv'].astype(float)) == 1


#%% score systems

cat['score'] = cat['transit_snr_nominal']

# zero out score for planets where atm escape known to be unlikely
outflow_unlikely = ~cat['flag_gaseous'] | ~cat['flag_outflow']
zero_out = outflow_unlikely.filled(False)
cat['score'][zero_out] = 0

# pick top score for each host
catutils.pick_planet_parameters(cat, 'score', np.max, 'score_host')

_, unq_inverse = np.unique(cat['score_host'], return_inverse=True)
rank = unq_inverse.max() - unq_inverse + 1
cat['stage1_rank'] = table.MaskedColumn(rank)


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt7__cut-low-snr__add-flags-scores.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__cut-low-snr__add-flags-scores.ecsv')


#%% verify ISR table

"""check nobody used a hostname in the table that doesn't match to a target"""
unmatched = catutils.unmatched_names(ref.mdwarf_isr['Target'], merged['hostname'])
if len(unmatched) > 0:
    raise KeyError(f'These names in the M dwarf ISR table have no match: {unmatched.tolist()}')


#%% setup for orbit allocation

if not toggle_mid_cycle_update:
    available_free = allocated_orbits
    available_planned = 0
    lemons = []
    planned_targets = []
else:
    # this is a mid-cycle update, in which case we need to track what visits have already been cemented
    # and what lemons have been identified
    latest_status_path = dbutils.pathname_max(paths.status_snapshots, 'HST-17804-visit-status*.xml')
    labels_in_phase2 = dc.parse_visit_labels_from_xml_status(latest_status_path)

    # eliminate redo orbits for failed observations
    # these can be identified by the fact that they start with a number
    labels_in_phase2 = [lbl for lbl in labels_in_phase2 if lbl[0] in string.ascii_uppercase]

    # add and remove according to hand-picked stuff
    labels_in_phase2.extend(hand_add_visits)
    [labels_in_phase2.remove(_lbl) for _lbl in hand_remove_visits]

    # load the hand updated observing status sheet to ID lemons
    path_main_table = dbutils.pathname_max(paths.status_snapshots, 'Observation Progress*.xlsx')
    progress_tbl = catutils.read_excel(path_main_table)
    progress_tbl.add_index('Target')
    drop = progress_tbl['Pass to\nStage 1b?'] == False
    lemons = progress_tbl['Target'][drop]
    lemons = [ref.locked2archive_name_map.get(name, name) for name in lemons] # update to the latest archive names

    # make sure every lemon has a match
    assert np.all(np.in1d(lemons, cat['hostname']))

    # load the visit label table
    labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')
    labeltbl.add_index('target')

    # list targets that are in the phase 2
    lya_planned = np.in1d(labeltbl['base'], labels_in_phase2)
    fuv_planned = np.in1d(labeltbl['pair'], labels_in_phase2)

    # update to the latest archive names
    _targnames = [ref.locked2archive_name_map.get(name, name) for name in labeltbl['target']]

    # make a table for accounting
    plantbl = table.Table(data=(_targnames, lya_planned, fuv_planned),
                          names='name lya fuv'.split())
    available_planned = lya_planned | fuv_planned
    plantbl = plantbl[available_planned]
    planned_targets = plantbl['name'].tolist()
    plantbl.add_index('name')

    # make sure all targets in phase 2 have a match
    assert np.all(np.in1d(plantbl['name'], cat['hostname']))

    # count how many freed-up orbits there are that we can allocate
    plantbl['lemon'] = False
    lemons_in_plan = [l for l in lemons if l in plantbl['name']]
    ilmn = plantbl.loc_indices[lemons_in_plan]
    plantbl['lemon'][ilmn] = True
    available_planned = sum(plantbl['lya']) + sum(plantbl['fuv'] & ~plantbl['lemon'])
    available_free = allocated_orbits - available_planned
    absent_fuv_mask = plantbl['lemon'] & ~plantbl['fuv']
    print('The fuv visits for these lemons were already absent in the Phase II:')
    print('\t' + ', '.join(plantbl['name'][absent_fuv_mask].tolist()))

    # add some columns for tracking as orbits are allocated
    plantbl['lya_registered'] = False
    plantbl['fuv_registered'] = False
    plantbl['lemon'] = np.in1d(plantbl['name'], lemons)
    catutils.scrub_indices(plantbl)
    plantbl = plantbl['name lya lya_registered fuv fuv_registered lemon'.split()] # grouping columns for easy viewing
    plantbl.add_index('name')


"""remember there are still targets we don't want to observe in the table for tracking purposes, 
so better clean those before we start building a list
this must happen before filtering for hosts or else some planets with stage1=False may be selected"""
candidates = cat[cat['stage1'].filled(False)
                 | (cat['requested_observed_transit'].filled(0) > 0)]

# get just the hosts since we're not observing planets in stage 1
candidates = catutils.planets2hosts(candidates)
candidates.sort('score_host', reverse=True)

# prep columns for which gratings will be used
catutils.set_index(cat, 'tic_id')
catutils.add_filled_masked_column(cat, 'stage1_g140m', nan, mask=True)
catutils.add_filled_masked_column(cat, 'stage1_g140l', nan, mask=True)
catutils.add_filled_masked_column(cat, 'stage1_e140m', nan, mask=True)

available_backup = backup_orbits
if backup_orbits:
    catutils.add_filled_masked_column(cat, 'stage1_backup', False, dtype=bool)

cat['stage1_orbit_total'] = table.MaskedColumn(len(cat), mask=True, dtype=int)


#%% build target list and allocate orbits, THIS IS WHERE THE MAGIC HAPPENS!

"""allocate orbits target by target working down the ranks"""
selected_tic_ids = []
observations_to_verify = []
i = 0
stela_orbit_count = 0
while (available_planned > 0) or (available_free > 0) or (available_backup > 0):
    if i >= len(candidates):
        raise ValueError('Reached the end of the candidate list without allocating all orbits. '
                         'Running the next cell will likely reveal what happened based on discrepancies.')

    print(f"\rTarget: {i+1}/{len(candidates)}", end="", flush=True)
    # find the indices of planets associated with the target host and whether any are already in stage 1
    # fom an earlier iteration of this loop
    host = candidates[i]
    name = host['hostname']
    tic_id_ = host['tic_id']
    j = cat.loc_indices[tic_id_]
    j = np.atleast_1d(j)
    in_stage1 = (cat['stage1'][j].filled(False) == True)

    # get key info about external observations and if e140m is needed
    valid_external_lya_obs = host['external_lya_status'] in ['planned', 'valid', 'unverified']
    valid_external_fuv_obs = host['external_fuv_status'] in ['planned', 'valid', 'unverified']
    lya_requires_e140m = apt.does_mdwarf_isr_require_e140m(name, 'lya')
    fuv_requires_e140m = apt.does_mdwarf_isr_require_e140m(name, 'fuv')

    # mark for verification if target would have been included but for having been already observed
    if available_free > 0:
        for band in ('lya', 'fuv'):
            if host[f'external_{band}_status'] == 'unverified':
                observations_to_verify.append(f'{name} {band}')

    # allocate orbits
    g140m, g140l, e140m = 0, 0, 0
    if not valid_external_lya_obs:
        if lya_requires_e140m:
            e140m = 1
        else:
            g140m = 1
    if not valid_external_fuv_obs:
        if name not in lemons:
            if fuv_requires_e140m:
                e140m = 1
            else:
                g140l = 1
    assert not (g140m and e140m and g140l)

    # counting...
    stela_orbits = g140m + g140l + e140m

    # classify the target as externally observed, already in plan, new addition, or backup
    if stela_orbits == 0:
        status = 'external'
    elif name in planned_targets:
        iplan = plantbl.loc_indices[name]
        planned = plantbl[iplan]
        if stela_orbits > available_planned:
            raise ValueError('Trying to allocate more planned orbits than expected.')
        status = 'planned'
        available_planned -= stela_orbits

        # record for accounting
        if not e140m:
            plantbl['lya_registered'][iplan] = g140m
            plantbl['fuv_registered'][iplan] = g140l
        else:
            # if e140m is the only mode used, things get trick bc we messed up some labels in the plan
            if g140m and g140l:
                raise ValueError('Trying to use all three modes for planned target.')
            elif not (g140m or g140l):
                if planned['lya'] and planned['fuv']:
                    raise ValueError('Code wants to allocate a single lya or fuv orbit but two are already planned.')
                else:
                    # match the "registered" column to whether we labeled the E140M visit as lya or fuv
                    plantbl['lya_registered'][iplan] = planned['lya']
                    plantbl['fuv_registered'][iplan] = planned['fuv']
            elif g140m:
                plantbl['lya_registered'][iplan] = True
                plantbl['fuv_registered'][iplan] = True
            elif g140l:
                raise ValueError("E140M used for Lya then G140L for FUV, which shouldn't happen")
            else:
                raise ValueError("how did we get here? I thought all options were accounted for")
    else:
        if available_free > 0:
            status = 'addition'
            available_free -= stela_orbits
        elif available_backup > 0:
            status = 'backup'
            available_backup -= stela_orbits
        else:
            i += 1
            continue

    if status == 'external':
        remove = j[in_stage1]
        cat['stage1'][remove] = False
        cat['decision'][remove] = 'Rejected because observations already exist.'
    elif status in ['planned', 'addition']:
        stela_orbit_count += stela_orbits

        # mark orbits for host, mark in stage 1
        selected_tic_ids.append(tic_id_)
        cat['stage1_g140m'][j] = g140m
        cat['stage1_g140l'][j] = g140l
        cat['stage1_e140m'][j] = e140m
        cat['stage1'][j] = True
        cat['stage1_orbit_total'][j] = stela_orbit_count

        # mark host and best planet as selected
        select = j[in_stage1]
        cat['decision'][select] = 'Host selected.'

        # mark planets in same system that were previously rejected, if any, as now included
        reintroduce = j[~in_stage1]
        prefix = 'Host ultimately selected. Planet previously '
        readd_decisions = [prefix + decision for decision in cat['decision'][reintroduce]]
        cat['decision'][reintroduce] = readd_decisions
    elif status == 'backup':
        cat['stage1_backup'][j] = True
        k = j[in_stage1]
        cat['decision'][k] = 'Reserved as backup target.'
    else:
        raise ValueError('status variable has unexpected value')
    i += 1

print('\n\n')

#%% check for discrepancies
if toggle_mid_cycle_update:
    discrepant = ((plantbl['lya'] ^ plantbl['lya_registered'])  # ^ is the xor operator (returns true if the two differ)
                  | (plantbl['fuv'] & ~plantbl['fuv_registered'] & ~plantbl['lemon'])
                  | (~plantbl['fuv'] & plantbl['fuv_registered']))
    if np.any(discrepant):
        print('Disrepancies present! Resolve these.')
        print('')
        plantbl[discrepant].pprint(-1, -1)


#%% flag selections

cut_mask = np.ones(len(cat), bool)
i_keep = cat.loc_indices[selected_tic_ids]
cut_mask[i_keep] = False
cut_comment = 'Cut because predicted SNR ranked too low.'
catutils.flag_cut(cat, cut_mask, cut_comment)


#%% checkpoint

if toggle_checkpoint_saves:
    cat.write(paths.selection_intermediates / 'chkpt8__target-build.ecsv', overwrite=True)


#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')


#%% save list of targets that need their planned or archival observations verified

"""If the existing observations are bad, we should reobserve these!"""

np.savetxt(paths.selection_outputs / 'verify_archival_observations.txt', list(observations_to_verify), fmt='%s')


#%% save reports on special requests

request_cols = [name for name in cat.colnames if 'requested' in name]
request_cols.remove('requested_target')
savecol_fmt_pairs = (
    ('id', 's'),
    ('stage1', ' '),
    ('stage1_rank', '.0f'),
    ('Flya_1AU_adopted', '.2f'),
    ('Flya_earth_no_ISM', '.2e'),
    ('transit_snr_nominal', '.1f'),
    ('st_rad', '.2f'),
    ('pl_bmasse', '.2f'),
    ('pl_orbsmax', '.3f'),
    ('st_rotp', '.1f'),
    ('st_age', '.1f'),
    ('decision', 's'))
savecols,_ = zip(*savecol_fmt_pairs)
for col,fmt in savecol_fmt_pairs:
    cat[col].format = fmt
for name in request_cols:
    mask = cat[name].filled(0) > 0
    requested = cat[mask]
    requested.sort(name)
    source = name.replace('requested_', '')
    filepath = paths.reports / f'{source} target evaluation results.txt'
    with open(filepath, 'w') as f:
        sys.stdout = f
        requested[savecols].pprint(-1, -1, align='<')
        sys.stdout = sys.__stdout__

#%% PRUNE to final selections

cat.sort('stage1_rank')
selected = cat[cat['stage1'].filled(False)]
print(f'{len(selected)} planets orbit the selected stage 1 targets.')
roster_set = [selected]
if backup_orbits > 0:
    backup_cat = cat[cat['stage1_backup'].filled(False)]
    print(f'{len(backup_cat)} planets orbit the selected stage 1 backup targets.')
    roster_set.append(backup_cat)

roster = table.vstack(roster_set)


#%% save selected planets and hosts

if toggle_save_outputs:
    selected.write(paths.selection_outputs / 'stage1_planet_catalog.ecsv', overwrite=True)
    selected_hosts = catutils.planets2hosts(selected)
    selected_hosts.write(paths.selection_outputs / 'stage1_host_catalog.ecsv', overwrite=True)

    if backup_orbits > 0:
        backup_cat.write(paths.selection_outputs / 'stage1_backup_planet_catalog.ecsv', overwrite=True)
        backup_hosts = catutils.planets2hosts(backup_cat)
        backup_hosts.write(paths.selection_outputs / 'stage1_backup_host_catalog.ecsv', overwrite=True)

    # simplified table of key info
    save_info_sets = (
        ('id', 's'),
        ('st_teff', '.0f'),
        ('pl_orbper', '.1f'),
        ('pl_rade', '.2f'),
        ('sy_dist', '.1f'),
        ('TSM', '.1f'),
        ('transit_snr_nominal', '.1f')
    )
    savecols, _, = zip(*save_info_sets)
    for name, fmt in save_info_sets:
        selected[name].format = fmt
    filepath = paths.selection_outputs / f'stage1_planets_basic_info.txt'
    with open(filepath, 'w') as f:
        sys.stdout = f
        selected[savecols].pprint(-1, -1, align='<')
        sys.stdout = sys.__stdout__


#%% save diff table

"""These are simplified tables to enable comparisons of target pools and ordering."""

if toggle_save_difftbl:
    save_info_sets = (
        ('hostname', 'name', 's'),
        ('stage1', 'slctd', '.0f'),
        ('stage1_rank', 'rank', '.0f')
    )
    savecols, _, _ = zip(*save_info_sets)
    difftbl = roster[savecols]
    for old, new, fmt in save_info_sets:
        difftbl.rename_column(old, new)
        difftbl[new].format = fmt
    timetag = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    savename = f'selection_diff_{timetag}_{diff_label}.csv'
    difftbl.write(paths.difftbls / savename)


#%% select TOI list to hand vet

"""
a list of TOIs that are still candidates have not been manually vetted so that we can vet them before actually
using them

these targets should then be put into new "remove" and "vetted" lists in the inputs/hand_checked directory
"""

# verify that there aren't any duplicates
vetted_ids = catutils.read_hand_checked_planets(('vetted',))
remove_ids = catutils.read_hand_checked_planets(('remove',))
all_ids = np.hstack((vetted_ids, remove_ids))
idcheck, check_count = np.unique(all_ids, return_counts=True)
if np.any(check_count > 1):
    duplicated_ids = idcheck[check_count > 1]
    raise ValueError(f'These targets appear more than once in the hand checked lists {duplicated_ids.tolist()}')

if toggle_save_outputs:
    tfop_disp = roster['tfopwg_disp'].filled('').astype(str)
    unconfirmed = np.char.count(tfop_disp, 'PC') > 0

    # remove those that have already been hand checked
    checked = np.in1d(roster['toi'], all_ids)
    unconfirmed[checked] = False

    list_to_vet = roster['toi'][unconfirmed].tolist()
    np.savetxt(paths.selection_outputs / 'tess_candidates_in_need_of_vetting.txt', list_to_vet, fmt='%s')



#%% APT info

targets = catutils.planets2hosts(roster)
n = len(targets)

guaranteed = targets['stage1'].filled(False) == True
n_orbits_check = sum(targets[f'stage1_{grating}'][guaranteed].sum() for grating in 'g140m g140l e140m'.split())
assert n_orbits_check in [allocated_orbits, allocated_orbits + 1]
assert n_orbits_check == selected['stage1_orbit_total'].max()

# start compact table for hand input to APT
apt_info = targets[['hostname']]
apt_info.rename_column('hostname', 'name')
apt_info['GM'] = targets['stage1_g140m']
apt_info['GL'] = targets['stage1_g140l']
apt_info['EM'] = targets['stage1_e140m']

no_simbad_match = targets['simbad_id'].mask
n_no_simbad = sum(no_simbad_match)
if n_no_simbad > 0:
    no_simbad_names = targets['hostname'][no_simbad_match].tolist()
    print(f'Warning: these targets will not match to an object in SIMBAD {no_simbad_names}.')

simbad = apt.get_simbad_info(targets['simbad_id'].filled('?'))

apt.fill_spectral_types(targets, simbad)
apt.fill_optical_magnitudes(targets, simbad)
apt_target_table = apt.make_apt_target_table(targets, simbad)
past_targets = table.Table.read(paths.locked / 'apt_target_export.targets', format='ascii.csv', data_start=5, header_start=4)

# identify new targets by coordinates in case of name changes
apt_coords = coord.SkyCoord(apt_target_table['RA'], apt_target_table['DEC'], unit=(u.hourangle, u.deg))
past_coords = coord.SkyCoord(past_targets['RA'], past_targets['DEC'], unit=(u.hourangle, u.deg))
i_past, i_apt, _, _ = apt_coords.search_around_sky(past_coords, 0.1*u.arcsec) # APT rounds DEC to the second decimal place
mask_new = np.ones(len(apt_target_table), bool)
mask_new[i_apt] = False

# print any name changes
current_names = apt_target_table['Target Name']
past_names = past_targets['# Name in the Proposal']
name_change_mask = ~np.isin(past_names[i_past], current_names[i_apt])
current_names = current_names[i_apt][name_change_mask]
past_names = past_names[i_past][name_change_mask]
name_change_tbl = table.Table((past_names, current_names), names=['previous','current'])
print('')
print('Old names of targets that have have had a name change')
print('')
name_change_tbl.pprint(-1)

if toggle_save_outputs:
    apt.write_target_table(apt_target_table[mask_new], paths.selection_outputs / 'new_targets_for_apt.csv', overwrite=True)

acq_setup_table = apt.acquisition_setup(targets)
apt_info['acq'] = acq_setup_table['acq_filter']
apt_info['Tacq'] = acq_setup_table['acq_Texp']

g140m_buffers = apt.buffer_times(targets, 'g140m')
g140l_buffers = apt.buffer_times(targets, 'g140l')
e140m_buffers = apt.buffer_times(targets, 'e140m')
apt_info['GMbuff'] = g140m_buffers
apt_info['GLbuff'] = g140l_buffers
apt_info['EMbuff'] = e140m_buffers

apt_info['Mdwarf'] = np.char.count(targets['st_spectype'].filled('M').astype(str), 'M') > 0

for col in apt_info.columns:
    if apt_info[col].dtype == float:
        apt_info[col].format = '.1f'

# new info for old targets
i_apt_, i_past_, _, _ = past_coords.search_around_sky(apt_coords, 0.1*u.arcsec)
i_apt_ = i_apt_[np.argsort(i_past_)]
old_target_new_info = apt_target_table[['Target Name', 'Other Fluxes', 'Comments']][i_apt_]
if toggle_save_outputs:
    old_target_new_info.write(paths.selection_outputs / 'new_info_old_targets.csv', overwrite=True)


#%% M dwarfs not in ISR table

"""You will need to rerun the selection if any of these must be observed with E140M, since that will make more orbits
available."""

if toggle_save_outputs:
    Ms = apt_info['Mdwarf']
    Mnames = apt_info['name'][Ms]
    not_in_isr = ~np.in1d(Mnames, ref.mdwarf_isr['Target'])
    Ms_to_add = Mnames[not_in_isr]
    np.savetxt(paths.selection_outputs / 'Mdwarfs_to_add_to_ISR_table.txt', Ms_to_add, fmt='%s')


#%% visit labels for APT

labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')

# be sure all targets have (or had, prior to cuts) a match
# if not, fix it
unmatched = catutils.unmatched_names(labeltbl['target'], merged['hostname'])
if len(unmatched) > 0:
    raise KeyError(f'These targets in the visit label table have no match: {unmatched.tolist()}')

new_batch_no = labeltbl['batch'].max() + 1
existing_base_labels = list(map(apt.VisitLabel, labeltbl['base'].tolist()))
last_base_label = max(existing_base_labels)

targets_selected = catutils.planets2hosts(selected)
for row in targets_selected:
    name = row['hostname']
    if name not in labeltbl['target']:
        last_base_label, pair = last_base_label.next_pair()
        new_row = [name, str(last_base_label), str(pair), new_batch_no, '', '']
        labeltbl.add_row(new_row)

catutils.set_index(labeltbl, 'target')
in_labeltbl = np.in1d(apt_info['name'], labeltbl['target'])
apt_info['lbl1'] = table.MaskedColumn(length=len(apt_info), mask=True, dtype='object')
apt_info['lbl2'] = table.MaskedColumn(length=len(apt_info), mask=True, dtype='object')
apt_info['lbl1'][in_labeltbl] = labeltbl.loc[apt_info['name'][in_labeltbl]]['base']
apt_info['lbl2'][in_labeltbl] = labeltbl.loc[apt_info['name'][in_labeltbl]]['pair']

if toggle_save_outputs:
    catutils.set_index(apt_info, 'name')
    in_apt = np.isin(labeltbl['target'], apt_info['name'])
    names_in_apt = labeltbl['target'][in_apt]
    apt_visit_info = apt_info.loc[names_in_apt]
    apt_visit_info.write(paths.selection_outputs / 'info_for_apt_entry.csv', overwrite=True)

    catutils.set_index(apt_target_table, 'Target Name')
    _labeltbl_apt_names = apt.cat2apt_names(labeltbl['target'])
    in_targets = np.isin(_labeltbl_apt_names, apt_target_table['Target Name'])
    names_in_targets = _labeltbl_apt_names[in_targets]
    apt_visit_info = apt_target_table.loc[names_in_targets]
    apt_visit_info.write(paths.selection_outputs / 'info_all_targets.csv', overwrite=True)


#%% save the label map

if toggle_save_visit_labels:
    labeltbl.write(paths.locked / 'target_visit_labels.ecsv', overwrite=True)


#%% examples for printing apt info


"""print the whole table"""
apt_info.pprint(-1)

# find a target or visit and print its row
apt_info.add_index('name')
print(apt_info.loc['name', 'GJ 143'])


#%% check that no targets exceed bright limits


active, _, _ = apt.categorize_activity(targets)
for grating in ['g140m', 'g140l', 'e140m']:
    max_local = 75
    max_global_inactive = 200000 if grating == 'e140m' else 30000
    max_global_active = 800000 if grating == 'e140m' else 12000
    obs_mask = targets[f'stage1_{grating}'].filled(0) == 1
    active_obs = active[obs_mask]
    cps_local, cps_global = apt.count_rate_estimates(targets[obs_mask], grating)
    local_violations = cps_local > max_local
    global_violations = np.zeros(len(cps_global), bool)
    global_violations[~active_obs] = cps_global[~active_obs] > max_global_inactive
    global_violations[active_obs] = cps_global[active_obs] > max_global_active
    all_violations = local_violations | global_violations
    if np.any(all_violations):
        raise ValueError('There were count rate violations. You will need to poke around to find out what and '
                         'deal with them. Some may be dealt with just by using a more accurate Lya estimate.')


#%% table for ETC input


active, _, _ = apt.categorize_activity(targets)
targets['variable'] = active

etcrows = apt.find_nearest_etc_rows(targets['st_teff'].filled(nan), 'g140m')
targets['template_teff'] = etcrows['Teff']

FUVmags = apt.conservatively_bright_FUV(targets)
targets['FUV'] = FUVmags

etctbl = labeltbl[['target', 'base', 'pair']]
names = labeltbl['target']
isintargets = np.in1d(names, targets['hostname'])
etctbl = etctbl[isintargets]

etctbl['target'] = etctbl['target'].astype('object')
joincols = ['hostname', 'st_teff', 'template_teff', 'FUV', 'variable']
etctbl = table.join(etctbl, targets[joincols], join_type='left', keys_left='target', keys_right='hostname')
etctbl.remove_column('hostname')

etctbl.meta['Notes'] = ("FUV values are measured values from GALEX when available and conservatively bright "
                        "estimates when not available.",
                        "Variable classification is based on indications of activity from age, rotation, or "
                        "measured FUV flux. If there is no measurement indicating inactivity, the star is assumed"
                        "active. These classifications *do not* follow the protocol for M dwarfs and so should not"
                        "be used for M dwarf clearance per ISR-2017.")

if toggle_save_outputs:
    etctbl.write(paths.selection_outputs / 'etc_inputs_for_brightness_checks.ecsv', overwrite=True)


#%% list of borderline SpTs for ISR

targets = catutils.planets2hosts(roster)
spt_flags = apt.fill_spectral_types(targets, simbad, return_flags=True)

# pull spectral type and subtypes in consistent formats
spt_letters, spt_numbers = [], []
for entry in targets['st_spectype'].tolist():
    match = re.match(r'([A-Z])', entry)  # Match first letter (spectral type)
    number_match = re.search(r'\d+(\.\d+)?', entry)  # Match first number (subtype)
    if match:
        spt_letters.append(match.group(1))
    else:
        spt_letters.append('')
    if number_match:
        spt_numbers.append(float(number_match.group(0)))
    else:
        spt_numbers.append(-1)
spt_letters, spt_numbers = map(np.array, (spt_letters, spt_numbers))

# identify targets with catalog or simbad type that is concerningly close to an ISR edge
spt_from_lit = spt_flags <= 1
spt_questionable = (simbad['SP_QUAL'].filled('F') >= 'C') | (simbad['SP_QUAL'].filled('') == '')
KM_edge_spt = ((spt_letters == 'K') & (spt_numbers >= 7))
midM_edge_spt = ((spt_letters == 'M') & ((spt_numbers >= 2) & (spt_numbers < 3)))
lateM_edge_spt = ((spt_letters == 'M') & ((spt_numbers >= 5) & (spt_numbers < 6)))
spt_needs_investigation = (spt_from_lit & spt_questionable
                           & (KM_edge_spt | midM_edge_spt | lateM_edge_spt))

# identify targets where SpT is based on Teff and Teff is within 200 K of an ISR division
spt_from_teff = spt_flags == 2
Teff_slop = 200
KM_edge_teff = ((targets['st_teff'] > 3850) & (targets['st_teff'] < 3850 + Teff_slop))
midM_edge_teff = ((targets['st_teff'] > 3430) & (targets['st_teff'] < 3430 + Teff_slop))
lateM_edge_teff = ((targets['st_teff'] > 2810) & (targets['st_teff'] < 2810 + Teff_slop))
teff_needs_investigation = (spt_from_teff
                            & (KM_edge_teff | midM_edge_teff | lateM_edge_teff))

needs_investigation = spt_needs_investigation | teff_needs_investigation & targets['stage1'].filled(False)
targets_to_investigate = targets[['hostname', 'st_spectype']][needs_investigation]

if toggle_save_outputs:
    targets_to_investigate.write(paths.selection_outputs / 'targets_needing_spT_followup.csv', overwrite=True)


#%% optional: check that acq entry in APT is correct
"""If they aren't way off, probably no need to correct and risk throwing off planning"""

# parse latest pro file
path = dbutils.pathname_max(paths.other, '*.pro')
acq_apt_tbl = apt.parse_acqs_from_formatted_listing(path)

# get apt names so we can match tables
_newnames = [ref.archive2locked_name_map.get(name, name) for name in apt_info['name']]
aptnames = apt.cat2apt_names(_newnames)
apt_info['aptname'] = aptnames

# run through checks
catutils.set_index(apt_info, 'aptname')
for row in acq_apt_tbl:
    name, aper, expt = row
    try:
        intended_aper = apt_info.loc[name]['acq']
        intended_expt = apt_info.loc[name]['Tacq']
        if (aper != intended_aper) or not np.isclose(intended_expt, expt, atol=0.05):
            print(f'Setting mismatch for {name}')
            print(f'\tIn APT {aper} {expt:.1f}')
            print(f'\tIntended {intended_aper} {intended_expt:.1f}')
    except KeyError:
        print(f'{name} not in apt_info table')

#%% End Note
"""Once you have the pipeline has completed to your satisfication, 
I recommend you archive a copy of the input, intermediate, and output folders labeled with todays date.
Future runs will overwrite the local files."""

