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
import empirical
import paths
import catalog_utilities as catutils
from processing import visit_status_xml_parser
from lya_prediction_tools import transit
from target_selection import status_tbl
from target_selection_tools import galex_query
from target_selection_tools import duplication_checking as dc
from target_selection_tools import reference_tables as ref
from target_selection_tools import query, columns, apt
import database_utilities as dbutils

erg_s_cm2 = u.Unit('erg s-1 cm-2')


#%% settings and toggles

allocated_orbits = 202 # reserve 2 orbits for HD 260655 and GJ 1214 request

# if true, pulls the latest versions of these files from the progress review folder:
# - visit list from status xml
# - manually udpated observation status excel sheet
# then makes sure all planned visits are kept and lemons are dropped
# whether an fuv visit is kept or dropped depends on the "pass to stage 1b?" column in the excel sheet
toggle_mid_cycle_update = True

# use these to specify visits we are going to add back in or remove for whatever reason
# for example, in the May update I wanted to remove the K2-72 fuv visit Z2 because of an error in the clearance sheet
# and add in NB for TOI-2015 due to revised pass through cut

# can use to keep FUV visits for transit targets even if there is external data
hand_add_visits = []

# can use this for targets that should no longer make rank for stage 1 and that are not flight ready, not Lya bright, and without archival lya transit
# hand_remove_visits = 'BT BU BV BW BX BY BZ CA CB CC OT OU OV OW OY OZ PA PB PC'.split()
hand_remove_visits = []

# adds this many orbits-worth of backup targets if desired so you can get ahead on vetting them
backup_orbits = 0
# backup_orbits = int(round(0.2 * 204))

toggle_plots = True

toggle_save_outputs = True
toggle_save_galex = True
toggle_save_difftbl = True
toggle_save_new_stela_names = True # necessary to avoid errors if, e.g., new hosts have shown up in exoplanet archvive
toggle_save_visit_labels = True

toggle_redo_all_galex = False # only needed if galex search methodology modified
toggle_remake_filtered_hst_archive = False  # as of 2026-04-08, last update was 12-01-2025 and I've remade since then
toggle_refresh_archival_transit_list = False
toggle_save_transit_list = False
archival_transit_blacklist = ['GJ 3929', 'GJ 486', 'L 98-59']

diff_label = 'target-backfill-ttrb-only-2026-04'
toggle_checkpoint_saves = True
toggle_target_removal_test = False # removes targets to see if sort order changes as a test for bugs
assumed_transit_range = [-100, 50] # based on typical ranges from actual transit observations
default_sys_rv = "ism"


#%% load old merged table (if available)

"""This is needed in scattered places even if you skip the block that creates it, so it is wise to load it."""
merged_path = paths.selection_intermediates / 'chkpt1__merged-confirmed-tois-community.ecsv'
if merged_path.exists():
    merged = catutils.load_and_mask_ecsv(merged_path)



#%% load checkpoint

if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__cut-low-snr__add-flags-scores.ecsv')




#%% mark external lya transits

if toggle_refresh_archival_transit_list:
    from target_selection_tools import log_archival_transits as lat
    from processing import observation_table as obt

    # allow just about any data through for this intial "has transit" definition
    # we should be more strict later when we consider reobserving actual transits
    usbldef = obt.UsabilityDefinition() # this will let it all through
    action_on_unknown_flagsnotes = 'ignore'

    check_mask = cat['external_lya'].filled(False)
    check_tics = cat['tic_id'][check_mask].filled(0)
    check_names = sorted(list(set(dbutils.stela_name_tbl.loc['tic_id', check_tics]['hostname'])))
    names_w_transits = lat.refresh_observed_transit_list(
        target_hostnames=check_names,
        usability_definition=usbldef,
        action_on_unknown=action_on_unknown_flagsnotes,
    )

    names_w_transits = set(names_w_transits)
    names_w_transits -= set(archival_transit_hand_remove)
    names_w_transits |= set(archival_transit_hand_add)
    names_w_tranits = sorted(list(names_w_transits))

    if toggle_save_transit_list:
        lat.write_observed_transit_list_file(names_w_transits)

    catutils.set_index(cat, 'hostname')
    i_w_transit = cat.loc_indices['hostname', names_w_transits]
    cat['requested_observed_transit'] = 0
    cat['requested_observed_transit'][i_w_transit] = 1


#%% flag TTRB approved targets

allowed = table.Table.read(paths.locked / 'ttrb approved.targets', format='ascii.csv', data_start=5, header_start=4)
allowed_hst_names = allowed['# Name in the Proposal']
allowed_hst_names = [n for n in allowed_hst_names if '-OFFSET' not in n]
allowed_tics = dbutils.stela_name_tbl.loc['hostname_hst', allowed_hst_names]['tic_id']
cat['in_ttrb_list'] = np.isin(cat['tic_id'], allowed_tics)


#%% make sure already-observed targets are selectable

already_observed = catutils.read_requested_targets(paths.requested / 'stela_already_observed.txt')
cat.add_index('hostname')
iloc = cat.loc_indices['hostname', already_observed]
cat['stage1'][iloc] = True
cat['decision'][iloc] = np.char.add(cat['decision'][iloc].astype('str'), ' Later reintroduced because already observed.')

#%% verify ISR table

"""check nobody used a hostname in the table that doesn't match to a target"""
unmatched = catutils.unmatched_names(ref.mdwarf_isr['Current Exo Archive Name'], merged['hostname'])
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
    latest_status_path = dbutils.pathname_max(paths.status_input, 'HST-17804-visit-status*.xml')
    status_tbl = visit_status_xml_parser.load_visit_status_xml_as_table(latest_status_path)
    nonrefundable_status_substrings = ('Archived', 'Flight Ready', 'Scheduled')
    status_tbl['tic_id'] =  dbutils.stela_name_tbl.loc['hostname_hst', status_tbl['target']]['tic_id']
    status_tbl['counted'] = False # add a column for accounting
    status_tbl['band'] = ['lya' if lbl[0] <= 'M' else 'fuv' for lbl in status_tbl['visit']]
    status_tbl['redo'] = [s.isdigit() for s in status_tbl['visit']]
    status_tbl['locator'] = np.char.add(status_tbl['tic_id'].astype(str), status_tbl['band'])
    nonrefundable_mask = [any(ss in statusstr for ss in nonrefundable_status_substrings)
                          for statusstr in status_tbl['status'].tolist()]
    nonrefundable_mask = np.array(nonrefundable_mask) & ~status_tbl['redo']
    status_tbl['nonrefundable'] = nonrefundable_mask
    status_tbl.add_index('locator')
    nonrfndbl_visit_tbl = status_tbl[np.array(nonrefundable_mask)]
    available_planned = sum(nonrefundable_mask)
    available_free = allocated_orbits - available_planned

    # load the hand updated observing status sheet to ID lemons
    path_main_table = dbutils.pathname_max(paths.status_input, 'Observation Progress*.xlsx')
    progress_tbl = catutils.read_excel(path_main_table)
    progress_tbl['lemon'] = progress_tbl['Pass to\nStage 1b?'] == False
    progress_tbl.add_index('TIC ID')

    # load the visit label table
    labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')

"""remember there are still targets we don't want to observe in the table for tracking purposes, 
so better clean those before we start building a list
this must happen before filtering for hosts or else some planets with stage1=False may be selected"""
candidates = cat[cat['stage1'].filled(False)
                 | (cat['requested_observed_transit'].filled(0) > 0)]

# get just the hosts since we're not observing planets in stage 1
candidates = catutils.planets2hosts(candidates)

# sort those with existing transit observations to the top to be sure they are considered for an fuv visit
# before all orbits are allocated
candidates['has_transit'] = candidates['requested_observed_transit'].filled(0) > 0
candidates.sort(('has_transit', 'score_host'), reverse=True)

# prep columns for which gratings will be used
catutils.set_index(cat, 'tic_id')
for status in ('stage1', 'backup'):
    for grating in ('g140m', 'g140l', 'e140m', 'g130m', 'lya', 'fuv'):
        catutils.add_filled_masked_column(cat, f'{status}_{grating}', 0, mask=True, dtype=int)

available_backup = backup_orbits
if backup_orbits:
    catutils.add_filled_masked_column(cat, 'stage1_backup', False, dtype=bool)

cat['stage1_orbit_total'] = table.MaskedColumn(len(cat), mask=True, dtype=int)


#%% add any new potential targets to STELa name table

if toggle_save_new_stela_names:
    nametbl = ref.stela_names.copy()
    not_in_namtbl_mask = ~np.in1d(candidates['tic_id'], nametbl['tic_id'])
    if np.any(not_in_namtbl_mask):
        newtargrows = candidates[not_in_namtbl_mask]
        newtargrows = catutils.planets2hosts(newtargrows)
        for targrow in newtargrows:
            hostname = targrow['hostname']
            hstname, = dbutils.target_names_stela2hst([hostname])
            filename, = dbutils.target_names_stela2file([hostname])
            namerow = dict(tic_id=targrow['tic_id'],
                           hostname=hostname,
                           hostname_hst=hstname,
                           hostname_file=filename)
            assert set(namerow.keys()) == set(nametbl.colnames)
            nametbl.add_row(namerow)
        nametbl.write(paths.stela_name_tbl, format='ascii.csv', overwrite=True)

        # need to reload some things so the name table used by helper functions gets updated
        from importlib import reload
        reload(ref)
        reload(dbutils)


#%% BUILD TARGET LIST

"""allocate orbits target by target working down the ranks"""
selected_tic_ids = set()
observations_to_verify = []
i = 0
stela_orbit_count = 0
bands = ('lya', 'fuv')
grating_map = {
    'lya': {True:'g140m', False:'e140m'},
    'fuv': {True:'g140l', False:'g130m'}
}

# NOTE systems with existing observed transit sorted to top earlier to ensure they get considered first for
# an fuv visit to homogenize the STELa + archival transit survey

# now go through the ranked list
while (available_planned > 0) or (available_free > 0) or (available_backup > 0):

    if i >= len(candidates):
        raise ValueError('Reached the end of the candidate list without allocating all orbits. '
                         'Running the next cell will likely reveal what happened based on discrepancies.')

    print(f"\rTarget: {i+1}/{len(candidates)}", end="", flush=True)

    # find the indices of planets associated with the target host and whether any are already in stage 1
    # fom an earlier iteration of this loop
    host = candidates[i]

    if not host['in_ttrb_list']:
        i += 1
        continue

    name = dbutils.target_names_tic2stela(host['tic_id'])
    tic_id_ = host['tic_id']
    j = cat.loc_indices[tic_id_]
    j = np.atleast_1d(j)

    # use this mask to only update decision for planets still under consideration
    in_stage1 = (cat['stage1'][j].filled(False) == True)
    up_for_selection = j[in_stage1]
    up_for_reconsideration = j[~in_stage1]

    # determine whether external observations satisfy
    valid_external_obs = {
        band:host[f'external_{band}_status'] in ['planned', 'valid', 'unverified'] for band in bands
    }
    if all(valid_external_obs.values()):
        cat['stage1'][up_for_selection] = False
        cat['decision'][up_for_selection] = 'Rejected because observations already exist.'
        i += 1
        continue

    # if target would have been included but for having been already observed by external program,
    # mark so those observations can be verified later
    if available_free > 0:
        for band in ('lya', 'fuv'):
            if host[f'external_{band}_status'] in ['unverified', 'tentative']:
                observations_to_verify.append(f'{name} {band}')

    # allocate orbits
    tallied_obs = 0
    selected_either_band = False
    backup_either_band = False
    lemon = tic_id_ in progress_tbl['TIC ID'] and progress_tbl.loc[tic_id_]['lemon']
    for band in ('lya', 'fuv'):
        isr_pass = apt.does_mdwarf_pass_isr(name, band) # get flare isr determination

        loc = f'{tic_id_}{band}'
        nonrefundable = loc in nonrfndbl_visit_tbl['locator']

        tally = False
        backup = None
        grating = grating_map[band][isr_pass]
        if nonrefundable:
            tally, backup = True, False
            available_planned -= 1
            nonrfndbl_visit_tbl.loc['locator', loc]['counted'] = True
        elif not valid_external_obs[band]:
            if band == 'fuv' and lemon:
                pass
            else:
                if available_free > 0:
                    tally, backup = True, False
                    available_free -= 1
                if available_backup > 0:
                    tally, backup = True, True

        if tally:
            tallied_obs += 1
            category = 'backup' if backup else 'stage1'
            cat[f'{category}_{grating}'][j] = 1
            cat[f'{category}_{band}'][j] = 1
            if not backup:
                if loc in status_tbl['locator']:
                    _iloc = status_tbl.loc_indices['locator', loc]
                    status_tbl['counted'][_iloc] = True
                selected_either_band = True
                cat['stage1'][j] = True
                selected_tic_ids |= {tic_id_}
            else:
                backup_either_band = True
                cat['stage1_backup'][j] = True

    # update main catalog
    stela_orbit_count += tallied_obs
    cat['stage1_orbit_total'][j] = stela_orbit_count
    if selected_either_band:
        # mark host and best planet as selected
        cat['decision'][up_for_selection] = 'Host selected.'

        # mark planets in same system that were previously rejected, if any, as now included
        prefix = 'Host ultimately selected. Planet previously '
        readd_decisions = [prefix + decision for decision in cat['decision'][up_for_reconsideration]]
        cat['decision'][up_for_reconsideration] = readd_decisions
    elif backup_either_band:
        k = j[up_for_selection]
        cat['decision'][k] = 'Reserved as backup target.'

    i += 1

print('\n\n')


#%% check for discrepancies

if toggle_mid_cycle_update:
    nr_tics = nonrfndbl_visit_tbl['tic_id']
    nr_lemons = progress_tbl.loc[nr_tics]['lemon'] & (nonrfndbl_visit_tbl['band'] == 'fuv')
    assert sum(nr_lemons) == 0 # no lemons should be nonrefundable bc we wait to release them

    uncounted = nonrfndbl_visit_tbl['counted'] == False
    if np.any(uncounted):
        print("Some nonrefundable visits weren't tallied. Inestigate.")
        print('')
        nonrfndbl_visit_tbl[uncounted].pprint(-1, -1)



#%% flag selections

cut_mask = np.ones(len(cat), bool)
selected_tic_ids = list(selected_tic_ids)
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


#%% PRUNE to final selections

cat.sort('stage1_rank')
selected = cat[cat['stage1'].filled(False)]
selected_hosts = catutils.planets2hosts(selected)
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
n_orbits_check = sum(targets[f'stage1_{grating}'][guaranteed].sum() for grating in 'g140m g140l e140m g130m'.split())
assert n_orbits_check in [allocated_orbits, allocated_orbits + 1]
assert n_orbits_check == selected['stage1_orbit_total'].max()

# start compact table for hand input to APT
apt_info = targets[['tic_id', 'stage1_rank']]
apt_info['name'] = dbutils.target_names_tic2stela(targets['tic_id'])
apt_info = apt_info[['name', 'tic_id', 'stage1_rank']]
apt_info['GM'] = targets['stage1_g140m']
apt_info['GL'] = targets['stage1_g140l']
apt_info['EM'] = targets['stage1_e140m']
apt_info['CM'] = targets['stage1_g130m']
apt_info['lya'] = targets['stage1_lya']
apt_info['fuv'] = targets['stage1_fuv']

no_simbad_match = targets['simbad_id'].mask
n_no_simbad = sum(no_simbad_match)
if n_no_simbad > 0:
    no_simbad_names = targets['hostname'][no_simbad_match].tolist()
    print(f'Warning: these targets will not match to an object in SIMBAD {no_simbad_names}.')

simbad = apt.get_simbad_info(targets['simbad_id'].filled('?'))

apt.fill_spectral_types(targets, simbad)
apt.fill_optical_magnitudes(targets, simbad)
apt_target_table = apt.make_apt_target_table(targets, simbad)
past_targets = table.Table.read(paths.locked / 'ttrb approved.targets', format='ascii.csv', data_start=5, header_start=4)

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


#%% M dwarfs not in ISR table (might want to check that they aren't just name changes)

"""You will need to rerun the selection if any of these must be observed with E140M, since that will make more orbits
available."""

if toggle_save_outputs:
    Ms = apt_info['Mdwarf']
    Mnames = apt_info['name'][Ms]
    Mtics = apt_info['tic_id'][Ms]
    catutils.set_index(selected_hosts, 'tic_id')
    catMnames = selected_hosts.loc[Mtics]['hostname']
    not_in_isr = ~np.in1d(catMnames, ref.mdwarf_isr['Current Exo Archive Name'])
    Ms_to_add = sorted(zip(Mnames[not_in_isr], catMnames[not_in_isr]))
    np.savetxt(paths.selection_outputs / 'Mdwarfs_to_add_to_ISR_table.txt',
               Ms_to_add, fmt='%s', delimiter=',', header='stela_name,exocat_name')


#%% visit labels for APT

labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')

# be sure all targets have (or had, prior to cuts) a match
# if not, fix it
unmatched = catutils.unmatched_names(labeltbl['tic_id'], merged['tic_id'])
if len(unmatched) > 0:
    raise KeyError(f'These targets in the visit label table have no match: {unmatched.tolist()}')

new_batch_no = labeltbl['batch'].max() + 1
existing_base_labels = list(map(apt.VisitLabel, labeltbl['base'].tolist()))
last_base_label = max(existing_base_labels)

targets_selected = catutils.planets2hosts(selected)
for row in targets_selected:
    tic_id = row['tic_id']
    if tic_id not in labeltbl['tic_id']:
        name = dbutils.target_names_tic2stela(tic_id)
        last_base_label, pair = last_base_label.next_pair()
        new_row = [name, str(last_base_label), str(pair), new_batch_no, '', '', tic_id]
        labeltbl.add_row(new_row)

catutils.set_index(labeltbl, 'target')
in_labeltbl = np.in1d(apt_info['tic_id'], labeltbl['tic_id'])
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
    _labeltbl_apt_names = dbutils.target_names_tic2stela(labeltbl['tic_id'])
    in_targets = np.isin(_labeltbl_apt_names, apt_target_table['Target Name'])
    names_in_targets = _labeltbl_apt_names[in_targets]
    apt_visit_info = apt_target_table.loc[names_in_targets]
    apt_visit_info.write(paths.selection_outputs / 'info_all_targets.csv', overwrite=True)


#%% save the label map

if toggle_save_visit_labels:
    labeltbl.write(paths.locked / 'target_visit_labels.ecsv', overwrite=True)


#%% print list of visits to remove from apt (lemons or don't make rank)

redo_mask = np.array([s.isdigit() for s in status_tbl['visit']])
remove_from_apt_mask = ~status_tbl['counted'] & ~redo_mask
status_tbl[['target', 'visit', 'status']][remove_from_apt_mask].pprint(-1,-1)

# print info on dropped targets
removed_tics = np.unique(status_tbl['tic_id'][remove_from_apt_mask])
catutils.set_index(cat, 'tic_id')
_temp = cat.loc[removed_tics]
_temp = catutils.planets2hosts(_temp)
_temp = table.join(_temp, apt_info, keys='tic_id')

_viewcols = 'hostname stage1_rank_1 stage1_g140m stage1_g140l stage1_e140m stage1_g130m lbl1 lbl2'.split()
_temp[_viewcols].pprint(-1,-1)



#%% print passes to release

progtbl_pass = progress_tbl['Pass to\nStage 1b?'] == True
tics_to_pass = progress_tbl['TIC ID'][progtbl_pass]
release_mask = (
    (status_tbl['band'] == 'fuv') & # fuv visit
    ~status_tbl['redo'] & # not a redo
    status_tbl['next'].mask & # no plan date yet
    status_tbl['obsdate'].mask & # not yet observed
    ~remove_from_apt_mask & # not set to be removed from apt
    np.isin(status_tbl['tic_id'], tics_to_pass) # has been deemed passing in the progress table
)
statlabels_to_pass = status_tbl['visit'][release_mask]
print(' '.join(statlabels_to_pass.tolist()))


#%% print information on visits not currently in the APT to add

# if your APT is out of sync with online visit status, update path below and use this block
pro_export_path = '/Users/parke/Google Drive/Research/STELa/phase IIs/cycle 32/apt/STELa cycle 32 stage 1 submission 20 diagnostic.pro'
lbls_in_apt = []
with open(pro_export_path) as f:
    lines = f.readlines()
for l in lines:
    result = re.findall(r'Visit: (..)', l)
    if result:
        lbls_in_apt.extend(result)

# else use this
# lbls_in_apt = status_tbl['visit']

# add target numbers to help me find them
allowed_names = dbutils.stela_name_tbl.loc['tic_id', allowed_tics]['hostname']
allowed_names = allowed_names.tolist()
target_no_apt = [allowed_names.index(n)+1 for n in apt_info['name']]
apt_info['no'] = target_no_apt
# _colorder = ['no'] + apt_info.colnames[:-1]
# apt_info = apt_info[_colorder]

add_lya = (
    ~np.isin(apt_info['lbl1'].filled(''), lbls_in_apt) & # label not already in apt
    (apt_info['lya'].filled(0) > 0)
)
add_fuv = (
    ~np.isin(apt_info['lbl2'].filled(''), lbls_in_apt) & # label not already in apt
    (apt_info['fuv'].filled(0) > 0)
)
to_add_apt = add_lya | add_fuv
apt_print = apt_info.copy()
apt_print['lbl1'].mask[~add_lya] = True
apt_print['lbl2'].mask[~add_fuv] = True
apt_print[to_add_apt].pprint(-1,-1)
print()
print(f"{sum(to_add_apt)} new visits")

#%% print info on fuv visits we'd like to move to COS

assert sum(selected_hosts['stage1_g130m'].filled(0)) == sum(apt_info['CM'].filled(0))
cos_mask = apt_info['CM'].filled(0) > 0
cos_lbls = apt_info[cos_mask]['lbl2']
print(' '.join(cos_lbls))


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
spt_questionable = (simbad['sp_qual'].filled('F') >= 'C') | (simbad['sp_qual'].filled('') == '')
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

