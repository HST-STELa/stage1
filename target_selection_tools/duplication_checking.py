import re
from functools import reduce
import warnings

import numpy as np
from astropy import table
from astropy import coordinates as coord
from astropy import units as u
from tqdm import tqdm

import database_utilities as db
from target_selection_tools import query

match_dist_default = 3*u.arcmin


def read_hst_observation_catalog(hst_obs_file):
    obs = table.Table.read(hst_obs_file, format='ascii.fixed_width_two_line', position_line=18, header_start=17)
    return obs


def filter_hst_observations(hst_obs_file):
    obs = read_hst_observation_catalog(hst_obs_file)

    # some basic cuts to reduce the table size
    moving_mask = np.zeros(len(obs), bool)
    moving_objects = 'venus mars jupiter saturn neptune uranus pluto comet asteroid'.split()
    for name in moving_objects:
        mask = np.char.count(obs['targname'], name.upper()) > 0
        moving_mask = moving_mask | mask
    instrument_mask = ((np.char.count(obs['config'], 'STIS') > 0)
                       | (np.char.count(obs['config'], 'COS') > 0))
    keep = (instrument_mask  # avoid the temptation to combine with the above! &s and |s don't mix well in Python
            & (obs['wave'] < 2000)
            & (np.char.count(obs['mode'], 'ACQ') == 0)
            & (obs['time'] > 0)
            & ~moving_mask)
    obs = obs[keep]

    return obs


def read_abstracts(abstracts_file):
    with open(abstracts_file) as f:
        txt = f.read()
    abstracts = txt.split('------------------------------------------------------------------------------')
    abstracts = abstracts[:-1]

    abdic, fails = {}, []
    for a in abstracts:
        result = re.search(r'ID: +(\d+)\n', a)
        if result is None:
            fails.append(a)
            continue
        id, = result.groups(0)
        id = int(id)
        abdic[id] = a

    return abdic, fails


def hst_observation_coordinates(hst_observations):
    obs = hst_observations
    obs['ra'] = coord.Angle(obs['ra'], unit=u.hourangle)
    obs['dec'] = coord.Angle(obs['dec'], unit=u.deg)
    obs_coords = coord.SkyCoord(obs['ra'], obs['dec'])
    return obs_coords


def hst_observation_matches(target, hst_obs_coords, maxdist=match_dist_default):
    targcoord = coord.SkyCoord(target['ra'] * u.deg, target['dec'] * u.deg)
    dist = hst_obs_coords.separation(targcoord)
    matches = dist < maxdist
    return matches


def hst_observation_xmatch(catalog, hst_obs_coords, maxdist=match_dist_default):
    cat_coords = coord.SkyCoord(catalog['ra'], catalog['dec'])
    i_cat, i_hst, _, _ = hst_obs_coords.search_around_sky(cat_coords, maxdist)
    return i_cat, i_hst


def identify_lya_observations(hst_observations):
    inst, spec, wave = hst_observations['config'], hst_observations['spec'], hst_observations['wave']
    e140m_lya = (np.char.count(spec, 'E140M') > 0) & (wave < 2000)
    g140m_lya = (np.char.count(spec, 'G140M') > 0) & ((wave == 1222) | (wave == 1218))
    g130m_lya = ((np.char.count(spec, 'G130M') > 0)
                 & (((1215 > wave - 150) & (1215 < wave - 7))
                    | ((1215 > wave + 7) & (1215 < wave + 150))))

    lya_mask = e140m_lya | g140m_lya | g130m_lya
    return lya_mask, e140m_lya, g140m_lya, g130m_lya


def flag_duplicates(planet_catalog, hst_observations, match_dist=match_dist_default, review_abstracts=False):
    cat = planet_catalog
    obs = hst_observations
    obs_coords = hst_observation_coordinates(obs)
    obs['coords'] = obs_coords

    # first do a basic position match to filter the hst_observations
    i_cat, i_hst = hst_observation_xmatch(cat, obs['coords'], match_dist)
    i_hst = np.unique(i_hst)
    obs = obs[i_hst]

    # get tic ids of all observation targets
    obs.sort('targname') # must sort bc np.unique will do so and it will mess me up when puting tic_ids back in
    obs_names = obs['targname']
    unq_names, i_map2obstbl = np.unique(obs_names, return_inverse=True)
    simbad_names = db.groom_hst_names_for_simbad(unq_names)
    tic_ids = query.query_simbad_for_tic_ids(simbad_names)
    obs['tic_ids'] = tic_ids[i_map2obstbl]

    # region search for observations
    name_template = 'n_{}_{}_obs'
    configs = (('stis', 'e140m'),
               ('stis', 'g140m'),
               ('stis', 'g140l'),
               ('cos', 'g130m'),
               ('cos', 'g140l'))

    n = len(cat)
    cat['n_stis_e140m_lya_obs'] = table.MaskedColumn(0, dtype=int, length=n)
    cat['n_stis_g140m_lya_obs'] = table.MaskedColumn(0, dtype=int, length=n)
    cat['n_cos_g130m_lya_obs'] = table.MaskedColumn(0, dtype=int, length=n)
    cat['external_lya'] = table.MaskedColumn(False, dtype=bool, length=n)
    cat['external_fuv'] = table.MaskedColumn(False, dtype=bool, length=n)
    for inst, grating in configs:
        colname = name_template.format(inst, grating)
        cat[colname] = table.MaskedColumn(0, dtype=int, length=n)
    if review_abstracts:
        cat['external_lya_transit'] = table.MaskedColumn(False, dtype=bool, length=n)

    stowaway_infos = []
    for i, target in tqdm(list(enumerate(cat))):
        matches = hst_observation_matches(target, obs['coords'], match_dist)
        matchobs = obs[matches]

        if not np.ma.is_masked(target['tic_id']):
            # id_matches = matchobs['tic_ids'] == target['tic_id']
            id_matches = np.char.count(matchobs['tic_ids'], str(target['tic_id'])) > 0
            if np.any(~id_matches):
                stowaway_names = np.unique(matchobs['targname'][~id_matches]).tolist()
                stowaway_info = f'{target['hostname']}: {', '.join(stowaway_names)}'
                stowaway_infos.append(stowaway_info)
                matchobs = matchobs[id_matches]

        inst, spec, wave = matchobs['config'], matchobs['spec'], matchobs['wave']
        # we could update this so that proper motions are taken into account as well, computing positions of the target
        # at the epoch of the initial matches and then narrowing the allowed distances to help avoid contaminants

        # determine if Lya line has been observed
        lya_mask, e140m_lya, g140m_lya, g130m_lya = identify_lya_observations(matchobs)

        external_lya = np.sum(lya_mask) > 0
        cat['external_lya'][i] = external_lya
        cat['n_stis_e140m_lya_obs'][i] = np.sum(e140m_lya)
        cat['n_stis_g140m_lya_obs'][i] = np.sum(g140m_lya)
        cat['n_cos_g130m_lya_obs'][i] = np.sum(g130m_lya)

        for instname, gratingname in configs:
            colname = name_template.format(instname, gratingname)
            mask = ((np.char.count(inst, instname.upper()) > 0)
                    & (np.char.count(spec, gratingname.upper()) > 0))
            cat[colname][i] = np.sum(mask)
        cat['external_fuv'][i] = ((cat['n_cos_g130m_obs'][i] > 0)
                                  | (cat['n_cos_g140l_obs'][i] > 0)
                                  | (cat['n_stis_g140l_obs'][i] > 0)
                                  | (cat['n_stis_e140m_obs'][i] > 0))

    if stowaway_infos:
        stowaway_warning = (f'\n\nSome stars matched HST observations within {match_dist} '
                            f'that were for other targets. These were:\n\t')
        stowaway_warning += '\n\t'.join(stowaway_infos)
        warnings.warn(stowaway_warning)


def add_abstracts(planet_catalog, hst_observations, abstract_dictonary, match_dist=match_dist_default,
                  manual_review=False):
    """
    manual_review shows abstracts so the user can mark whether the Lya observations are legit and whether
    the program observed transits.
    """

    # I was about to just do the math to see if any observations align with transit, but this has two problems:
    # (1) the duplication database doesn't have data on when the observation occurred, so I'd have to pull in another databse
    # (2) won't work for planned observations not yet executed
    cat = planet_catalog
    obs = hst_observations
    abdic = abstract_dictonary
    obs_coords = hst_observation_coordinates(obs)

    cat['hst abstracts'] = table.MaskedColumn('', dtype=object, length=len(cat), fill_value='', mask=True)

    for i, target in tqdm(list(enumerate(cat))):
        matches = hst_observation_matches(target, obs_coords, match_dist)
        matchobs = obs[matches]
        lya_mask, _, _, _ = identify_lya_observations(matchobs)

        if np.sum(lya_mask) > 0:
            pids = np.unique(matchobs[lya_mask]['prop'])
            obsabs = '\n'.join([abdic[pid] for pid in pids])
            cat['hst abstracts'][i] = obsabs

            if manual_review:
                name = target['id']
                print('')
                print(name)
                print('------')
                print(obsabs)
                cols = 'config mode spec aper time prop dataset'.split()
                lyaobs = matchobs[lya_mask]
                lyaobs.sort(['prop', 'dataset'])
                lyaobs[cols].pprint(-1)
                print('\n\n\n')
                answer = input("Hit enter if lya obs are good, any letter if not.".format(name))
                if answer != '':
                    cat['external_lya'][i] = False
                    cat['n_stis_e140m_obs'][i] = 0
                    cat['n_stis_g140m_obs'][i] = 0
                    cat['n_cos_g130m_obs'][i] = 0
                answer = input("Mark transit as observed for {} (y/n)?".format(name))
                cat['external_lya_transit'][i] = answer == 'y'


def merge_verified(planet_catalog, verification_table):
    valid_verification_entries = 'none pass fail unchecked planned tentative'.split()
    for band in ('lya', 'fuv'):
        valids = [verification_table[band] == value for value in valid_verification_entries]
        valid = reduce(np.logical_or, valids[1:], valids[0])
        if not np.all(valid):
            raise ValueError('There are invalid values in the verification table that need fixing.')

    slim = planet_catalog[['tic_id']]
    slim['order'] = list(range(len(slim))) # needed bc join will order by tic_id
    joined = table.join(slim, verification_table, 'tic_id', join_type='left')
    joined.sort('order')
    planet_catalog['lya_verified'] = joined['lya'].astype(object)
    planet_catalog['fuv_verified'] = joined['fuv'].astype(object)
    for band in ('lya', 'fuv'):
        name = f'external_{band}_status'
        statuscol = table.MaskedColumn(length=len(planet_catalog), mask=True, dtype='object')

        none = ~planet_catalog[f'external_{band}'].filled(False)
        statuscol[none] = 'none'

        verification_results = planet_catalog[f'{band}_verified'].filled('')
        valid = ((verification_results == 'pass')
                 | (verification_results == 'planned'))
        statuscol[valid] = 'valid'

        fail = verification_results == 'fail'
        statuscol[fail] = 'failed'

        tentative = verification_results == 'tentative'
        statuscol[tentative] = 'tentative'

        unchecked = ~none & statuscol.mask
        statuscol[unchecked] = 'unverified'

        planet_catalog[name] = statuscol
