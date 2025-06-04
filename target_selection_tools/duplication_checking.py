import re
from functools import reduce

import numpy as np
from astropy import table
from astropy import coordinates as coord
from astropy import units as u
from tqdm import tqdm
import xml.etree.ElementTree as ET

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
    obs_coords = hst_observation_coordinates(hst_observations)

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

    for i, target in tqdm(list(enumerate(cat))):
        matches = hst_observation_matches(target, obs_coords, match_dist)
        matchobs = obs[matches]
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


def parse_visit_labels_from_xml_status(xml_path):
    # parse the xml visit status export from STScI
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate over each visit element in the visit status, find the appropriate row, and update dates
    labels = []
    for visit in root.findall('visit'):
        visit_label = visit.attrib.get('visit')
        labels.append(visit_label)

    return labels