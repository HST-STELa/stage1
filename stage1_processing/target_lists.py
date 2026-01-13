from datetime import datetime
import re

import numpy as np
from astropy.table import Table, Row, vstack

import database_utilities as dbutils
import paths

from stage1_processing import preloads


def eval_no(n):
    prog = preloads.progress_table
    mask = prog["Stage 2 Eval\nBatch"] == n
    tics =  prog['TIC ID'][mask]
    selected = preloads.stela_names.loc['tic_id', tics]
    return selected['hostname_file'].tolist()


def observed_since(isot_date_string, type='lya_or_fuv'):
    date = datetime.fromisoformat(isot_date_string)
    status = preloads.visit_status
    fillvalue = datetime.fromisoformat('0001-01-01')
    mask = (status['obsdate'].filled(fillvalue) >= date) & (status['status'] == 'Archived')
    if type == 'lya':
        mask &= status['visit'] <= 'MZ'
    elif type == 'fuv':
        mask &= status['visit'] >= 'N0'
    else:
        if type != 'lya_or_fuv':
            raise ValueError

    hst_names = status['target'][mask]
    selected = preloads.stela_names.loc['hostname_hst', hst_names]
    return selected['hostname_file'].tolist()


def everything_in_progress_table():
    tics = preloads.progress_table['TIC ID']
    selected = preloads.stela_names.loc['tic_id', tics]
    return selected['hostname_file'].tolist()


def new_fuv_search():
    prog = preloads.progress_table
    mask = ((prog["External\nFUV Good?"] != 'yes') &
            (prog["FUV Data\nGood?"] != 'yes'))
    tics = prog['TIC ID'][mask]
    names = preloads.stela_names.loc['tic_id', tics]['hostname_file']
    return names.tolist()


def selected_for_transit(batch_no):
    prog = preloads.progress_table
    planets = preloads.planets.copy()
    planets.add_index('tic_id')
    planets['pl_id'] = dbutils.planet_suffixes(planets)
    planets.add_index('pl_id')
    mask = (prog["Stage 2 Eval\nBatch"] == batch_no) & (prog["Stage 2 Transits\nAssigned"] != '')
    planet_rows = []
    for row in prog[mask]:
        tic = row['TIC ID']
        hostplanets = planets.loc['tic_id', tic]
        planet_ids = row["Stage 2 Transits\nAssigned"]
        planet_ids = re.split(', *', planet_ids)
        if isinstance(hostplanets, Row):
            assert planet_ids[0] == hostplanets['pl_id'], 'Planet id mismatch'
            planet_rows.append(hostplanets)
        else:
            selected_planets = hostplanets.loc['pl_id', planet_ids]
            if isinstance(selected_planets, Table):
                planet_rows.extend(selected_planets)
            else:
                planet_rows.append(selected_planets)
    # selected_planets = Table(rows=planet_rows)
    selected_planets = vstack(planet_rows)
    return selected_planets


def new_data(last_n=1):
    files = list(paths.new_data_lists.rglob('*new_data*.txt'))
    files = sorted(files)
    names_lists = [np.loadtxt(f, dtype='O') for f in files]
    names_lists = list(map(np.atleast_1d, names_lists))
    names = np.hstack(names_lists[-last_n:])
    names = list(set(names))
    return names