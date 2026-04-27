from typing import Literal

from astropy import table
import numpy as np
import matplotlib as mpl

import paths
import catalog_utilities as catutils

from processing import transit_evaluation_utilities as tutils
from processing import preloads


#%% settings

target = 'gj436'

sigma_threshold = 3
min_samples = 5**4 # used as a check later to ensure all grid pts of Ethan's sims were sampled

mpl.use('qt5agg') # plots are shown
np.seterr(divide='raise', over='raise', invalid='raise') # whether to raise arithmetic warnings as errors


#%% planet catalog

with catutils.catch_QTable_unit_warnings():
    planet_catalog = preloads.planets.copy()
    planet_catalog = table.QTable(planet_catalog)
    host_catalog = catutils.planets2hosts(planet_catalog)
    planet_catalog.add_index('tic_id')
    host_catalog.add_index('tic_id')


#%% a few loose closures


def path_snrs(planet, host, tst_type: Literal['model', 'flat']):
    filenamer = tutils.FileNamer(tst_type, planet, host)
    return filenamer.snr_tbl_full

def load_snr_db(planet, host, tst_type: Literal['model', 'flat']):
    path = path_snrs(planet, host, tst_type)
    return tutils.DetectabilityDatabase.from_file(path)

def load_best_snrs(planet, host, tst_type: Literal['model', 'flat']):
    snrs = load_snr_db(planet, host, tst_type)
    best_snrs = snrs.filter_obs_config(aperture='best', offset='best safe')
    return best_snrs


#%% construct host and planet

host = tutils.Host(target, host_catalog, planet_catalog)
planet = host.planets[0]
snr_db = load_snr_db(planet, host, 'model')
best_snrs = load_best_snrs(planet, host, 'model')

off_snrs = best_snrs.trim_to_offset_sampling()


#%% smoke test of group and select function

"""the idea is to test the function that I will use to reduce these tables to a best offset for the 
snr mesh plots"""