
from astropy.io import fits

import paths
import database_utilities as dbutils

from processing.observation_table import ObsTable
from processing import target_lists
from processing.scripts.evaluate_for_stage2 import load_best_snrs

#%%

targets_all = target_lists.everything_in_database()

targets = []
for target in targets_all:
    obstbl = ObsTable.load_from_targname(target)
    if len(obstbl) == 0:
        targets.append(target)

#%%

targets
