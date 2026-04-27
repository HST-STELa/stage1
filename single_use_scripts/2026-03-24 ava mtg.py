
import numpy as np

from processing import transit_evaluation_utilities as tutils

import paths
import database_utilities as dbutils

#%%

path = dbutils.one_glob(paths.target_data('gj436'), '**/*model.detection-sigmas.ecsv')
snrs = tutils.DetectabilityDatabase.from_file(path)

#%% viewcols

viewcols = [
    'eta',
    'mdot_star',
    'Tion',
    'mass',
    'aperture',
    'transit sigma',
]

best_offset = snrs.meta['best time offset']
snrs_filtered = snrs.filtered({'time offset': best_offset, 'grating':'g140m'})
snrs_filtered = snrs_filtered.clean_duplicates()

#%%

viewcols = [
    'eta',
    'mdot_star',
    'Tion',
    'mass',
    'aperture',
    'transit sigma',
]

np.unique(snrs_filtered['aperture'])
snrs_filtered[viewcols].pprint(None,-1)