from astropy import table
from tqdm import tqdm

import paths
import database_utilities as dbutils

from stage1_processing import transit_evaluation_utilities as tutils

#%% load and stack all of the snr tables

snr_files = list(paths.data_targets.rglob('*tail-model*sigmas*.ecsv'))

keep_cols = (
    'transit sigma',
    'eta',
    'mdot_star',
    'Tion',
    'mass',
    'time offset',
)

def simplify_db(db):
    tbl = db.snrs[keep_cols]
    tbl.meta = {}
    return tbl

tbls = []
best_off_tbls = []
for file in tqdm(snr_files):
    planet = file.name.split('.')[0]
    snrdb.snrs['planet'] = planet
    snrdb = tutils.DetectabilityDatabase.from_file(file)
    snrdb = snrdb.trim_to_offset_sampling()
    best_off = snrdb.filter_obs_config(offset='best').clean_duplicates()
    tbls.append(simplify_db(snrdb))
    best_off_tbls.append(simplify_db(snrdb))

bigtbl = table.vstack(tbls)
best_off_bigtbl = table.vtsack(best_off_tbls)


#%% pick subsets based on best offsets




#%% save as hdf5

date = dbutils.timestamp(date_only=True)
savepath = paths.packages / f'{date}.combined_snr_table_all_targets.h5'
bigtbl.write(savepath, path='table', serialize_meta=True, overwrite=True)


#%% make plots of median snr and