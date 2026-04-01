
import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from stage1_processing import preloads
from stage1_processing import target_lists


#%%

viewcols = 'hostname st_teff sy_gaiamag'.split()


#%% batch 1

targets = (
"""
toi-1759
l98-59
toi-1231
hd63935
hd42813
toi-421
ltt1445a
"""
)
targets = [t.strip() for t in targets.split('\n') if t.strip() != '']


#%% batch 2


#%% print info


tics = preloads.stela_names.loc['hostname_file', targets]['tic_id']
tcat = preloads.hosts.loc[tics]
tcat[viewcols].pprint()