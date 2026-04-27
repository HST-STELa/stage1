
import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from astropy import table
import numpy as np

from processing import preloads
from processing import target_lists


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

#%%
etc_output = ( # target aperture T40 Tsat etcnumber
"""
toi-1759 clear 0.0366 1.66 
l98-59 clear 0.0302 1.44
toi-1231 clear 0.0610 2.90
hd63935 ndb 0.0373 1.52 
hd42813 clear 0.0077 0.32
toi-421 clear 0.0123 0.51
ltt1445a clear 0.0182 0.87 
"""
)
rows = [t.strip().split() for t in etc_output.split('\n') if t.strip() != '']

exptime_tbl = table.Table(rows=rows, names='target ap T40 Tsat'.split())
exptime_tbl['Tsat'] = exptime_tbl['Tsat'].astype(float)
exptime_tbl['T40'] = exptime_tbl['T40'].astype(float)
exptime_tbl['Tbest'] = np.sqrt(exptime_tbl['T40']*exptime_tbl['Tsat'])
exptime_tbl['Tbest'].format='.1f'
joined = table.hstack((tcat[viewcols], exptime_tbl))

joined.pprint(-1,-1)

#%%

joined['etc'] = [
2330200,
2330201,
2330204,
2330205,
2330206,
2330207,
2330208,
]