
import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from processing import preloads
from processing import target_lists


#%%

viewcols = 'hostname st_teff sy_gaiamag'.split()


#%% batch 1

targets = (
"""
gj143
hd207897
hd95338
k2-136
toi-1434
toi-2134
toi-2285
toi-2443
toi-260
toi-431
wasp-107
toi-2459
"""
)
targets = [t.strip() for t in targets.split('\n') if t.strip() != '']


#%% batch 2


#%% print info


tics = preloads.stela_names.loc['hostname_file', targets]['tic_id']
tcat = preloads.hosts.loc[tics]
tcat[viewcols].pprint()

#%% ETC calc notes

"""I found time to hit SNR 40 (roughly the min DN count recommended by the handbook) and the time
 to saturation and took the geometric mean.
 
 use 52x0.05, if 0.1 s is within factor 3 of saturation use 31x0.05ndb 
 
 remember gaia uses the vega system for mags
 use pickles models cuz they're actual data
 
 after making table, go back and check in reverse by entering exposure time and looking at predicted snr & saturation
 use the check ETC runs for the numbers 
 """

etc_output = ( # target aperture T40 Tsat etcnumber
"""
gj143 ndb 0.0204 0.86 2330144
hd207897 ndb 0.0301 1.25 2330145
hd95338 ndb 0.0380 1.57 2330146
k2-136 clear 0.0362 1.60 2330148
toi-1434 ndb 0.0448 1.84 2330160 
toi-2134 ndb 0.0438 1.84 2330149
toi-2285 clear 0.1534 6.76 2330161
toi-2443 ndb 0.0757 3.31 2330174
toi-260 clear 0.0090 0.41 2330154
toi-431 ndb 0.0569 2.39 2330156
wasp-107 clear 0.0486 2.14 2330157
toi-2459 clear 0.0192 0.85 2330158
"""
)
rows = [t.strip().split() for t in etc_output.split('\n') if t.strip() != '']

exptime_tbl = table.Table(rows=rows, names='target ap T40 Tsat etcno'.split())
exptime_tbl['Tsat'] = exptime_tbl['Tsat'].astype(float)
exptime_tbl['T40'] = exptime_tbl['T40'].astype(float)
exptime_tbl['Tbest'] = np.sqrt(exptime_tbl['T40']*exptime_tbl['Tsat'])
exptime_tbl['Tbest'].format='.1f'
joined = table.hstack((tcat[viewcols], exptime_tbl))

joined.pprint(-1,-1)

#%% notes on entering in APT

"""
!SAVE FREQUENTLY!
copy patterns from a 17997 or 18226
copy ACQ
make acq/peak with values from table above
copy to after each wave in base and trans visits
make pattern, duplicate so 2 in base and 5 in transit
in spreadsheet set pattern type
move patterns below each peak
mave sci exposures into patterns
in spreadsheet check target, ap, exptime all correct
check for no phase constraints
set exptime of science visits to 400 for first in each visit, 500 for others
orbit planner clear durations
auto adjust
check that orbits are packed correctly
visit planner
submit
"""