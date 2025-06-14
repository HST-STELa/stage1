import re

import numpy as np

import catalog_utilities as catutils
import paths

#%% load list of targets with archival data, but no transit observation

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
mask = (cat['external_lya'].filled(False) & ~cat['external_fuv'])
tids = cat[mask]['tic_id']


"""Originally I was going to download everything using the API, but with only 21 stars that have data I need to download, 
I think I'll just use the online form."""

tids = np.char.add('TIC ', tids.astype(str))
print(', '.join(tids)) # can just copy and paste this into MAST form

#%% check for stowaways
pass

# need to check and make sure there are no accidental matches in the position search when the duplication matching,
# was done, so better compare target names
# I'll just put the list of targets from MAST into SIMBAD and get the TIC IDs, then compare

mast_targets = """
STKM-1-649-1
K2-233
K2-136
WASP-107
WASP-107
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
K2-25
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
HD-178085
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
2M11301450+0735180
HD-97507
HD-85426
HD-85426
L-248-27
L-248-27
L-248-27
L-248-27
L-248-27
L-248-27
L-248-27
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
BD+41-3306
HR-8832
HR-8832
HR-8832
HR-8832
"""

mast_targets = mast_targets.split('\n')[1:-1]
mast_targets = np.unique(mast_targets)
mast_simbad_id_format_map = {
    'HD-': 'HD ',
    'HR-': 'HR ',
    'L-': 'L ',
    r'STKM-(\d-\d+)(-\d+)?': r'StKM \1',
    '2M': '2MASS J',
    r'BD([+-]\d\d)[+-](\d+)': r'BD \1 \2',
}
mast_targets_groomed = []
for targ in mast_targets:
    for mast, simbad in mast_simbad_id_format_map.items():
        targ = re.sub(mast, simbad, targ)
    mast_targets_groomed.append(targ)

np.savetxt('/Users/parke/Downloads/simbad_id_query_STELa_2025-06-06.txt', mast_targets_groomed, fmt='%s')
# you will have to hand groom these some so simbad recognizes them

mast_tics = """
TIC 388804061
TIC 394172596
TIC 464646604
TIC 4897275  
TIC 23434737 
TIC 283722336
TIC 18310799 
TIC 428820090
TIC 434226736
TIC 447061717
StKM 1-649   
TIC 429302040
"""
mast_tics = mast_tics.split('\n')[1:-1]
mast_tics = [x.strip() for x in mast_tics]
for tic in mast_tics:
    if tic not in tids:
        print(tic)

"""Looks like HAT-P-20 is matching to StKM 1-649. I'ma check to see if that one got miscategorized. Yep, it did."""