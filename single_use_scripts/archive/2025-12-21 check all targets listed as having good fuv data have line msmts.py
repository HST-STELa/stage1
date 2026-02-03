import re

import paths
import catalog_utilities as catutils

from stage1_processing import preloads

#%%

prog = preloads.progress_table
ver = catutils.load_and_mask_ecsv(paths.checked / 'verified_external_observations.csv')
ver = ver[ver['tic_id'] > 0]

prog_fuv_good = prog["External\nFUV Good?"] == 'yes'
chck_fuv_good = ver['fuv'] == 'pass'

tics = (prog['TIC ID'][prog_fuv_good].tolist()
        + ver['tic_id'][chck_fuv_good].tolist())
tics = list(set(tics))
names = preloads.stela_names.loc['tic_id', tics]['hostname_file']


#%%

def is_fuv(file):
    srchstr = r'(g140l|g130m|g160m|e140m)'
    result = re.findall(srchstr, file.name)
    return len(result) > 0

#%%

for name in names:
    folder = paths.target_hst_data(name)
    files = list(folder.glob('*_x1d*.fits'))
    fuv = list(map(is_fuv, files))
    if not any(fuv):
        print(name)