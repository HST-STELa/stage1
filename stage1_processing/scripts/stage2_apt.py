import numpy as np

import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from target_selection_tools import apt


#%% planets

stage2_planet_names = ['GJ 143 b', 'HD 207897 b', 'HD 42813 b', 'HD 95338 b', 'K2-136 c', '1434.01', 'TOI-2076 d', 'TOI-2134 b', 'TOI-2134 c', 'TOI-2285 b', 'TOI-2443 b', 'TOI-260 b', 'TOI-431 d', 'WASP-107 b']
                       #'HD 63935 b', 'HD 63935 c', 'L 98-59 c', 'L 98-59 d', 'LTT 1445 A c', 'TOI-1231 b', 'TOI-1759 b', 'TOI-421 c'


#%% load latest table

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__cut-low-snr__add-flags-scores.ecsv')
cat.add_index('id')


#%% find planets in table

planets = cat.loc[stage2_planet_names]


#%% now get APTinfo

targets = catutils.planets2hosts(planets)
n = len(targets)

# start compact table for hand input to APT
apt_info = targets[['tic_id']]
apt_info['name'] = dbutils.target_names_tic2stela(targets['tic_id'])
apt_info = apt_info[['name', 'tic_id']]

no_simbad_match = targets['simbad_id'].mask
n_no_simbad = sum(no_simbad_match)
if n_no_simbad > 0:
    no_simbad_names = targets['hostname'][no_simbad_match].tolist()
    print(f'Warning: these targets will not match to an object in SIMBAD {no_simbad_names}.')

simbad = apt.get_simbad_info(targets['simbad_id'].filled('?'))

apt.fill_spectral_types(targets, simbad)
apt.fill_optical_magnitudes(targets, simbad)

acq_setup_table = apt.acquisition_setup(targets)
apt_info['acq'] = acq_setup_table['acq_filter']
apt_info['Tacq'] = acq_setup_table['acq_Texp']

g140m_buffers = apt.buffer_times(targets, 'g140m')
g140l_buffers = apt.buffer_times(targets, 'g140l')
e140m_buffers = apt.buffer_times(targets, 'e140m')
apt_info['GMbuff'] = g140m_buffers
apt_info['GLbuff'] = g140l_buffers
apt_info['EMbuff'] = e140m_buffers

apt_info['Mdwarf'] = np.char.count(targets['st_spectype'].filled('M').astype(str), 'M') > 0

for col in apt_info.columns:
    if apt_info[col].dtype == float:
        apt_info[col].format = '.1f'


#%% examples for printing apt info

"""print the whole table"""
apt_info.pprint(-1, -1)

# find a target or visit and print its row
apt_info.add_index('name')
print(apt_info.loc['name', 'GJ 143'])