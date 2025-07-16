from datetime import datetime

from astropy import table
import numpy as np
from matplotlib import pyplot as plt

import catalog_utilities as catutils
import paths

from stage1_processing import preloads


#%% table of target parameters combined with observation table values

progress_table = preloads.progress_table.copy()

obscols = 'Rank,Target,Peak\nLya Flux,Integrated\nLya Flux,(O-C)/sigma,Pass to\nStage 1b?'.split(',')
colmap = {'Rank': 'rank',
          'Target': 'target',
          'Peak\nLya Flux': 'peak lya flux',
          'Integrated\nLya Flux': 'integrated lya flux',
          '(O-C)/sigma': 'lya (O-C)/sigma',
          'Pass to\nStage 1b?': 'pass'}
oldcols, newcols = list(zip(*colmap.items()))

bigtable = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__add-flags-scores.ecsv')
bigtable = catutils.planets2hosts(bigtable)
hostcols = 'hostname ra dec sy_dist st_radv st_teff st_mass st_rad st_rotp st_age st_agelim'.split()

_temp = bigtable.copy()
_temp['hostname'] = _temp['hostname'].astype(str)

diagnostic_table = table.join(_temp[hostcols], progress_table[obscols],
                              keys_left='hostname', keys_right='Target', join_type='right')
assert len(diagnostic_table) == len(progress_table)
diagnostic_table.rename_columns(oldcols, newcols)
diagnostic_table.remove_column('target')

numerical_cols = 'peak lya flux,integrated lya flux,lya (O-C)/sigma'.split(',')
for name in numerical_cols:
    empty = diagnostic_table[name] == ''
    diagnostic_table[name][empty] = 'nan'
    diagnostic_table[name] = table.MaskedColumn(diagnostic_table[name], mask=empty, fill_value=np.nan, dtype=float)


today = datetime.today().isoformat()[:10]
diagnostic_table.write(paths.status_input / f'target properties and lya fluxes {today}.ecsv')

#%% plot coordinates of observed targets

observed = ~diagnostic_table['integrated lya flux'].mask
ot = diagnostic_table[observed]

# plot positions on sky
from astropy.coordinates import SkyCoord
import astropy.units as u

# SkyCoord object
c = SkyCoord(ra=ot['ra'], dec=ot['dec'], frame='icrs')

# Convert to radians for plotting
ra_rad = c.ra.wrap_at(180 * u.deg).radian  # wrap RA to [-180, +180]
dec_rad = c.dec.radian

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True)
ax.scatter(ra_rad, dec_rad, marker='o', color='C0')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')


#%% plot other params vs Lya fluxes

observed = ~diagnostic_table['integrated lya flux'].mask
ot = diagnostic_table[observed]

oc = ot['lya (O-C)/sigma']

names = 'sy_dist st_radv st_mass st_rotp'.split()
for name in names:
    plt.figure()
    plt.plot(ot[name], ot['lya (O-C)/sigma'], 'o')
    plt.xlabel(name)
    plt.ylabel('Lya Flux (O-C)/sigma')


plt.figure()
nolim = ot['st_agelim'].filled(0) == 0
lolim = ot['st_agelim'].filled(0) == -1
uplim = ot['st_agelim'].filled(0) == 1
plt.errorbar(ot['st_age'][nolim], ot['lya (O-C)/sigma'][nolim], fmt='oC0')
plt.errorbar(ot['st_age'][lolim], ot['lya (O-C)/sigma'][lolim], fmt='oC0', xerr=0.2, xlolims=True)
plt.errorbar(ot['st_age'][uplim], ot['lya (O-C)/sigma'][uplim], fmt='oC0', xerr=0.2, xuplims=True)
plt.xlabel('age')
plt.ylabel('Lya Flux (O-C)/sigma')