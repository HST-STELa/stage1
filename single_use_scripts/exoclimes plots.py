from pathlib import Path

from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy import table
from matplotlib import pyplot as plt
import matplotlib as mpl

import database_utilities as dbutils
import paths
import catalog_utilities as catutils
from lya_prediction_tools import lya, ism

#%%
mpl.rcParams['font.size'] = 18

saveplots = True
vstd = (lya.wgrid_std/1215.67/u.AA - 1)*const.c.to('km s-1')

target_table = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
target_table = catutils.planets2hosts(target_table)
target_table.add_index('tic_id')

#%%
targname = 'TOI-1696'
targname_file = 'toi-1696'
stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')
tic_id = stela_name_tbl.loc['hostname', targname]['tic_id']
data_dir = Path(f'/Users/parke/Google Drive/Research/STELa/data/targets/{targname_file}')
xf, = dbutils.find_data_files('x1d', instruments='hst-stis', directory=data_dir)

#%%
h = fits.open(xf, ext=1)
data = h[1].data
spec = {}
exp = 13
for name in data.names:
    spec[name.lower()] = data[name][0]

# shift to velocity frame
rv = target_table.loc[tic_id]['st_radv'] * u.km/u.s
v = (spec['wavelength']/1215.67 - 1) * const.c - rv
v = v.to_value('km s-1')

fig, ax = plt.subplots(1,1, dpi=50, figsize=[428/72, 310/72])
plt.step(v, spec['flux']*10**exp, color='C0', where='mid')
plt.xlim(-500, 500)

# predicted lines
ylim = plt.ylim(-0.05, None)
itarget = target_table.loc_indices[tic_id]
predicted_fluxes = []
for pct in (-34, 0, +34):
    n_H = ism.ism_n_H_percentile(50 + pct)
    lya_factor = lya.lya_factor_percentile(50 - pct)
    profile, = lya.lya_at_earth_auto(target_table[[itarget]], n_H, lya_factor=lya_factor, default_rv='ism')
    plt.plot(vstd - rv, profile*10**exp, color='0.5', lw=1)

# plt.legend(('Observed', 'Predicted\n(-1σ,0,+1σ)'))

plt.xlabel('Velocity in System Frame (km s-1)')
plt.ylabel(f'Flux Density\n($\\mathrm{{10^{{{-exp}}}\\ erg\\ s^{{-1}}\\ cm^{{-2}}\\ Å^{{-1}} }}$)')
fig.subplots_adjust(0.206, 0.16, 0.995, 0.997)

pngfile = f'/Users/parke/Google Drive/Conferences/2025-07 Exoclimes VII/plots/{targname_file}.png'
fig.savefig(pngfile, dpi=150)