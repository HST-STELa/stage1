import re

from the_usuals import *
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize

import paths

from lya_prediction_tools import lya
from stage1_processing import preloads

hostcat = preloads.hosts.copy()
name_tbl = preloads.stela_names.copy()
hostcat.add_index('tic_id')

target = 'toi-1434'
targfolder = paths.target_data(target)
tic_id = name_tbl.loc['hostname_file', target]['tic_id']
props = hostcat.loc[tic_id]

rv = props['st_radv']

date = '2025-08-11'

#%%

proposed_planets_str = """TOI-1759	b
TOI-2194	b
L 98-59	c
L 98-59	d
TOI-1231	b
TOI-1774	b
HD 63935	b
HD 63935	c
TOI-1434	b
HD 42813	b
TOI-421	c"""
_temp = re.split('\t|\n', proposed_planets_str)
prop_hosts, prop_letters = _temp[::2], _temp[1::2]

#%% lya reconstruction plot

lya_file, = targfolder.rglob('*lya-recon.csv')
lya_tbl = table.Table.read(lya_file)
w = lya_tbl['wave_lya']
v = lya.w2v(w) - rv

plt.figure()
cases = 'low_2sig low_1sig median high_1sig high_2sig'.split()
for case in cases:
    plt.plot(v, lya_tbl[f'lya_model_{case}'], color='0.5')
    plt.plot(v, lya_tbl[f'lya_intrinsic_{case}'], color='0.5', lw=0.5)
plt.step(v, lya_tbl['flux_lya'], where='mid')

plt.xlabel('Velocity in System Frame (km s-1)')
plt.ylabel('Flux Density (erg s-1 cm-2 Å-1)')
plt.xlim(-210, 210)
plt.ylim(-3e-14, 5e-13)
plt.title(target.upper())

plt.tight_layout()
plt.savefig(paths.scratch / f'{target}.{date}.plot-lya-recon.png', dpi=300)


#%% outflow sim plot

import h5py

sim_file, = targfolder.rglob('*outflow-tail*.h5')
with h5py.File(sim_file) as f:
    t = f['tgrid'][:]
    w = f['wavgrid'][:] * 1e8
    tr_raw = f['intensity'][:]
v = lya.w2v(w) - rv

transmaxs = np.max(tr_raw, axis=2)
offsets = 1 - transmaxs
tr = tr_raw + offsets[:, :, None]
tr = tr[0]

fig, axs = plt.subplots(1,2, figsize=[10,4])

ax = axs[0]

tr_plt = tr[::2, :]
n = tr_plt.shape[0]
t_plt = t[::2]

norm = Normalize(vmin=t_plt[0], vmax=t_plt[-1])
cmap = cm.viridis

# Plot each line with a different color based on time step
for t_plt_i, tr_plt_i in zip(t_plt, tr_plt):
    ax.plot(v, tr_plt_i, color=cmap(norm(t_plt_i)), linewidth=1)

# Add a colorbar to show the gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for the ScalarMappable
plt.colorbar(sm, ax=ax, label='Time from Mid-Transit (h)')

ax.set_xlabel('Velocity in System Frame (km s-1)')
ax.set_ylabel('Transit Transmission')

ax = axs[1]

v_plt = np.arange(-150,150,25)
interp_v = lambda a: np.interp(v_plt, v, a)
tr_plt = np.apply_along_axis(interp_v, -1, tr).T

norm = Normalize(vmin=v_plt[0], vmax=v_plt[-1])

for v_plt_i, tr_plt_i in zip(v_plt, tr_plt):
    ax.plot(t, tr_plt_i, color=cmap(norm(v_plt_i)), linewidth=1, label=f'{v_plt_i:.0f}')

ax.set_xlabel('Time from Mid-Transit (h)')

# Add a colorbar to show the gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for the ScalarMappable
plt.colorbar(sm, ax=ax, label='Velocity (km s-1)')

plt.title(target.upper())
plt.tight_layout()
plt.savefig(paths.scratch / f'{target}.{date}.transit-sims.png', dpi=300)


#%% x-ray recon plot

x_file, = targfolder.rglob('*xray-recon.fits')
x_tbl = table.Table.read(x_file)

plt.figure(figsize=(8,4))
plt.step(x_tbl['wavelength'], x_tbl['flux'], where='mid')
plt.yscale('log')
plt.ylim(1e-17, 1e-12)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux Density (erg s-1 cm-2 Å-1)')

plt.title(target.upper())
plt.tight_layout()
plt.savefig(paths.scratch / f'{target}.{date}.xray-recon.png', dpi=300)


#%% eval batch 1 params

import matplotlib.patheffects as path_effects

eval_filename = 'stage2_evalution_metrics.ecsv'
eval_path = paths.catalogs / eval_filename
eval = table.Table.read(eval_path)
eval.add_index('hostname')
eval.add_index('planet')

frac = eval['frac models w snr > 3']
mask = frac > 0
with np.errstate(divide='ignore'):
    logfrac = np.log10(frac)
zeros = frac == 0
logfrac = np.clip(logfrac, np.min(logfrac[~zeros])-1, np.inf)

ionkey = 'H ionztn time (h)'
rkey = 'planet radius (Re)'
Tkey = 'stellar eff temp (K)'

ion = eval[ionkey]
r = eval[rkey]
T = eval[Tkey]

ionlbl = 'Hydrogen Ionization Time (h)'
Rlbl = 'Planet Radius (Rearth)'
Tlbl = 'Stellar Effective Temperature (K)'

def label_planets(xkey, ykey):
    for host, letter in zip(prop_hosts, prop_letters):
        row = eval.loc['hostname', host]
        if hasattr(row, 'loc'):
            row = row.loc['planet', letter]
        name = f'  {row['hostname']} {row['planet']}'
        x = row[xkey]
        y = row[ykey]
        lbl = plt.annotate(name, xy=(x,y), ha='left', va='bottom', fontsize='x-small')
        lbl.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal()
        ])

#%%
plt.figure()
plt.scatter(ion, r, c=logfrac)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(ionlbl)
plt.yticks((0.6, 1, 2, 3, 5, 10), '0.6 1 2 3 5 10'.split())
plt.ylabel(Rlbl)
plt.colorbar(label='Log10 Frac Models > 3σ Transit')
label_planets(ionkey, rkey)

plt.tight_layout()
plt.savefig(paths.scratch / f'batch1.{date}.rad-vs-ion.png', dpi=300)

#%%

plt.figure()
plt.scatter(T, r, c=logfrac)
plt.yscale('log')
plt.xlabel(Tlbl)
plt.yticks((0.6, 1, 2, 3, 5, 10), '0.6 1 2 3 5 10'.split())
plt.ylabel(Rlbl)
plt.colorbar(label='Log10 Frac Models > 3σ Transit')
label_planets(Tkey, rkey)

plt.tight_layout()
plt.savefig(paths.scratch / f'batch1.{date}.teff-vs-ion.png', dpi=300)