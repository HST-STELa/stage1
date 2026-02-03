import os

import numpy as np
from matplotlib import pyplot as plt

import paths
from lya_prediction_tools import lya
from stage1_processing import transit_evaluation_utilities as tutils

#%%

target = 'toi-2459'
scratchfolder = paths.scratch / '2025-12-16 bimodal stellar wind in snr corner plots digging'
if not scratchfolder.exists(): os.mkdir(scratchfolder)

host = tutils.Host(target)
planet, = host.planets
transit = tutils.get_transit_from_simulation(host, planet)

n_lines = 20


#%%

def plot_set(fix_param, fix_i, vary_param):
    eta = np.max(transit.params['eta'])
    Tion = np.max(transit.params['Tion'])

    fix_value = np.unique(transit.params[fix_param])[fix_i]

    vary_vals = np.unique(transit.params[vary_param])
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, n_lines))
    velocgrid = lya.w2v(transit.wavegrid)
    stride = len(transit.timegrid) // n_lines
    t = transit.timegrid[::stride]

    fig, axs = plt.subplots(1,5, figsize=(10,2.5), sharex=True, sharey=True)
    for ax, vary_val in zip(axs, vary_vals):
        loc_kws = {'eta':eta, 'Tion':Tion, fix_param:fix_value, vary_param:vary_val}
        trans = transit.loc_transmission(**loc_kws)
        trans = trans[::stride, :]
        for tt, ttrans, c in zip(t, trans, colors):
            lbl = f"t = {tt:.1f} h" if tt in [t.max(), t.min()] else '_'
            ax.plot(velocgrid, ttrans, color=c, label=lbl)
        ax.set_title(f"{vary_val:.1e}")

    ax.legend()
    fig.supxlabel("RV (km s-1)")
    fig.supylabel('Transmission')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.07, wspace=0)

    filename = f"{target} fix {fix_param} at {fix_value.value:.1e} vary {vary_param}.png"
    plt.savefig(scratchfolder / filename, dpi=300)


#%%

plot_set('mass', 0, 'mdot_star')
plot_set('mass', 2, 'mdot_star')
plot_set('mdot_star', 0, 'mass')
plot_set('mdot_star', 4, 'mass')