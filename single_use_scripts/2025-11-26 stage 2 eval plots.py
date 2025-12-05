import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from astropy import table

import catalog_utilities as catutils
import paths

from stage1_processing import preloads

slide_size = [9, 5]

mpl.rcParams['savefig.dpi'] = 300

#%% read in the results

url_google_sheet = 'https://docs.google.com/spreadsheets/d/1G77756ETnRfSoCjw-ZOD7rCVP124XmpjYLzRW2-ckzQ/export?format=xlsx&gid=985313130'
eval = catutils.read_excel(url_google_sheet)


#%% make some masks and some global style settings

masks = dict(
    allpts=np.ones(len(eval), dtype=bool),
    batch1=eval['Selected in Eval 1'] == True,
    detections=eval['Lya detected?'] == 'Y',
    tentative=eval['Lya detected?'] == 'T',
    nondetections=eval['Lya detected?'] == 'ND',
    unknown=eval['Lya detected?'] == 'UK'
)

pt_styles = dict(
    allpts=dict(color='0.5', label='all'),
    batch1=dict(color='C0', label='STELa batch 1'),
    detections=dict(color='C2', zorder=10, label='known detections'),
    tentative=dict(color='C1', label='known tentative'),
    nondetections=dict(color='C3', label='known nondetections'),
    unknown=dict(color='C4', label='(to be) observed, result unknown')
)


#%% plot detection fractions vs lya snr

def make_detectability_plot(ykey, yfloor=0):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xscale('log')
    ax.set_xlabel('Lya Flux (erg s-1 cm-2)')
    ax.set_yscale('log')

    x = eval['Lya Flux (erg s-1 cm-2)']
    y = eval[ykey]
    y = np.clip(y, yfloor, np.inf)

    def plot_set(mask, *args, **kws):
        xx = x[mask]
        yy = y[mask]
        ln, = ax.plot(xx, yy, 'o', *args, **kws)
        return ln

    plot_set(masks['allpts'], **pt_styles['allpts'])
    plot_set(masks['batch1'], **pt_styles['batch1'])
    plot_set(masks['detections'], **pt_styles['detections'])
    plot_set(masks['tentative'], **pt_styles['tentative'])
    plot_set(masks['nondetections'], **pt_styles['nondetections'])
    plot_set(masks['unknown'], **pt_styles['unknown'])

    plt.legend(fontsize='small')

    return fig, ax

frac_fig, frac_ax = make_detectability_plot('sim safe offset frac w snr > 3', 1e-4)
frac_ax.set_ylabel('Simulation Detectability Fraction')
frac_fig.tight_layout()
frac_fig.savefig(paths.scratch / '2025-11-26 detctability fraction vs lya flux plot.png')

flat_fig, flat_ax = make_detectability_plot('flat transit max snr', 0)
flat_ax.set_ylabel('Simple Flat Transit SNR')
flat_fig.tight_layout()
flat_fig.savefig(paths.scratch / '2025-11-26 flat transit SNR vs lya flux plot.png')


#%% load in confirmed table

all_confirmed_path = 'target_selection_data/inputs/exoplanet_archive/confirmed_uncut.ecsv'
confirmed = table.Table.read(all_confirmed_path)


#%% plot detection fractions on Teq-Rad plot

def make_scatter(skey, scalefunc=None):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xlabel('Orbital Period (d)')
    ax.set_ylabel('Radius ($R_\\oplus$)')


    s = eval[skey]
    if scalefunc is not None:
        s = scalefunc(s)

    high_precision = confirmed['pl_radeerr1']/confirmed['pl_rade'] < 0.1

    ax.loglog(
        confirmed['pl_orbper'][high_precision],
        confirmed['pl_rade'][high_precision],
        '.', alpha=0.2, mew=0, color='C5'
    )

    def plot_set(mask, **kws):
        ln = ax.scatter(
            eval['orbital period (d)'][mask],
            eval['radius (Re)'][mask],
            s=s[mask],
            **kws
        )
        return ln

    plot_set(masks['allpts'], **pt_styles['allpts'])
    plot_set(masks['batch1'], **pt_styles['batch1'])
    plot_set(masks['detections'], **pt_styles['detections'])
    plot_set(masks['tentative'], **pt_styles['tentative'])
    plot_set(masks['nondetections'], **pt_styles['nondetections'])
    plot_set(masks['unknown'], **pt_styles['unknown'])

    ax.set_xlim(0.3, 200)
    ax.set_ylim(0.3, 30)

    plt.legend(fontsize='small')

    return fig, ax

def frac_scale(s):
    s = np.clip(s, 1e-4, np.inf)
    s = np.cbrt(s)*100
    return s
frac_fig_scat, frac_ax_scat = make_scatter('sim safe offset frac w snr > 3', frac_scale)
frac_ax_scat.set_title('Simulation Detectability Fraction')
frac_fig_scat.tight_layout()
frac_fig_scat.savefig(paths.scratch / '2025-11-26 detectability fraction population.png')

def flat_scale(s):
    s = s*5
    return s
flat_fig_scat, flat_ax_scat = make_scatter('flat transit max snr', flat_scale)
flat_ax_scat.set_title('Flat Transit SNR')
flat_fig_scat.tight_layout()
flat_fig_scat.savefig(paths.scratch / '2025-11-26 flat transit SNR population.png')