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


#%%
def frac_scale(s):
    s = np.clip(s, 1e-5, np.inf)
    s = np.cbrt(s)*100
    return s
frac_fig_scat, frac_ax_scat = make_scatter('weird ratio', frac_scale)
frac_ax_scat.set_title('Ratio of Simulation Detectability Fraction to Max Flat Transit SNR')
frac_fig_scat.tight_layout()
frac_fig_scat.savefig(paths.scratch / '2025-12-16 weird detectability ratio population.png', dpi=300)