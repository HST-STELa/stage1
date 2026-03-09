import re

from astropy import table
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import paths
import catalog_utilities as catutils
import database_utilities as dbutils

mpl.use('qt5agg')


#%% settings

# eiher load google sheet or csv
# url_google_sheet = 'https://docs.google.com/spreadsheets/d/1G77756ETnRfSoCjw-ZOD7rCVP124XmpjYLzRW2-ckzQ/export?format=xlsx&gid=985313130'
# eval_table = catutils.read_excel(url_google_sheet)
eval_filename = 'stage2_evaluation_metrics.ecsv'
eval_path = paths.catalogs / eval_filename
eval_table = catutils.load_and_mask_ecsv(eval_path)

slide_size = [9, 5]
mpl.rcParams['savefig.dpi'] = 300


#%% load table and make labels

labels = np.char.add(eval_table['hostname'], ' ')
labels = np.char.add(labels, eval_table['planet'])
eval_table['labels'] = labels


#%% infer det vol thresh

vol_col = [n for n in eval_table.colnames if 'frac w snr' in n][0]
snr_thresh, = re.findall(r'snr > (\d)', vol_col)
x = snr_thresh


#%% replace any newline characters in colnames for ease of use

for name in eval_table.colnames:
    eval_table.rename_column(name, name.replace('  ', ' '))

#%% make some masks for population plots

masks = dict(
    allpts=np.ones(len(eval_table), dtype=bool),
    detections=eval_table['Lya detected?'] == 'Y',
    tentative=eval_table['Lya detected?'] == 'T',
    nondetections=eval_table['Lya detected?'] == 'ND',
    unknown=eval_table['Lya detected?'] == 'UK',
    rocky=~eval_table['flag_gaseous']
)

pt_styles = dict(
    allpts=dict(color='0.5', label='all'),
    detections=dict(color='C2', zorder=10, label='known detections'),
    tentative=dict(color='C1', label='known tentative'),
    nondetections=dict(color='C3', label='known nondetections'),
    unknown=dict(color='C4', label='(to be) observed, result unknown'),
)


#%% plot detection fractions vs lya snr

def make_detectability_plot(ykey, yfloor=0):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xscale('log')
    ax.set_xlabel('Lya Flux (erg s-1 cm-2)')
    ax.set_yscale('log')

    x = eval_table['Lya Flux (erg s-1 cm-2)']
    y = eval_table[ykey]
    y = np.clip(y, yfloor, np.inf)

    def plot_set(mask, *args, addlbls=False, dim_rocks=True, **kws):
        xx = x[mask]
        yy = y[mask]
        if dim_rocks:
            rky = masks['rocky'][mask]
            ln, = ax.plot(xx[~rky], yy[~rky], 'o', *args, **kws)
            ax.plot(xx[rky], yy[rky], 'o', alpha=0.33, *args, **kws)
            ax.annotate('transulcent = rocky',
                        xy=(0.98,0.33), xycoords='axes fraction',
                        ha='right', va='center',
                        bbox=dict(fc='white', pad=1, alpha=0.5, ls='none'))
        else:
            ln, = ax.plot(xx, yy, 'o', *args, **kws)
        if addlbls:
            lbls = eval_table['labels'][mask]
            for _x,_y,lbl in zip(xx, yy, lbls):
                ax.annotate(' ' + lbl, xy=(_x,_y), fontsize='xx-small', rotation=45)
        return ln

    lna = plot_set(masks['allpts'], addlbls=False, dim_rocks=True, **pt_styles['allpts'])
    lnd = plot_set(masks['detections'], addlbls=True, dim_rocks=True, **pt_styles['detections'])
    lnt = plot_set(masks['tentative'], addlbls=True, dim_rocks=True, **pt_styles['tentative'])
    lnn = plot_set(masks['nondetections'], addlbls=True, dim_rocks=True, **pt_styles['nondetections'])
    lnu = plot_set(masks['unknown'], addlbls=True, dim_rocks=True, **pt_styles['unknown'])

    plt.legend(handles=(lna, lnd, lnt, lnn, lnu), fontsize='small')

    return fig, ax

date = dbutils.timestamp(date_only=True)

frac_fig, frac_ax = make_detectability_plot(f'sim safe offset frac w snr > {x}', 1e-4)
frac_ax.set_ylabel('Simulation Detectability Fraction')
frac_fig.tight_layout()
frac_fig.savefig(paths.scratch / f'{date} detectability fraction vs lya flux plot.png')

flat_fig, flat_ax = make_detectability_plot('flat transit snr', 0)
flat_ax.set_ylabel('Simple Flat Transit SNR')
flat_fig.tight_layout()
flat_fig.savefig(paths.scratch / f'{date} flat transit SNR vs lya flux plot.png')


#%% load in confirmed table

all_confirmed_path = paths.ipac / 'confirmed_uncut.ecsv'
confirmed = table.Table.read(all_confirmed_path)


#%% plot detection fractions on Teq-Rad plot

def make_scatter(skey, scalefunc=None):
    fig, ax = plt.subplots(1, 1, figsize=slide_size)
    ax.set_xlabel('Orbital Period (d)')
    ax.set_ylabel('Radius ($R_\\oplus$)')


    s = eval_table[skey]
    if scalefunc is not None:
        s = scalefunc(s)

    high_precision = confirmed['pl_radeerr1']/confirmed['pl_rade'] < 0.1

    ax.loglog(
        confirmed['pl_orbper'][high_precision],
        confirmed['pl_rade'][high_precision],
        '.', alpha=0.2, mew=0, color='C5'
    )

    def plot_set(mask, addlbls=False, **kws):
        ln = ax.scatter(
            eval_table['orbital period (d)'][mask],
            eval_table['radius (Re)'][mask],
            s=s[mask],
            **kws
        )
        return ln

    plot_set(masks['allpts'], addlbls=False, **pt_styles['allpts'])
    plot_set(masks['detections'], addlbls=True, **pt_styles['detections'])
    plot_set(masks['tentative'], addlbls=True, **pt_styles['tentative'])
    plot_set(masks['nondetections'], addlbls=True, **pt_styles['nondetections'])
    plot_set(masks['unknown'], addlbls=True, **pt_styles['unknown'])

    ax.set_xlim(0.3, 200)
    ax.set_ylim(0.3, 30)

    plt.legend(fontsize='small')

    return fig, ax

date = dbutils.timestamp(date_only=True)

def frac_scale(s):
    s = np.clip(s, 1e-4, np.inf)
    s = np.cbrt(s)*100
    return s
frac_fig_scat, frac_ax_scat = make_scatter(f'sim safe offset frac w snr > {x}', frac_scale)
frac_ax_scat.set_title('Simulation Detectability Fraction')
frac_fig_scat.tight_layout()
frac_fig_scat.savefig(paths.scratch / f'{date} detectability fraction population.png')

def flat_scale(s):
    s = s*5
    return s
flat_fig_scat, flat_ax_scat = make_scatter('flat transit snr', flat_scale)
flat_ax_scat.set_title('Flat Transit SNR')
flat_fig_scat.tight_layout()
flat_fig_scat.savefig(paths.scratch / f'{date} flat transit SNR population.png')