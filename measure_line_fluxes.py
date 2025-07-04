import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from astropy.io import fits
from matplotlib import pyplot as plt

import database_utilities as dbutils
import paths
import utilities as utils

#%% general setup

linecat = table.Table.read('reference_files/fuv_line_list.ecsv')
linewaves = linecat['wave'].value

configs = (
    'stis_g140m',
    'stis-g140l',
    'stis-e140m',
    'cos-g130m',
    'cos-g160m',
    'cos-g140l'
)

stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')


def interactive_click_loop(fig, plot_fn):
    def get_and_plot():
        xy = utils.click_coords(fig)
        if len(xy):
            x, y = zip(*xy)
            plotted_artists = plot_fn(x, y)
            plt.draw()
            return x, plotted_artists
        else:
            return [], []

    print('Click points to mark the intervals of viable data.')
    x, artists = get_and_plot()
    while True:
        print('Click off the plot if satisfied. Click new points if not.')
        xnew, newartists = get_and_plot()
        if xnew:
            # Remove plotted artists before repeating
            for artist in artists:
                artist.remove()
            x, artists = xnew, newartists
        else:
            break



    return x

#%% target specific
# taget = 'GJ 486'
tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
targname_file = str(targname_file)
data_folder = paths.data / targname_file

#%% custom coadds?



#%% loop
for config in configs:
    files = dbutils.find_data_files('coadd', instruments=config, directory=data_folder)
    if not files:
        files = dbutils.find_data_files('x1d', instruments=config, directory=data_folder)
        if not files:
            continue

    for file in files:
        spec = fits.getdata(file, 1)
        fig = plt.figure(figsize=(15,5))
        plt.title(file.name)
        plt.step(spec['wavelength'].T, spec['flux'].T, where='mid')
        plt.tight_layout()

        # gather viable edges for fitting
        edgeplot = lambda x, y: [plt.axvline(xx, color='0.5', lw=2) for xx in x]
        edges = interactive_click_loop(fig, edgeplot)
        edges = np.reshape(edges, (-1,2))

        # plot the lines that are within
        within = (linewaves[:,None] >= edges[:,0]) & (linewaves[:,None] <= edges[:,1])
        within = within.any(axis=1)
        obslines = linecat[within].copy()
        for line in obslines:
            plt.axvline(line['wave'], color='0.5', lw=0.5)
            plt.annotate(line['name'], xy=(line['wave'], 0.99), xycoords=('data', 'axes fraction'),
                         rotation='vertical', ha='center', va='top')



"""
thoughts for possible automation in the future
  account for instrument lsf in bandpass
  account for stellar rotation in bandpass (can be 500 km s-1 for a 0.1 Rsun star rotating 1 d)
  account for spread of blended lines in bandpass
  sy radv in locating band -- can find rough offset based on spec too perhaps
  have some settings for excluding some lines due to airglow or other reasons
  generate plots to check by eye
"""