#%% imports

from astropy.io import fits
from matplotlib import pyplot as plt

import database_utilities as dbutils


#%% set the observations you want to inspect, locate files

directory = '/Users/parke/Google Drive/Research/STELa/data/uv_observations/hst-stis'
# targets = ['hd17156', 'k2-9', 'toi-1434', 'toi-1696', 'wolf503', 'hd207496']
# targets = ['toi-2015', 'toi-2079']
targets = 'any'
inst = 'hst-stis-mirvis'
rawfiles = dbutils.find_target_files('raw', targets, directory)
rawfiles = [f for f in rawfiles if inst in f.name]


#%% plot

stages = ['coarse', 'fine', '0.2x0.2']
for file in rawfiles:
    fig, axs = plt.subplots(1, 3, figsize=[7,3])
    h = fits.open(file)
    for i, ax in enumerate(axs):
        data = h['sci', i+1].data
        ax.imshow(data)
        ax.set_title('')
    fig.suptitle(file.name)
    fig.supxlabel('dispersion')
    fig.supylabel('spatial')
    fig.tight_layout()

