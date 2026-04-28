
from types import SimpleNamespace

import plotly
from astropy import table
from astropy import time

import paths
import ephemeris


#%% setup

star = 'l98-59'
planet = f'{star}-c'
fd = paths.target_data(star)


#%% files

x1d_files = sorted(fd.glob('hst/*g140m*2026-04*flt.fits'))
flt_files = [f.parent / f.name.replace('_x1d', '_flt') for f in x1d_files]


#%% folder to store transit results

tst_fd = fd / 'transits' / f'{planet}.


#%% plot flats

flt_files = sorted(fd.glob('hst/*g140m*2026-04*flt.fits'))



#%% plot spectra

x1d_files = paths.


#%% plot lightcurves
