import numpy as np
from astropy import units as u

import catalog_utilities as catutils
import paths
from target_selection_tools import reference_tables as ref


#%% T3 visit contaminating target type

BP = 2.289
G = 13.032
plx = 27.2456
dist = 1000/plx * u.pc

Teff = catutils.safe_interp_table(BP, 'Bp-Rp', 'Teff', ref.mamajek)
i = np.searchsorted(ref.mamajek['Teff'][::-1], Teff)
i = len(ref.mamajek) - i - 1
ref.mamajek['Teff'][i]
SpT = ref.mamajek['SpT'][i]

G_V = catutils.safe_interp_table(BP, 'Bp-Rp', 'G-V', ref.mamajek)
B_V = catutils.safe_interp_table(BP, 'Bp-Rp', 'B-V', ref.mamajek)
U_B = catutils.safe_interp_table(BP, 'Bp-Rp', 'U-B', ref.mamajek)
U = G - G_V + B_V + U_B

R = catutils.safe_interp_table(BP, 'Bp-Rp', 'R_Rsun', ref.mamajek)

ref.mamajek[['Teff', 'SpT']][i-5:i+5]

#%% NX note

"""This one already cleared, see email thread with Josh."""


#%% OP

import astropy.coordinates as coord
from astropy import time

# from gaia
ra = 289.7521963412162 * u.deg
dec = 41.63164455302341 * u.deg
plx = 27.607874153583527 * u.mas
pmra = 94.5078744156069 * u.mas/u.yr
pmracosdec = pmra * np.cos(dec)
pmdec = -630.781056151556 * u.mas/u.yr
dist = 1000*u.pc/u.mas/plx
pos = coord.SkyCoord(ra, dec, pm_ra_cosdec=pmracosdec, pm_dec=pmdec, distance=dist, obstime='J2016.0')

spex_obstimes = time.Time(
    (
        57610.407251157,
        60067.611585648,
        60135.472447917,
        60474.534247685
    ),
    format='mjd'
)

for t in spex_obstimes:
    newpos = pos.apply_space_motion(t)
    print(f"{t.mjd}: {newpos.to_string(precision=6)}")


#%% redo U mag calc for 444-B just to check


Teff = 3464
MK = 6.91 # absolute mags
dist = 35.7 * u.pc
K = MK + 5*np.log10(dist/(10*u.pc))
colors = {}
for key in ('V-Ks', 'B-V', 'U-B'):
    colors[key] = catutils.safe_interp_table(MK, 'M_Ks', key, ref.mamajek)
c = colors
U = K + c['V-Ks'] + c['B-V'] + c['U-B']


#%% plot Kepler-444 Lya to see whether we will actually be able to measure any FUV lines in all likelihood

fxs = sorted(paths.target_data('kepler-444').rglob('*_x1d.fits'))
fx = fxs[0]

from astropy.io import fits
from matplotlib import pyplot as plt
h = fits.open(fx)
d = h[1].data
plt.plot(d['wavelength'].T, d['flux'].T)