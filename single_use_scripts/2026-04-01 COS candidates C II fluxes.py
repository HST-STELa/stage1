
from astropy import table

import paths
import database_utilities as dbutils


#%%

targets = """gj143
hd95338
toi-1434
toi-2443
toi-431""".split('\n')

#%%

for target in targets:
    fd = paths.target_data(target)
    f = dbutils.one_glob(fd,'*line-flux-table.ecsv')
    tbl = table.Table.read(f)
    tbl.add_index('name')
    i = tbl.loc_indices['C II'][0]
    print(f"{target}: {tbl['flux'][i]:.1e}   {tbl['source'][i]}")


#%% measure gj 143's with new data

path = '/Users/parke/Google Drive/Research/STELa/data/targets/gj143/hst/gj143.hst-cos-g140l.2025-09-18T114347--2025-09-18T115941.pgm17794.4exposure_coadd.fits'
import spectralPhoton as sp
from astropy.io import fits
from astropy import units as u
h = fits.open(path)
w, f, e = [h[1].data[s][0] for s in ['wavelength', 'flux', 'error']]
fcgs = u.Unit('erg s-1 cm-2 AA-1')
spec = sp.Spectrum(w*u.AA, f*fcgs, err=e*fcgs)
spec.plot()
F, E = spec.integrate((1332, 1338)*u.AA)

#%% adjust for larger effective area of G130M vs G140L

import numpy as np
F/E * np.sqrt(1800/245)