from astropy.table import Table
import numpy as np

import paths


#%%

files = list(paths.data_targets.rglob('*xray-recon.fits'))

for file in files:
    tbl = Table.read(file)
    try:
        tbl.rename_columns(
            'Wave Flux Rate'.split(),
            'wavelength flux rate'.split()
        )
    except:
        pass
    tbl.sort('wavelength')
    if tbl['bin_width'][-1] < np.diff(tbl['wavelength'])[-1]*0.6:
        tbl['bin_width'] *= 2
    tbl.write(file, overwrite=True)