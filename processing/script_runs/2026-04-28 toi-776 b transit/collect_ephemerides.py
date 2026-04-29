import os
import warnings
from functools import reduce

import numpy as np
from astropy.table import Table, QTable, vstack
from astropy import time
from astropy import units as u
from tqdm import tqdm

import paths
import database_utilities as dbutils

from processing import preloads

from target_selection_tools import query

#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

target = 'toi-776'

planet_arx_names = [f'TOI-776 {letter}' for letter in 'bc']


#%% test date

test_date_str = '2026-01-13'
test_date = time.Time(test_date_str)


#%%

report_rows = []
exceptions = []
for arx_name in tqdm(planet_arx_names):


    #%% get exoplanet archive ephemerides

    arx_epehm_cols = (
        'pl_tranmid',
        'pl_tranmiderr1',
        'pl_tranmiderr2',
        'pl_orbper',
        'pl_orbpererr1',
        'pl_orbpererr2',
        'pl_refname'
    )
    arx_ephems = query.get_exoarchive_parameters_from_all_sources([arx_name], cols=arx_epehm_cols)

    # merge errors
    Perr1, Perr2 = arx_ephems['pl_orbpererr1'], arx_ephems['pl_orbpererr2']
    Perr = np.max((Perr1, -Perr2), 0)
    T0err1, T0err2 = arx_ephems['pl_tranmiderr1'], arx_ephems['pl_tranmiderr2']
    T0err = np.max((T0err1, -T0err2), 0)

    arx_ephems['pl_orbpererr'] = Perr1
    arx_ephems['pl_tranmiderr'] = T0err1

    arx_ephems = arx_ephems['pl_tranmid pl_tranmiderr pl_orbper pl_orbpererr pl_refname'.split()]
    masks = [arx_ephems[col].mask for col in 'pl_tranmid pl_tranmiderr pl_orbper pl_orbpererr'.split()]
    keep = ~reduce(np.logical_or, masks[1:], masks[0])
    ephems = arx_ephems[keep]

    assert len(ephems) > 0


    #%% pick the most accurate

    T0 = time.Time(ephems['pl_tranmid'].tolist(), format='jd')
    T0err = ephems['pl_tranmiderr'].tolist() * u.d # the tolist and u.d is because astropy isn't recognizing the units attached to the column
    P = ephems['pl_orbper'].tolist() * u.d
    Perr = ephems['pl_orbpererr'].tolist() * u.d
    Tdiff = test_date - T0
    Nper = Tdiff.to('d')/P
    err = np.sqrt(T0err**2 + (Nper*Perr)**2)
    err[err == 0] = np.inf # ignore ephemerides without reported errors
    ibest = np.argmin(err)
    min_err = err[ibest].to('min')

    ephems[f'err at {test_date_str}'] = err.to('min')
    ephems['picked'] = False
    ephems['picked'][ibest] = True


    #%% save to the target folder

    folder = paths.target_data(target) / 'ephemerides'
    assert folder.exists()
    letter = arx_name[-1]
    filename = f'{target}-{letter}.ephemerides.ecsv'
    ephems.write(folder / filename, overwrite=True)


    #%% add accuracy line to report

    row = {'planet': arx_name}
    row['err'] = min_err
    report_rows.append(row)


#%% print report

report_tbl = Table(rows=report_rows)
report_tbl.pprint(-1,-1)