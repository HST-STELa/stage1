import os
import warnings
from functools import reduce

import numpy as np
from astropy.table import Table, vstack
from astropy import time
from astropy import units as u
from tqdm import tqdm

import paths
import database_utilities as dbutils

from stage1_processing import target_lists
from stage1_processing import preloads

from target_selection_tools import query

#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

targets = target_lists.new_data(4)

tic_ids = preloads.stela_names.loc['hostname_file', targets]['tic_id']
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    planetcat = preloads.planets.copy()
planetcat.add_index('tic_id')
planets = planetcat.loc[tic_ids]
planets['pl_id'] = dbutils.planet_suffixes(planets)


#%% test date

test_date_str = '2026-06-01'
test_date = time.Time(test_date_str)


#%%

report_rows = []
exceptions = []
for planet in tqdm(planets):

    try:
        #%% get default exoarchive name

        name = planet['pl_name']
        has_name = not np.ma.is_masked(name)
        toi = planet['toi']
        has_toi = not np.ma.is_masked(toi)
        assert has_name or has_toi, 'Planet does not have a name or TOI.'
        query_name = name if has_name else toi
        arx_name = query.get_default_exoarchive_name(query_name)


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
        arx_ephems = arx_ephems[keep]


        #%% get exofop ephemerides and merge

        if has_toi:
            try:
                fop_ephems = query.get_exofop_ephemerides(toi)
                # this will return ephemerides for all TOIs in the system, so need to filter
                P = planet['pl_orbper']
                keep = np.isclose(fop_ephems['per']*u.d, P, rtol=0.1)
                fop_ephems = fop_ephems[keep]

                fop_groomed = fop_ephems['per per_e epoch epoch_e'.split()].copy()
                fop_groomed.rename_column('per', 'pl_orbper')
                fop_groomed.rename_column('per_e', 'pl_orbpererr')
                fop_groomed.rename_column('epoch', 'pl_tranmid')
                fop_groomed.rename_column('epoch_e', 'pl_tranmiderr')
                src = np.char.add(fop_ephems['puser'], ' ')
                src = np.char.add(src, fop_ephems['pdate'])
                src = np.char.add('ExoFOP: ', src)
                fop_groomed['pl_refname'] = src
                fop_groomed['pl_refname'] = fop_groomed['pl_refname'].astype('object')

                ephems = vstack((arx_ephems, fop_groomed))
            except Exception as e:
                if 'No period/epoch found' in str(e):
                    ephems = arx_ephems
                else:
                    raise
        else:
            ephems = arx_ephems

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

        targfilename = preloads.stela_names.loc['tic_id', planet['tic_id']]['hostname_file']
        folder = paths.target_data(targfilename) / 'ephemerides'
        if not folder.exists():
            os.mkdir(folder)
        filename = f'{targfilename}-{planet['pl_id']}.ephemerides.ecsv'
        ephems.write(folder / filename, overwrite=True)


        #%% add accuracy line to report

        row = dict(planet[['hostname', 'pl_id']])
        row['err'] = min_err
        report_rows.append(row)

    #%% record exceptions
    except Exception as e:
        # exceptions.append(e)
        # continue
        raise


#%% print report

report_tbl = Table(rows=report_rows)
report_tbl.pprint(-1,-1)