import os
import re
from functools import reduce
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astropy import time
from astropy import units as u
from tqdm import tqdm
import requests
from urllib.parse import quote_plus
import pyvo as vo

# region setup

#target data directory
target_data_dir = Path('/Users/parke/Google Drive/Research/STELa/data/targets')


# load STELa target name mapping table
stela_name_tbl_path = '/Users/parke/Google Drive/Research/STELa/public github repo/stage1/reference_files/locked_choices/stela_names.csv'
stela_names = Table.read(stela_name_tbl_path)
stela_names.add_index('tic_id')
stela_names.add_index('hostname')
stela_names.add_index('hostname_file')
stela_names.add_index('hostname_hst')


# load catalog of planets to process
planet_cat_path = '/Users/parke/Google Drive/Research/STELa/scratch/stage2_batch1_initial_selections.ecsv'
planets = Table.read(planet_cat_path)


# test date
test_date_str = '2026-06-01'
test_date = time.Time(test_date_str)

# enregion setup

# region query functions

def get_exoarchive_parameters_from_all_sources(pl_names, cols='*'):
    service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    column_string = ','.join(cols)
    pieces = [f"pl_name = '{name}'" for name in pl_names]
    planets_string =  ' or '.join(pieces)
    resultset = service.search(f'SELECT {column_string} FROM ps WHERE {planets_string}')
    catalog = resultset.to_table()
    return catalog


def get_default_exoarchive_name(objname: str) -> str:
    BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py?objname="
    """Return the default (canonical) name from NASA Exoplanet Archive System Aliases."""
    url = BASE + quote_plus(objname)
    data = requests.get(url, timeout=30).json()
    return data.get("manifest", {}).get("resolved_name")


def get_exofop_ephemerides(toi_pfx: str | int):
    """
    Return latest (best-available) transit ephemeris for a TOI from ExoFOP.
    Output: dict with period [d], t0_bjd_tdb [BJD_TDB], notes, and source URL.
    """
    toi_str = str(toi_pfx).replace("TOI", "").replace(" ", "")
    # normalize like "1066.01"
    if not re.match(r"^\d+(\.\d+)?$", toi_str):
        raise ValueError("TOI must look like '1066.01' or '1066'.")

    # ExoFOP will resolve by TOI number:
    base = f"https://exofop.ipac.caltech.edu/tess/target.php?toi={toi_str}"
    # The page exposes a JSON export; appending '&json' returns the page data as JSON.
    url = base + "&json"
    j = requests.get(url, timeout=30).json()

    pp_rows  = j.get("planet_parameters", [])

    ephemerides = []
    for row in pp_rows:
        name = row.get('name', '')
        toi = row.get('toi', '')
        pdate = row.get('pdate', '')
        puser = row.get('puser', '')
        if not name and not toi:
            continue
        values = [row.get(key, None) for key in 'epoch epoch_e per per_e'.split()]
        if any(value is None for value in values):
            continue
        ephemerides.append((name, toi, *map(float, values), puser, pdate))

    if not ephemerides:
        raise RuntimeError("No period/epoch found on the ExoFOP page JSON for this TOI.")

    ephemeris_table = Table(rows=ephemerides, names='name toi epoch epoch_e per per_e puser pdate'.split())
    return ephemeris_table

# endregion query functions


#  loop through and process planets

report_rows = []
for planet in tqdm(planets):

    # get default exoarchive name
    name = planet['pl_name']
    has_name = not np.ma.is_masked(name)
    toi = planet['toi']
    has_toi = not np.ma.is_masked(toi)
    assert has_name or has_toi, 'Planet does not have a name or TOI.'
    query_name = name if has_name else toi
    arx_name = get_default_exoarchive_name(query_name)


    # get exoplanet archive ephemerides
    arx_epehm_cols = (
        'pl_tranmid',
        'pl_tranmiderr1',
        'pl_tranmiderr2',
        'pl_orbper',
        'pl_orbpererr1',
        'pl_orbpererr2',
        'pl_refname'
    )
    arx_ephems = get_exoarchive_parameters_from_all_sources([arx_name], cols=arx_epehm_cols)

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


    # get exofop ephemerides and merge
    if has_toi:
        fop_ephems = get_exofop_ephemerides(toi)
        # this will return ephemerides for all TOIs in the system, so need to filter
        P = planet['pl_orbper']
        keep = np.isclose(fop_ephems['per'], P, rtol=0.1)
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
    else:
        ephems = arx_ephems


    # pick the most accurate
    T0 = time.Time(ephems['pl_tranmid'].tolist(), format='jd')
    T0err = ephems['pl_tranmiderr'].tolist() * u.d # the tolist and u.d is because astropy isn't recognizing the units attached to the column
    P = ephems['pl_orbper'].tolist() * u.d
    Perr = ephems['pl_orbpererr'].tolist() * u.d
    Tdiff = test_date - T0
    Nper = Tdiff.to('d')/P
    err = np.sqrt(T0err**2 + (Nper*Perr)**2)
    ibest = np.argmin(err)
    min_err = err[ibest].to('min')

    ephems[f'err at {test_date_str}'] = err.to('min')
    ephems['picked'] = False
    ephems['picked'][ibest] = True


    # save to the target folder
    targfilename = stela_names.loc['tic_id', planet['tic_id']]['hostname_file']
    folder = target_data_dir / targfilename / 'ephemerides'
    if not folder.exists():
        os.mkdir(folder)
    filename = f'{targfilename}-{planet['pl_id']}.ephemerides.ecsv'
    ephems.write(folder / filename, overwrite=True)


    # add accuracy line to report
    row = dict(planet[['hostname', 'pl_id']])
    row['err'] = min_err
    report_rows.append(row)


# print report

report_tbl = Table(rows=report_rows)
report_tbl.pprint(-1,-1)