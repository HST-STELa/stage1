"""
One-off: pair Ava vs Parke *.line_fluxes.ecsv for the same coadds, merge on
(key, wave), write comparison ECSV to compared/.

dflux = flux_parke - flux_ava
err_mean = (error_ava + error_parke) / 2
dflux_over_err_mean = dflux / err_mean (masked if err_mean <= 0 or a flux missing)
"""

from pathlib import Path

import numpy as np
from astropy import table

#%% paths

BASE = Path(
    "/Users/parke/Google Drive/Research/STELa/scratch/"
    "2026-04-02 ava line flux comparison"
)
AVA = BASE / "ava"
PARKE = BASE / "parke"
OUT = BASE / "compared"

discard_cols = 'key atom ionztn Tform blend'.split()
def compare_pair(path_ava: Path, path_parke: Path) -> table.Table:
    t_ava = table.Table.read(path_ava)
    t_parke = table.Table.read(path_parke)

    for t in (t_ava, t_parke):
        t.remove_columns(discard_cols)

    cmp = table.join(t_ava, t_parke, keys=['name', 'wave'], table_names=['a', 'p'], join_type='outer')
    col_order = []
    for name in cmp.colnames:
        if '_a' in name:
            col_order.extend([name, name.replace('_a', '_p')])
        elif '_p' in name:
            continue
        else:
            col_order.append(name)
    cmp = cmp[col_order]

    dflux = cmp['flux_a'] - cmp['flux_p']
    derr = np.sqrt((cmp['error_a']**2 + cmp['error_p']**2))
    dflux_derr = dflux/derr

    cmp['dflux'] = dflux
    cmp['derr'] = derr
    cmp['dflux/derr'] = dflux_derr
    cmp['dflux'].format = '.2e'
    cmp['derr'].format = '.2e'
    cmp['dflux/derr'].format = '.1f'

    return cmp


#%%

ava_files = {p.name: p for p in AVA.glob("*.line_fluxes.ecsv")}
parke_files = {p.name: p for p in PARKE.glob("*.line_fluxes.ecsv")}
names = sorted(set(ava_files) & set(parke_files))

for name in names:
    pout = OUT / f"{name[:-5]}.compared.ecsv"
    cmp = compare_pair(ava_files[name], parke_files[name], )
    cmp.write(pout, overwrite=True)

    print(name)
    cmp.pprint(-1,-1)
    print()
    print()


