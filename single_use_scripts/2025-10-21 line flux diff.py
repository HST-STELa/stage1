from astropy import table
import numpy as np

import paths

from stage1_processing import target_lists


#%%

oldfolder = paths.data / 'packages' / '2025-06-16.stage2.eval1.staging_area' / 'fuv_line_fluxes'


targets = target_lists.eval_no(1)

#%%

rows = []
for target in targets:
    row = {'target': target}

    def read_tbl(folder):
        tbl_path, = folder.glob(f'{target}*line-flux-table.ecsv')
        tbl = table.Table.read(tbl_path)
        tbl.add_index('name')
        return tbl

    oldtbl = read_tbl(oldfolder)

    targfolder = paths.target_data(target)
    newtbl = read_tbl(targfolder)

    lines = set(oldtbl['name'].tolist() + newtbl['name'].tolist())

    def get_flux_sum(tbl, line):
        fluxes = tbl.loc[line]['flux']
        if hasattr(fluxes, 'filled'):
            fluxes = fluxes.filled(0)
        return np.sum(fluxes)

    for line in lines:
        oldflux = get_flux_sum(oldtbl, line)
        newflux = get_flux_sum(newtbl, line)
        row[f'{line}_0'] = oldflux
        row[f'{line}_1'] = newflux
        if oldflux == 0 and newflux == 0:
            ratio = np.ma.masked
        else:
            ratio = oldflux/newflux
        row[f'{line}_0/1'] = ratio
    rows.append(row)

difftbl = table.Table(rows=rows)

difftbl.write(paths.scratch / 'line flux diff table 2025-10-21.csv', overwrite=True)