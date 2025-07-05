from copy import copy

import numpy as np
from astropy.table import Table

table = Table.read('reference_files/fuv_line_list.ecsv')

wave_tol = 10

# Sort table by 'name' and 'wave'
table.sort(['name', 'wave'])

merged_rows = []
used = np.zeros(len(table), dtype=bool)

for i, row in enumerate(table):
    if used[i]:
        continue

    # Group rows by same 'name' and similar 'wave'
    same_name = (table['name'] == row['name'])
    wave_diff = np.abs(table['wave'] - row['wave']) <= wave_tol
    group_mask = same_name & wave_diff & ~used

    group = table[group_mask]
    used[group_mask] = True

    # Average wave
    avg_wave = np.round(np.mean(group['wave']),0)

    # Mark as blend=True if group has more than 1 row or any were blended
    blend_val = True if len(group) > 1 else group[0]['blend']

    # Use the first row for all other fields
    new_row = copy.copy(row)
    new_row['wave'] = avg_wave
    new_row['blend'] = blend_val

    merged_rows.append(new_row)

merged = Table(rows=merged_rows, names=table.colnames)
merged.sort('wave')
merged.remove_column('Tform')
merged['wave'] = merged['wave'].astype(int)

merged.write('reference_files/fuv_line_list_merged.ecsv')