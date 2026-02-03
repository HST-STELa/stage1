from math import nan

from astropy import table

#%%
path = '/Users/parke/Google Drive/Research/STELa/data/packages/inbox/2025-11-24 detectability volumes/stage2_evaluation_metrics.ecsv'
eval_table = table.Table.read(path)

for name in eval_table.colnames:
    try:
        eval_table[[name]].write('/Users/parke/Downloads/temp_test.csv', overwrite=True)
    except TypeError:
        if hasattr(eval_table[name], 'filled'):
            eval_table[name] = eval_table[name].filled(nan).astype(float)
        else:
            eval_table[name] = eval_table[name].astype(float)