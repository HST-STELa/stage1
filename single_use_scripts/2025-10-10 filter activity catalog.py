from astropy import table
from stage1_processing import target_lists
from stage1_processing import preloads


targets = target_lists.eval_no(2)
cat = table.Table.read('/Users/parke/Downloads/activity_catalog_full.csv')
tics = preloads.stela_names.loc['hostname_file', targets]['tic_id']

cat.add_index('tic_id')
slim = cat.loc[tics]

slim.write('/Users/parke/Downloads/activity_catalog_eval2.csv')