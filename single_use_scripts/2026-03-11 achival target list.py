import paths
import catalog_utilities as catutils
import database_utilities as dbutils

#%%
progtbl = catutils.read_excel(paths.observation_progress_google_sheet_xlsx_export)
lya_good = progtbl["External\nLya Good?"]
prog_mask = progtbl["External\nLya"] & (lya_good != 'fail')
prog_tics = progtbl['TIC ID'][prog_mask]

cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt7__cut-low-snr__add-flags-scores.ecsv')
cat = cat[cat['stage1_rank'] <= 200]
cat_tics = cat[cat['external_lya'].filled(False)]['tic_id']

all_tics = set(prog_tics) | set(cat_tics)
names = dbutils.stela_name_tbl.loc['tic_id', list(all_tics)]['hostname_file']
names.tolist()