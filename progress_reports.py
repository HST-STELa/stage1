#%% imports

from astropy import table

from stage1.lya_prediction_tools import lya
from stage1.target_selection_tools import catalog_utilities as catutils, paths

#%% load selections

cat = catutils.load_and_mask_ecsv(paths.intermediates / 'chkpt8__target-build.ecsv')
cat.sort('stage1_rank')
selected = cat[cat['stage1'].filled(False)]
backup = cat[cat['stage1_backup'].filled(False)]
roster = table.vstack((selected, backup))

#%% information for input to observation progress sheet

roster_hosts = catutils.planets2hosts(roster)

params = dict(default_rv=default_sys_rv, show_progress=True)

wgrid = lya.wgrid_std
sets = (('nominal', 0),
        ('optimistic', 34))
lya_fluxes_earth = []
for lbl, pcntl in sets:
    n_H = ism.ism_n_H_percentile(50 - pcntl)
    lya_factor = lya.lya_factor_percentile(50 + pcntl)
    observed = lya.lya_at_earth_auto(roster_hosts, n_H, lya_factor=lya_factor, **params)
    _fluxes = np.trapz(observed, wgrid[None, :], axis=1)
    lya_fluxes_earth.append(_fluxes)

aptnames = apt.cat2apt_names(roster_hosts['hostname'].tolist())
obs_export = table.Table([aptnames], names=['Target'])



labeltbl = table.Table.read(paths.locked / 'target_visit_labels.ecsv')