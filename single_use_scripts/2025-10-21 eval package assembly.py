import shutil

import paths

from stage1_processing import target_lists


#%% move line files

targets = target_lists.eval_no(1) + target_lists.eval_no(2)
targets.remove('v1298tau')
destination = paths.data / 'packages/2025-09-26.stag2.eval2.staging_area/fuv_line_fluxes'

for target in targets:
    folder = paths.target_data(target)
    line_file, = folder.glob('*line-flux-table.ecsv')
    shutil.copy(line_file, destination / line_file.name)