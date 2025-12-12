import paths
from functools import lru_cache

from stage1_processing import target_lists

targets = target_lists.eval_no(1) + target_lists.eval_no(2)

#%%

targets_to_run = []
for target in targets:
    folder = paths.target_data(target)
    sigma_files = list(folder.rglob('*detection-sigmas.ecsv'))
    if len(sigma_files) < 2:
        targets_to_run.append(target)


#%%

targets = targets_to_run
# now run the stage2 eval script using these targets


#%% targets missing lya reconstruction files

missing_lya = []
for target in targets:
    folder = paths.target_data(target)
    lya_files = list(folder.rglob('*lya-recon*'))
    if not lya_files:
        missing_lya.append(target)

# there turned out to be none, this was a red herring from a weird traceback

