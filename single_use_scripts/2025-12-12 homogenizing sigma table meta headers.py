from tqdm import tqdm

import paths
import database_utilities as dbutils

from stage1_processing import transit_evaluation_utilities as tutils

files = list(paths.data_targets.rglob('*.detection-sigmas.ecsv'))

baseline_apertures = dict(g140m='52x0.2', e140m='6x0.2')

#%%
for file in tqdm(files):
    target = file.name.split('.')[0]
    target, _ = dbutils.split_hostname_planet_letter(target, '-')
    host = tutils.Host(target)
    grating = host.anticipated_grating
    base_aperture = baseline_apertures[grating]

    snrs = tutils.DetectabilityDatabase.from_file(file)
    snrs.infer_best_observing_configuration(grating, base_aperture)

    snrs.write(file, overwrite=True)