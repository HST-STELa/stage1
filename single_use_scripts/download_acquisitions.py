from pathlib import Path

import numpy as np
from astroquery.mast import MastMissions
from astropy.io import fits

#%% get aquistion ids for all tag files

data_dir = Path('/Users/parke/Google Drive/Research/STELa/data/targets')
paths = list(data_dir.rglob('*tag.fits')) + list(data_dir.rglob('*tag_a.fits')) + list(data_dir.rglob('*tag_b.fits'))
paths = sorted(paths)
acq_ids = []
paths_wo_acq_info = []
for path in paths:
    hdr = fits.getheader(path)
    if 'hst-stis' in path.name:
        acq = hdr['acqname'][:-1] + '*'
        if len(acq) > 1:
            acq_ids.append(acq)
        else:
            paths_wo_acq_info.append(path)
    elif 'hst-cos' in path.name:
        acqlst = []
        for key in ['ACQINAME', 'PEAKXNAM', 'PEAKDNAM']:
            acq = hdr[key][:-1] + '*'
            if len(acq) > 1:
                acqlst.append(acq)
        if len(acqlst) > 0:
            acq_ids.extend(acqlst)
        else:
            paths_wo_acq_info.append(path)
acq_ids = np.unique(acq_ids)


#%% download data for files with acquisition ids in header

mission = MastMissions(mission='hst')
dnld_dir = Path('/Users/parke/Downloads/mast_acqs_2025-06-14')
results = mission.query_criteria(sci_data_set_name=', '.join(acq_ids))
datasets = mission.get_product_list(results)
filtered = mission.filter_products(datasets, file_suffix=['RAW', 'RAWACQ'])
manifest = mission.download_products(filtered, download_dir=dnld_dir, flat=True)


#%% download data for files without acquisition in header

