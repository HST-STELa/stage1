import spectralPhoton as sp
from random import choices
from astropy import units as u
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import paths
import database_utilities as dbutils

#%% find random e140m files

fd = paths.data_targets
e140m_paths = sorted(fd.rglob('*stis-e140m*x1d.fits'))
e140m_paths = [p for p in e140m_paths if dbutils.parse_filename(p)['datetime'] > '2022-01-01T000000']
np.random.seed(20260505)
e140m_paths_random = choices(e140m_paths,k=10) # mostly au mic but thats ok bc mostly i want to sample different airglow

#%% define noise bands (selected by eye from au-mic spec, regions adjacent to major lines)

noise_bands = (
(1172, 1174.5),
(1202, 1203),
(1232, 1234),
(1330, 1333),
(1396, 1398),
)
noise_bands *= u.AA

#%% calculate detection limits

noise_sigma = []
for p in tqdm(e140m_paths_random):
    spec = sp.Spectrum.read_x1d(p)
    for b in noise_bands:
        for s in spec:
            if (s.w[0] < b[0]) and (s.w[-1] > b[-1]):
                _, e = s.integrate(b)
                noise_sigma.append(e.to_value('erg s-1 cm-2'))

#%% pick limit to use

_ = plt.hist(noise_sigma, 10)

# distribution is long tailed, so let's use the median
detection_limit = 5 * np.median(noise_sigma)
print(f'detection limit to adopt = {detection_limit:.1e} cgs flux')