
import numpy as np

import utilities as utils


def read_lsf(path_lsf, wavelength):
    lsfs = np.loadtxt(path_lsf, skiprows=1)
    with open(path_lsf) as f:
        hdr = f.readline()
    waves = list(map(float, hdr.split(' ')))

    y = utils.interpolate_many([wavelength], waves, lsfs)
    y = np.squeeze(y)

    # normalize so convolution preserves flux
    ynormfac = np.trapz(y)
    ynorm = y / ynormfac

    x = np.arange(len(ynorm))

    return x, ynorm


default_etc_filenames = dict(
    g130m = {
        'psa': 'etc.hst-cos-g130m.2025-08-14.2026269.exptime900_flux1e-13_aperturepsa.csv'
    }
)

