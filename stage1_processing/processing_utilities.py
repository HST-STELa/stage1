
import numpy as np
from astropy.table import Table

import paths

def get_intrinsic_lya_flux(target_name_file, return_16_50_84=False):
    recon_folder = paths.target_data(target_name_file) / 'reconstructions'
    lya_file, = recon_folder.rglob('*lya-recon.csv')
    lyarecon = Table.read(lya_file)
    suffixes = ['low_1sig', 'median', 'high_1sig'] if return_16_50_84 else ['median']
    Fs = []
    for suffix in suffixes:
        F = np.trapz(lyarecon[f'lya_intrinsic unconvolved_{suffix}'], lyarecon['wave_lya'])
        Fs.append(F)
    if return_16_50_84:
        return Fs
    else:
        return Fs[0]