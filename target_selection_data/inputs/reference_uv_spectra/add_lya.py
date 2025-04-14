from astropy import table
from astropy import units as u
import numpy as np

import lya
import paths

masked_folder = paths.selection_inputs / 'reference_spectra/lya_masked'
added_folder = paths.selection_inputs / 'reference_spectra/lya_added_intrinsic'
files = list(masked_folder.glob('*.spec'))
for path in files:
    spec = table.Table.read(path, format='ascii.ecsv')
    path_pieces = path.name.split('-')
    Teff = float(path_pieces[1])
    Prot = 1 if Teff < 2700 else 50 # this just puts the star in the right activity category for the linsky relationship
    Flya_1au, = lya.Lya_from_Teff_linsky13(np.array([Teff]), np.array([Prot]))
    d = spec.meta['distance']
    Flya_at_dist = Flya_1au * (u.AU/(d*u.pc))**2
    Flya_at_dist = Flya_at_dist.to_value('')
    lyamask = np.isnan(spec['f']) & (spec['w'] < 1250)
    wlya = spec['w'][lyamask]
    ylya = lya.reversed_lya_profile(wlya*u.AA, 0*u.km/u.s, Flya_at_dist*u.Unit('erg s-1 cm-2'))
    spec['f'][lyamask] = ylya.to_value('erg s-1 cm-2 AA-1')
    fillmask = np.isnan(spec['f']) | (spec['f'] < 0)
    spec['f'][fillmask] = 0

    savepath = added_folder / path.name.replace('.spec', '.dat')
    data = np.array((spec['w'].data, spec['f'].data))
    np.savetxt(savepath, data.T)