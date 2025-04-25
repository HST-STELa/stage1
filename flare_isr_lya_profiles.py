import numpy as np
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt

import synphot

import paths

from target_selection_tools import catalog_utilities as catutils
from lya_prediction_tools import ism, lya

#%% load and join parameter and ISR tables -- make sure to download a new csv export first!

parameters = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt3__fill-basic_properties.ecsv')
parameters = catutils.planets2hosts(parameters)
isr_table = table.Table.read(paths.checked / 'mdwarf_isr_continuously_updated flux values export.csv')
isr_table.add_index('Target')
i = isr_table.loc_indices['EXAMPLES BELOW']
isr_table = isr_table[:i]

target_names = isr_table['Target']
parameters.add_index('hostname')
target_parameters = parameters.loc[target_names]
# need to use hstack instead of join or table will be resorted
target_parameters = table.hstack((target_parameters, isr_table))


#%% compute ism rvs

ras, decs = target_parameters['ra'].quantity, target_parameters['dec'].quantity
ism_rvs = u.Quantity([ism.ism_velocity(ra, dec) for ra, dec in zip(ras, decs)])
target_parameters['ism_radv'] = ism_rvs
target_parameters['ism_radv'].format = '.2f'


#%% Check to be sure the targets are in the right order

target_parameters['hostname'].pprint(-1)


#%% print stellar RVs to copy and paste into ISR table

target_parameters['st_radv'].pprint(-1)


#%% print ISM RVs to copy and paste into ISR table

target_parameters['ism_radv'].pprint(-1)


#%% tools to generate spectra for input to etc

def flare_line(w, w0, flux, fwhm):
    sigma = lya.sig_from_fwhm_wave(lya.wlab_H, fwhm)
    amp = lya.gaussian_amp(flux, sigma)
    return lya.gaussian_profile(w, w0, amp, sigma)


def flare_lya_with_ISM(w, flux, rv_offset, Nh):
    w0 = 1215.67*u.AA
    fwhm = 0.5*u.AA
    Tism = 1e4*u.K
    intrinsic = flare_line(w, w0, flux, fwhm)
    transmitted = lya.transmission(w, rv_offset, Nh, Tism)
    observed = intrinsic * transmitted
    return observed

vegaspec = synphot.SourceSpectrum.from_vega()
u_band = synphot.SpectralElement.from_filter('johnson_u')
def normed_bb(w, Umag):
    Tbb = 9000*u.K
    bb = synphot.models.BlackBody1D(temperature=Tbb)
    bb = synphot.SourceSpectrum(bb)
    bb_normed = bb.normalize(Umag*synphot.units.VEGAMAG, band=u_band, vegaspec=vegaspec)
    y = bb_normed(w, flux_unit='FLAM')
    return y.to('erg s-1 cm-2 AA-1')


#%% make and save spectra for etc input for each star

w = np.arange(1100, 1800, 0.1)*u.AA
Nh_cols = [6e16, 1e17] * u.cm**-2

for target in target_parameters:
    if np.ma.is_masked(target['st_radv']):
        print(f"No spectrum generated for {target['hostname']} because there is no stellar RV measurement in the catalog.")
        continue

    fluxcols = ['F_C4', 'F_Si4', 'F_Lya']
    Fc4, Fsi4, Flya = [target[key] * u.Unit('erg s-1 cm-2') for key in fluxcols]
    yc4 = flare_line(w, 1548.2*u.AA, Fc4, 0.2*u.AA)
    ysi4 = flare_line(w, 1393.8*u.AA, Fc4, 0.2*u.AA)
    rv_offset = (target['st_radv'] - target['ism_radv']) * u.km/u.s

    ybb = normed_bb(w, target['U_flare'])

    for Nh in Nh_cols:
        ylya = flare_lya_with_ISM(w, Flya, rv_offset, Nh)

        y = ybb + yc4 + ysi4 + ylya

        name = f"{target['hostname']}_Nh{Nh.value:.0e}.dat"
        fullpath = paths.selection_outputs / 'isr_flare_spectra' / name
        data = np.array((w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1'))).T
        np.savetxt(str(fullpath), data)


#%% load and plot

name = 'TOI-1231'
Nh_load = 6e16
folder = paths.selection_outputs / 'isr_flare_spectra'
file, = folder.glob(f'*{name}*{Nh_load:.0e}.dat')
ww, ff = np.loadtxt(str(file)).T

plt.figure()
plt.plot(ww, ff)
plt.title(f'{name} Nh={Nh_load}')
plt.yscale('log')