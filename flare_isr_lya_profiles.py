import numpy as np
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt

import synphot

import paths

import catalog_utilities as catutils
from lya_prediction_tools import ism, lya

#%% load and join parameter and ISR tables -- make sure to download a new csv export first!

parameters = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt3__add-archival_obs_counts.ecsv')
parameters = catutils.planets2hosts(parameters)
isr_table = table.Table.read(paths.checked / 'mdwarf_isr_continuously_updated flux values export.csv', header_start=2, data_start=4)
(i,), = np.nonzero(isr_table['Target'] == 'EXAMPLES BELOW')
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
    cat_rv = target['st_radv']
    valid_cat_rv = not np.ma.is_masked(cat_rv)
    xls_rv = target['Stellar RV']
    valid_xls_rv = not (np.ma.is_masked(xls_rv) or (xls_rv in ['', '--']))
    if valid_xls_rv:
        xls_rv = float(xls_rv)
    if not valid_cat_rv and not valid_xls_rv:
        print(f"No spectrum generated for {target['hostname']} because there is no stellar RV measurement in the catalog.")
        continue
    elif valid_cat_rv and not valid_xls_rv:
        rv = cat_rv
    elif not valid_cat_rv and valid_xls_rv:
        rv = xls_rv
    else:
        if not np.isclose(cat_rv, xls_rv, atol=5):
            print(f'RVs in the exocat and ISR spreadsheet differ by > 5 km s-1 for {target['hostname']}')
        rv = cat_rv

    fluxcols = ['f(C IV)6', 'f(Si IV)6', 'f(Ly a)6,7']
    Fc4, Fsi4, Flya = [target[key] * u.Unit('erg s-1 cm-2') for key in fluxcols]
    yc4 = flare_line(w, 1548.2*u.AA, Fc4, 0.2*u.AA)
    ysi4 = flare_line(w, 1393.8*u.AA, Fsi4, 0.2*u.AA)
    rv_offset = (rv - target['ism_radv']) * u.km/u.s

    ybb = normed_bb(w, target['U_flare'])

    for Nh in Nh_cols:
        ylya = flare_lya_with_ISM(w, Flya, rv_offset, Nh)

        y = ybb + yc4 + ysi4 + ylya

        name = f"{target['hostname']}_Nh{Nh.value:.0e}.dat"
        fullpath = paths.selection_outputs / 'isr_flare_spectra' / name
        data = np.array((w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1'))).T
        np.savetxt(str(fullpath), data)


#%% load and plot

name = 'TOI-4556'
Nh_load = 1e17
folder = paths.selection_outputs / 'isr_flare_spectra'
file, = folder.glob(f'*{name}*{Nh_load:.0e}.dat')
ww, ff = np.loadtxt(str(file)).T

plt.figure()
plt.plot(ww, ff)
plt.title(f'{name} Nh={Nh_load}')
plt.yscale('log')