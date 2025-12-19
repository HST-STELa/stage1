import os
import re
import shutil as sh
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.ion()
plt.rcParams['savefig.dpi'] = 150

from astropy import units as u
from astropy.io import fits
from astropy import table

import paths
import utilities as utils
import catalog_utilities as catutils
import empirical
from stage1_processing import preloads
from stage1_processing import target_lists
from stage1_processing import processing_utilities as pcutils


#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

# matplotlib.use('Qt5Agg')
matplotlib.use('agg'); plt.ioff()

targets = target_lists.eval_no(2)
targets.remove('v1298tau')
batch_mode = True
care_level = 0
wsplice_phx = 292.
wsplice_dem = 100.

folder_phoenix = paths.inbox / '2025-10-28 xuv phoenix'
folder_dem = paths.inbox / '2025-10-30 xuv dem'
staging_folder = paths.data / 'packages/2025-09-26.stag2.eval2.staging_area/xuv_reconstructions'


#%% paths and tables

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    hosts = preloads.hosts.copy()
hosts.add_index('tic_id')


#%% boilerplate plot setup

def setup_plot():
    fig, ax = plt.subplots(1,1, figsize=(9, 4))
    plt.yscale('log')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux Density (cgs)')
    return fig, ax

def plot_and_save(spec, basename, ylim=None):
    # save spec
    spec.write(reconfolder / basename, overwrite=True)

    # save plot of spliced spec
    fig, ax = setup_plot()
    utils.step_mids(spec['wavelength'], spec['flux'], bin_widths=spec['bin_width'], ax=ax)
    plt.ylim(ylim)
    plt.title(basename)
    plt.tight_layout()
    pngname = basename.replace('.fits', '.plot.png')
    fig.savefig(reconfolder / pngname)


#%% SKIP? set up batch processing (skip if not in batch mode)

if batch_mode:
    print("When 'Continue?' prompts appear, hit enter to continue, anything else to break out of the loop.")

itertargets = iter(targets)
k = 0
while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:


#%% move to next target

    k += 1
    target = next(itertargets)

    headline = f"{k}/{len(targets)}: {target.upper()}"
    print(f"{'='*len(headline)}\n"
          f"{headline}\n"
          f"{'='*len(headline)}")


#%% ids and paths

    tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
    targfolder = paths.target_hst_data(target) / '..'
    targfolder = targfolder.resolve()
    reconfolder = targfolder / 'reconstructions'


#%% load properties and x-ray spectrum

    props = hosts.loc[tic_id]
    R = props['st_rad']
    d = props['sy_dist']
    Teff = props['st_teff']

    xfile, = targfolder.rglob(f'{target}*xray-recon*')
    xspec = fits.getdata(xfile)
    try:
        xw, xdw, xf = xspec['wavelength'], xspec['bin_width'], xspec['flux']
    except KeyError:
        xw, xdw, xf = xspec['Wave'], xspec['bin_width']*2, xspec['Flux']

    isort = np.argsort(xw)
    xw, xdw, xf = [ary[isort] for ary in [xw, xdw, xf]]

    # xspec christian used produces rounding errors that can result in duplicate points in xw
    # so reconstruct a more accurate array
    if not np.all(np.diff(xw) > 0):
        xr = 1/xw
        xi = np.arange(len(xr))
        xp = np.polyfit(xi[[0,-1]], xr[[0,-1]], 1)
        xr_new = np.polyval(xp, xi)
        xw = 1/xr_new

    xwbins_test = utils.mids2bins(xw, xdw)
    assert np.all(np.diff(xwbins_test) > 0)
    xw_test = utils.midpts(xwbins_test)
    assert np.allclose(xw, xw_test, rtol=1e-5)


#%% initialize some lists to store various specs

    vspecs = []
    filenames = []
    final_specs = []
    legend_names = []


#%% load in and standardize sarah's specs

    phoenix_files = list(folder_phoenix.glob(f'{target}*.fits'))
    for file in phoenix_files:
        filename = file.name
        filename = re.sub(r'(=\d+)\.(\d)', r'\1pt\2', filename)
        pieces = filename.split('.')
        pieces = [piece.lower() for piece in pieces]
        config = '-'.join(pieces[2:-1])
        config = config.replace('_', '-')
        newname = f'{target}.{config}.na.na.na.xuv-recon.fits'
        filenames.append(newname)

        spec = table.Table.read(file, 1)
        catutils.add_masks(spec, inplace=True)
        spec.rename_columns(('Wavelength', 'Flux_Density'), ('wavelength', 'flux'))
        spec['flux'] = spec['flux'] * (R/d).to_value('')**2
        vspecs.append(spec)

        legname = ' '.join(config.split('-')[:2])
        legend_names.append(config)


    #%% load in and standardize girish's spectra

    targname_girish = re.sub(r'(?!k2)([a-z])(\d)', r'\1*\2', target)
    targname_girish = re.sub(r'[_-]', r'*', targname_girish)
    targname_girish = re.sub(r'[a-z]$', '', targname_girish)
    if targname_girish.endswith('*'):
        targname_girish = targname_girish[:-1]
    dem_files = list(folder_dem.rglob(f'*{targname_girish}*.fits'))
    dem_file, = [file for file in dem_files if 'xray' not in file.name]

    config = dem_file.name.split('_')[-1][:-5]
    newname = f'{target}.dem-{{}}.na.na.na.xuv-recon.fits'

    spec = table.Table.read(dem_file, hdu=1)
    colnames = spec.colnames
    newcolnames = [name.lower() for name in colnames]
    spec.rename_columns(colnames, newcolnames)
    mask = spec['wavelength'] < 1100
    spec = spec[mask]

    if 'flux_density' in spec.colnames:
        spec.rename_column('flux_density', 'flux')
    if 'lower_error_16' not in spec.colnames:
        spec['lower_error_16'] = spec['error']
        spec['upper_error_84'] = spec['error']

    # nominal
    spec_nominal = spec[['wavelength', 'flux']]
    vspecs.append(spec_nominal)
    filenames.append(newname.format('nominal'))
    legend_names.append('DEM nominal')

    # 16th
    spec_lo = spec[['wavelength']]
    spec_lo['flux'] = spec_nominal['flux'] - spec['lower_error_16']
    vspecs.append(spec_lo)
    filenames.append(newname.format('lobound'))
    legend_names.append('DEM lower')

    # 84th
    spec_hi = spec[['wavelength']]
    spec_hi['flux'] = spec_nominal['flux'] + spec['upper_error_84']
    vspecs.append(spec_hi)
    filenames.append(newname.format('hibound'))
    legend_names.append('DEM upper')


#%% splice xuv spectra

    _legend_names = legend_names
    legend_names = []
    for vspec, newname, legname in zip(vspecs, filenames, _legend_names):
        vw, vf = vspec['wavelength'].value, vspec['flux'].value

        if 'phoenix' in newname:
            wsplice = wsplice_phx
        elif 'dem' in newname:
            wsplice = wsplice_dem
        else:
            raise ValueError

        # some spectra have gaps, if so, use the midpoint for splicing
        if wsplice > xw[-1]:
            if xw[-1] > vw[0]:
                wsplice = xw[-1] + xdw[-1]/2
            else:
                # make the splice between where the two spectra end
                wsplice = (xw[-1] + vw[0])/2

                # add pts that average the flux near the end of each spectrum to avoid
                # an unusually high or low point resulting in an unrealistically high
                # or low flux in the gap
                dw_gap = vw[0] - xw[-1]
                dw_avg = dw_gap / 2

                mask = vw < vw[0] + dw_gap/2
                F = np.trapz(vf[mask], vw[mask])
                vf0 = F/dw_avg
                vw = np.append(vw[:1] - np.diff(vw)[0]/2, vw)
                vf = np.append([vf0], vf)

                mask = xw > xw[-1] - dw_avg
                F = np.trapz(xf[mask], xw[mask])
                xf1 = F/dw_avg
                xw = np.append(xw, xw[-1] + xdw[-1]/2)
                xdw = np.append(xdw, xdw[-1])
                xf = np.append(xf, xf1)

        # do the splice
        vwbins = utils.mids2bins(vw)
        xwbins = utils.mids2bins(xw, xdw)
        assert np.all(vwbins[1:] > vwbins[:-1])
        assert np.all(xwbins[1:] > xwbins[:-1])
        xmask = xwbins[:-1] < wsplice
        vmask = vwbins[1:] > wsplice
        wbins = np.hstack((xwbins[:-1][xmask], vwbins[1:][vmask]))
        wbins = np.insert(wbins, sum(xmask), wsplice)
        assert np.all(wbins[1:] > wbins[:-1])
        f = np.hstack((xf[xmask], vf[vmask]))

        # plot splice location and spec
        fig, ax = setup_plot()

        utils.step_mids(vw, vf, alpha=0.5, ax=ax)
        utils.step_mids(xw, xf, bin_widths=xdw*2, alpha=0.5, ax=ax)
        utils.step_mids(vw[vmask], vf[vmask], color='C0', ax=ax)
        utils.step_mids(xw[xmask], xf[xmask], color='C1', bin_widths=xdw[xmask]*2, ax=ax)

        ylo = np.min(f[wbins[1:] > 100]) * 0.5
        if ylo <= 0:
            ylo = None
        plt.autoscale()
        ylo, yhi = plt.ylim(ylo, None)
        plt.vlines(wsplice, ylo, yhi, ls='--', lw=2, color='0.5')

        plt.title(newname)

        # groom plot and pause to show
        plt.autoscale()
        plt.ylim(ylo, None)
        plt.tight_layout()
        _ = utils.click_coords(fig)

        # make spectrum
        w = utils.midpts(wbins)
        dw = np.diff(wbins)
        splicetbl = table.Table(
            (w, dw, f),
            names='wavelength bin_width flux'.split(),
            units='AA,AA,erg s-1 cm-2 AA-1'.split(',')
        )
        final_specs.append(splicetbl)
        legend_names.append(legname)

        # save plot of full spectrum
        plot_and_save(splicetbl, newname, ylim=(ylo, None))

        _ = utils.click_coords()

    utils.query_next_step(batch_mode, care_level, 1)

#%% close plots

    plt.close('all')


#%% Lya-based XUV

    Flya = pcutils.get_intrinsic_lya_flux(target)
    Flya_1au = Flya * (d/u.au)**2
    w, dw, f = empirical.EUV_Linsky14(Flya_1au.to_value(''), Teff.to_value('K'), return_spec=True)
    f = f * (u.au/d).to_value('')**2
    f = np.array(f)
    linsky_spec = table.Table(
        (w, dw, f),
        names = 'wavelength bin_width flux'.split(),
        units = 'AA,AA,erg s-1 cm-2 AA-1'.split(',')
    )
    final_specs.append(linsky_spec)
    linsky_name = f'{target}.empirical-lya-linsky14.na.na.na.xuv-recon.fits'
    filenames.append(linsky_name)
    plot_and_save(linsky_spec, linsky_name)
    legend_names.append('Lya Empirical')


#%% Prot-based XUV

    getprop = lambda key: catutils.get_quantity_flexible(key, props, hosts)
    try:
        Prot = getprop('st_rotp')
        if np.ma.is_masked(Prot):
            print(f'{target} has no measured Prot, no Prot-based XUV possible')
            raise ValueError
        M = getprop('st_mass')
        if M < 0.1*u.Msun:
            print(f'{target} has mass < 0.1 Msun, no Prot-based XUV possible')
            raise ValueError
        print(f'{target} has measured Prot of {Prot:.1f}')
        age = getprop('st_age')
        if not np.ma.is_masked(age):
            print(f'{target} has age estimate of {age:.0f}')
        else:
            age = empirical.age_from_Prot_johnstone21(Prot, M)
            print(f'{target} age estimate made from Mors of {age:.0f}')
        w, dw, F = empirical.XUV_from_Prot_johnstone21(Prot, M, age)
        f = F / (4*np.pi*d**2)
        f = f.to('erg s-1 cm-2 AA-1')

        mors_spec = table.Table(
            (w.to('AA'), dw.to('AA'), f),
            names = 'wavelength bin_width flux'.split()
        )

        final_specs.append(mors_spec)
        mors_name = f'{target}.empirical-rotation-johnstone21.na.na.na.xuv-recon.fits'
        filenames.append(mors_name)
        plot_and_save(mors_spec, mors_name)
        legend_names.append('Prot Empirical')
    except ValueError:
        pass


#%% comparison plot

    fig, ax = setup_plot()
    lns = []
    n = len(final_specs)
    w3bins = empirical.mors_bin_edges.to('AA')
    w3 = utils.midpts(w3bins)
    dw3 = np.diff(w3bins)
    for i, spec in enumerate(final_specs):
        ln, = utils.step_mids(spec['wavelength'], spec['flux'], bin_widths=spec['bin_width'],
                              ax=ax, zorder=i, alpha=0.5)
        old_edges = utils.mids2bins(spec['wavelength'].quantity, spec['bin_width'].quantity)
        dmask = w3bins > old_edges[0]
        mask = dmask[:-1]
        f_3bin = utils.rebin(w3bins[dmask], old_edges, spec['flux'].quantity)
        ln3 = plt.errorbar(w3[mask], f_3bin.to('erg s-1 cm-2 AA-1'),
                           xerr=dw3[mask]/2, marker='none',
                           elinewidth=3, color=ln.get_color(), linestyle='none', zorder=n+i)
        lns.append(ln)

    plt.legend(lns, legend_names)
    ys = [ln.get_data()[1] for ln in ax.get_lines()]
    ylos = [y[int(len(y)/2):] for y in ys]
    yall = np.hstack(ys)
    ylos = np.hstack(ylos)
    ylo = np.percentile(ylos, 5)/2
    yhi = np.max(yall)*2
    plt.ylim(ylo, yhi)
    plt.title(target)
    plt.tight_layout()
    pngname = f'{target}.xuv-comparison-plot.png'
    fig.savefig(reconfolder / pngname)

    _ = utils.click_coords(fig)


#%% close plots

    plt.close('all')


#%% loop close

  except StopIteration:
    break


#%% move files to staging folder

if not staging_folder.exists():
    os.mkdir(staging_folder)
    os.mkdir(staging_folder / 'comparison_plots')

for target in targets:
    reconfolder = paths.target_data(target) / 'reconstructions'
    files = reconfolder.glob('*xuv-recon.fits')
    dem, phx = False, False
    for file in files:
        if 'dem' in file.name:
            dem = True
        elif 'phoenix' in file.name:
            phx = True
        else:
            continue
        sh.copy(file, staging_folder / file.name)
    if not (dem or phx):
        raise ValueError
    compfile, = reconfolder.glob('*xuv-comparison*png')
    sh.copy(compfile, staging_folder / 'comparison_plots' / compfile.name)