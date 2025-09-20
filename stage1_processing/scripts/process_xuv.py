import os
import re
import shutil as sh

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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


#%% target list, batch mode?

targets = target_lists.eval_no(1)
batch_mode = True
care_level = 0


#%% paths and tables

folder_phoenix = paths.inbox / '2025-07-22 xuv phoenix'
folder_dem = paths.inbox / '2025-07-16 xuv dem'

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


# %% move to next target

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
    R = props['st_rad'] * u.Rsun
    d = props['sy_dist'] * u.pc
    Teff = props['st_teff'] * u.K

    xfile, = targfolder.rglob(f'{target}*xray-recon*')
    xspec = fits.getdata(xfile)
    xw, xdw, xf = xspec['wavelength'], xspec['bin_width'], xspec['flux']
    isort = np.argsort(xw)
    xw, xdw, xf = [ary[isort] for ary in [xw, xdw, xf]]


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
        spec.rename_columns(('Wavelength', 'Flux_Density'), ('wavelength', 'flux'))
        spec['flux'] *= (R/d).to_value('')**2
        vspecs.append(spec)

        legname = ' '.join(config.split('-')[:2])
        legend_names.append(config)


#%% load in and standardize girish's spectra

    targname_girish = re.sub(r'(?!k2)([a-z])(\d)', r'\1_\2', target)
    targname_girish = re.sub(r'[a-z]$', '', targname_girish)
    dem_file, = folder_dem.rglob(f'spectrum*{targname_girish}*.fits')

    config = dem_file.name.split('_')[-1][:-5]
    newname = f'{target}.dem-{config}-{{}}.na.na.na.xuv-recon.fits'

    spec = table.Table.read(dem_file)
    spec.rename_column('Wavelength', 'wavelength')
    mask = spec['wavelength'] < 1100
    spec = spec[mask]

    # nominal
    spec_nominal = spec[['wavelength', 'Flux_density']]
    spec_nominal.rename_column('Flux_density', 'flux')
    vspecs.append(spec_nominal)
    filenames.append(newname.format('nominal'))
    legend_names.append('DEM nominal')

    # 16th
    spec_lo = spec[['wavelength']]
    spec_lo['flux'] = spec_nominal['flux'] - spec['Lower_Error_16']
    vspecs.append(spec_lo)
    filenames.append(newname.format('lobound'))
    legend_names.append('DEM lower')

    # 84th
    spec_hi = spec[['wavelength']]
    spec_hi['flux'] = spec_nominal['flux'] + spec['Upper_Error_84']
    vspecs.append(spec_hi)
    filenames.append(newname.format('hibound'))
    legend_names.append('DEM upper')


#%% splice xuv spectra

    _legend_names = legend_names
    legend_names = []
    for vspec, newname, legname in zip(vspecs, filenames, _legend_names):
        vw, vf = vspec['wavelength'].value, vspec['flux'].value

        # initial plot
        fig, ax = setup_plot()
        utils.step_mids(vw, vf, alpha=0.5, ax=ax)
        utils.step_mids(xw, xf, bin_widths=xdw*2, alpha=0.5, ax=ax)
        plt.title(newname)
        plt.ylim(np.min(vf)/10)

        # pick splice location
        default = 200.
        print('Click location for splice. Last click trumps. Click off plot when done.\n'
              f'No clicks or clicking below 100 will use default value of {default}.')
        xy = utils.click_coords(fig)
        if len(xy):
            wsplice = xy[-1][0]
            if wsplice < 100:
                wsplice = default
        else:
            wsplice = default

        # do the splice
        vwbins = utils.mids2bins(vw)
        assert np.all(vwbins[1:] > vwbins[:-1])
        xwbins = utils.mids2bins(xw, xdw)
        assert np.all(xwbins[1:] > xwbins[:-1])
        xmask = xwbins[:-1] < wsplice
        vmask = vwbins[1:] > wsplice
        wbins = np.hstack((xwbins[:-1][xmask], vwbins[1:][vmask]))
        wbins = np.insert(wbins, sum(xmask), wsplice)
        assert np.all(wbins[1:] > wbins[:-1])
        f = np.hstack((xf[xmask], vf[vmask]))

        # plot splice location and spec
        ylo = np.min(f[wbins[1:] > 100]) * 0.5
        plt.autoscale()
        _, yhi = plt.ylim(ylo, None)
        plt.vlines(wsplice, ylo, yhi, ls='--', lw=2, color='0.5')
        utils.step_mids(vw[vmask], vf[vmask], color='C0', ax=ax)
        utils.step_mids(xw[xmask], xf[xmask], color='C1', bin_widths=xdw[xmask]*2, ax=ax)

        # save splice plot
        plt.autoscale()
        plt.ylim(ylo, None)
        plt.tight_layout()

        # save spliced spectrum
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

    utils.query_next_step(batch_mode, care_level, 1)

#%% close plots

    plt.close('all')


#%% Lya-based XUV

    Flya = pcutils.get_intrinsic_lya_flux(target)
    Flya_1au = Flya * (d/u.au)**2
    w, dw, f = empirical.EUV_Linsky14(Flya_1au.to_value(''), Teff.to_value('K'), return_spec=True)
    f *= (u.au/d).to_value('')**2
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
    if props['st_rotp']:
        Prot = getprop('st_rotp')
        print(f'{target} has measured Prot of {Prot:.1f}')
        M = getprop('st_mass')
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
    else:
        print(f'{target} has no measured Prot')


#%% comparison plot

    fig, ax = setup_plot()
    lns = []
    n = len(final_specs)
    for i, spec in enumerate(final_specs):
        ln, = utils.step_mids(spec['wavelength'], spec['flux'], bin_widths=spec['bin_width'],
                              ax=ax, zorder=i, alpha=0.5)
        old_edges = utils.mids2bins(spec['wavelength'].quantity, spec['bin_width'].quantity)
        f_3bin = utils.rebin(empirical.mors_bin_edges, old_edges, spec['flux'])
        ln3 = plt.errorbar(mors_spec['wavelength'], f_3bin.to('erg s-1 cm-2 AA-1'),
                           xerr=mors_spec['bin_width']/2, marker='none',
                           elinewidth=3, color=ln.get_color(), linestyle='none', zorder=n+i)
        lns.append(ln)

    plt.legend(lns, legend_names)
    ys = [ln.get_data()[1] for ln in ax.get_lines()]
    yall = np.hstack(ys)
    ylo = np.percentile(yall, 5)/2
    yhi = np.max(yall)*2
    plt.ylim(ylo, yhi)
    plt.tight_layout()
    pngname = f'{target}.xuv-comparison-plot.png'
    fig.savefig(reconfolder / pngname)


#%% flag outliers we shouldn't include in outflow modeling
    pass

    # print fuv line msmts so I know what XUV was based on
    fuv_line_file, = targfolder.glob('*line-flux-table*')
    fuv_line_tbl = table.Table.read(fuv_line_file)
    mask = fuv_line_tbl['flux'].mask
    fuv_line_tbl['name wave flux error source'.split()][~mask].pprint(-1,-1)

    # make a disposition table
    configs = [name.split('.')[1] for name in filenames]
    xuv_disposition_table = table.Table((configs,), names=('source',))
    xuv_disposition_table['usable'] = True
    xuv_disposition_table.pprint(-1,-1)
    xdt = xuv_disposition_table

    # xy = utils.click_coords(fig) # kludge to make sure I can see figure
    # while True:
    #     sub = input('Any spectra that should be flagged unusable? If so, give subsrting of source.\n'
    #                 'Hit enter if none. Prompt will loop until an empty answer is given.')
    #     if sub == '':
    #         break
    #     mask = np.char.count(xdt['source'].astype(str), sub) > 0
    #     if sum(mask) > 1:
    #         print("More than one match. Try again.")
    #     elif sum(mask) == 0:
    #         print("No matches. Try again.")
    #     else:
    #         xdt['usable'][mask] = False
    #
    # utils.query_next_step(batch_mode, care_level, 1)


#%% save disposition table

    xuv_disposition_table.write(reconfolder / f'{target}.xuv-disposition-table.ecsv', overwrite=True)


#%% close plots

    plt.close('all')


#%% loop close

  except StopIteration:
    break


#%% move files to staging folder

staging_folder = paths.data / 'packages/2025-06-16.stage2.eval1.staging_area/xuv_reconstructions'

# os.mkdir(staging_folder)
# os.mkdir(staging_folder / 'comparison_plots')

for target in targets:
    reconfolder = paths.target_data(target) / 'reconstructions'
    files = reconfolder.glob('*xuv-recon.fits')
    for file in files:
        sh.copy(file, staging_folder / file.name)
    compfile, = reconfolder.glob('*xuv-comparison*png')
    sh.copy(compfile, staging_folder / 'comparison_plots' / compfile.name)
