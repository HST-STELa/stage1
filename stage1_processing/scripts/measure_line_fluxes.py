import warnings
from pprint import pprint
from pathlib import Path
from copy import copy

import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import patches

import database_utilities as dbutils
import paths
import utilities as utils
import catalog_utilities as catutils

from stage1_processing import processing_utilities as pcutils
from stage1_processing import target_lists
from stage1_processing import preloads


#%% options

batch_mode = True
care_level = 0
redo_bands = False

targets = target_lists.eval_no(1) + target_lists.eval_no(2)
shift_errors = True


#%% general setup

linecat = table.Table.read(paths.uv_lines / 'fuv_line_list.ecsv')
linewaves = linecat['wave'].value

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    target_table = preloads.hosts.copy()
target_table.add_index('tic_id')


#%% target iterator

itertargets = iter(targets)


#%% set up batch processing (can skip if not in batch mode)

while True:
  # I'm being sneaky with 2-space indents here because I want to avoid 8 space indents on the cells
  if not batch_mode:
    break

  try:


    # %% move to next target

    target = next(itertargets)


    #%% target details

    m = len(target)
    print('='*m)
    print(target.upper())
    print('='*m)

    tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
    hostprops = target_table.loc[tic_id]
    data_folder = paths.target_data(target)


    #%% configs

    configs = ( # keep this here in case I rerun a single case with fewer configs
        'stis-g140m',
        'stis-g140l',
        'stis-e140m',
        'cos-g130m',
        'cos-g160m',
        'cos-g140l'
    )


    # #%% loop
    #
    # for config in configs:
    #
    #     #%% find files
    #
    #     files = dbutils.find_coadd_or_x1ds(target, out_of_transit_coadd=False, instruments=config, directory=data_folder)
    #
    #     #%% skip if none
    #     if len(files) == 0:
    #         print(f'\n{config}: no files\n')
    #         continue
    #     else:
    #         print(f'\n{config}\n')
    #
    #
    #     #%% next file
    #
    #     for file in files:
    #
    #         #%% load data
    #
    #         spec = fits.getdata(file, 1)
    #         spec.wearth_data, = spec['wavelength']
    #         spec.f, = spec['flux']
    #         spec.e, = spec['error']
    #
    #
    #         #%% mitigate inflated errors
    #         """Also hopefully in the future I will fix coadd script so that it uses poisson errors and avoids the annoying error
    #         inflation issue."""
    #
    #         if shift_errors:
    #             var_mod = utils.shift_floor_to_zero(spec.e**2, window_size=50)
    #             spec.emod = np.sqrt(np.abs(var_mod))
    #         else:
    #             spec.emod = spec.e
    #
    #
    #         #%% see if there is an existing line table to draw bands from
    #
    #         linefile = str(file).replace('.fits', '.line_fluxes.ecsv')
    #         linefile = Path(linefile)
    #
    #         #%% select bands by hand if needed/desired
    #
    #         if redo_bands or not linefile.exists():
    #             newbands = True
    #
    #
    #             #%% plot data
    #             fig = plt.figure(figsize=(14,5))
    #             plt.title(file.name)
    #             plt.step(spec.wearth_data, spec.f, where='mid')
    #             plt.step(spec.wearth_data, spec.emod, where='mid', lw=0.5, color='C0', alpha=0.5)
    #             plt.tight_layout()
    #             ax, = fig.get_axes()
    #
    #             print('Zoom to a good range, then click off the plot.')
    #             _ = utils.click_coords(fig)
    #             xhome = plt.xlim()
    #             yhome = plt.ylim()
    #             def gohome():
    #                 plt.xlim(xhome)
    #                 plt.ylim(yhome)
    #                 plt.draw()
    #
    #             def plotspans(x, **kws):
    #                 if len(x) % 2 == 1:
    #                     print('Odd number of clicks. Even number required. Try again.')
    #                     return []
    #                 bands = np.reshape(x, (-1,2))
    #                 artists = []
    #                 for a, b in bands:
    #                     artist = patches.Rectangle((a, 0), (b-a), yhome[1], **kws)
    #                     ax.add_patch(artist)
    #                     artists.append(artist)
    #                 return artists
    #
    #
    #             #%% ranges of viable data
    #
    #             print('Intervals of viable data:')
    #             edgeplot = lambda x: [plt.plot([xx]*2, yhome, color='0.5', lw=2)[0] for xx in x]
    #             edges, _ = utils.click_n_plot(fig, edgeplot)
    #             edges = np.reshape(edges, (-1,2))
    #
    #             utils.query_next_step(batch_mode, care_level, 3, msg='Continue to next step?')
    #
    #
    #             #%% plot the lines that are within
    #
    #             rv = hostprops['st_radv'].filled(0)
    #             within = (linewaves[:,None] >= edges[:,0]) & (linewaves[:,None] <= edges[:,1])
    #             within = within.any(axis=1)
    #             obslines = linecat[within].copy()
    #             for line in obslines:
    #                 w0_star = line['wave']
    #                 w0_earth = (rv/const.c + 1) * w0_star
    #                 plt.plot([w0_earth]*2, yhome, color='0.5', lw=0.5)
    #                 plt.annotate(line['name'] + ' ', xy=(w0_earth, 0),
    #                              rotation='vertical', ha='center', va='top')
    #
    #
    #             #%% pick bands for the lines
    #
    #             print('Line bands: ')
    #             print('\tFor blends, cover the full band for the first line, then click the same spots '
    #                   '(i.e. give a zero-width band) for other lines in the blend.')
    #             wbuf = 2 if config.endswith('m') else 15
    #             bandplot = lambda x: plotspans(x, color='C2', alpha=0.3, lw=0)
    #             MCfloat = lambda: table.MaskedColumn(length=len(obslines), dtype=float, mask=True, format='.2f')
    #             obslines['wa'] = MCfloat()
    #             obslines['wb'] = MCfloat()
    #             for i, line in enumerate(obslines):
    #                 print(f'{line['name']} {line['wave']}')
    #                 xlim = line['wave'] - wbuf, line['wave'] + wbuf
    #                 plt.xlim(xlim)
    #                 inplt = (spec.wearth_data > xlim[0]) & (spec.wearth_data < xlim[1])
    #                 ymx = spec.f[inplt].max()
    #                 plt.ylim(-0.05*ymx, 1.5*ymx)
    #                 plt.draw()
    #                 band, _ = utils.click_n_plot(fig, bandplot)
    #                 while len(band) != 2:
    #                     print('Either too few or too many clicks. Specify a single band (two clicks).')
    #                     band, _ = utils.click_n_plot(fig, bandplot)
    #                 if np.diff(band) < np.diff(spec.wearth_data, axis=0)[0]:
    #                     continue
    #                 obslines['wa'][i] = band[0]
    #                 obslines['wb'][i] = band[1]
    #             gohome()
    #
    #             utils.query_next_step(batch_mode, care_level, 3, msg='Continue to next step?')
    #
    #
    #             #%% pick continuum ranges
    #
    #             print('Continuum bands:')
    #             chunksize = 50
    #             overlap = 10
    #             cplot = lambda x: plotspans(x, color='C9', alpha=0.3, lw=0)
    #             cband_list = []
    #             wa = np.min(edges)
    #             while wa < np.max(edges):
    #                 wb = wa + chunksize
    #                 plt.xlim(wa, wb)
    #                 plt.draw()
    #                 cbands, _ = utils.click_n_plot(fig, cplot)
    #                 cbands = np.reshape(cbands, (-1,2))
    #                 cband_list.append(cbands)
    #                 wa += chunksize - overlap
    #             cbands = np.vstack(cband_list)
    #
    #             obslines.meta['continuum bands'] = cbands
    #             gohome()
    #
    #             utils.query_next_step(batch_mode, care_level, 3, msg='Continue to next step?')
    #
    #
    #             #%% pick fitting intervals
    #
    #             print('Fitting intervals:')
    #             chunksize = 200 if 'g140l' in config else 50
    #             overlap = chunksize / 5
    #             vplot = lambda x: plotspans(x, color='0.5', alpha=0.1, lw=1)
    #             intvls_list = []
    #             wa = np.min(edges)
    #             while wa < np.max(edges):
    #                 wb = wa + chunksize
    #                 plt.xlim(wa, wb)
    #                 plt.draw()
    #                 intvls, _ = utils.click_n_plot(fig, vplot)
    #                 intvls = np.reshape(intvls, (-1,2))
    #                 intvls_list.append(intvls)
    #                 wa += chunksize - overlap
    #             intvls = np.vstack(intvls_list)
    #
    #             obslines.meta['continuum fitting intervals'] = intvls
    #             gohome()
    #
    #             utils.query_next_step(batch_mode, care_level, 3, msg='Continue to next step?')
    #
    #
    #         #%% else load bands
    #
    #         else:
    #             print('Using bands from preexisting line flux table.')
    #             newbands = False
    #             obslines = table.Table.read(linefile)
    #             cbands = obslines.meta['continuum bands']
    #             intvls = obslines.meta['continuum fitting intervals']
    #
    #
    #         #%% continuum estimates
    #
    #         cfluxes, cerrors = [], []
    #         clns = []
    #         for intvl in intvls:
    #             cmask = (cbands[:,0] > intvl[0]) & (cbands[:,0] < intvl[1])
    #             fitbands = cbands[cmask]
    #
    #             Fs, Es, dws = [], [], []
    #             for band in fitbands:
    #                 wave = np.sum(band)/2
    #                 dw = np.diff(band)[0]
    #                 m = (spec.wearth_data > band[0]) & (spec.wearth_data < band[1])
    #                 n = np.sum(m)
    #                 if n == 0:
    #                     continue
    #                 elif n == 1:
    #                     F, E = spec.f[m][0], spec.emod[m][0]
    #                 else:
    #                     F, E = utils.flux_integral(spec.wearth_data[m], spec.f[m], e=spec.emod[m])
    #                 Fs.append(F)
    #                 Es.append(E)
    #                 dws.append(dw)
    #             Es = np.array(Es)
    #             dw_sum = np.sum(dws)
    #             cflux = np.sum(Fs)/dw_sum
    #             cerr = np.sqrt(np.sum(Es**2))/dw_sum
    #             cfluxes.append(cflux)
    #             cerrors.append(cerr)
    #
    #             if newbands:
    #                 wplt = np.array((fitbands.min(), fitbands.max()))
    #                 cln, = plt.plot(wplt, [cflux]*2, '-', color='C1', alpha=0.5)
    #                 clns.append(cln)
    #         obslines.meta['continuum fluxes'] = cfluxes
    #         obslines.meta['continuum flux errors'] = cerrors
    #
    #
    #         #%% compute line fluxes
    #
    #         MCfloat = lambda: table.MaskedColumn(length=len(obslines), dtype=float, mask=True, format='.2e')
    #         obslines['flux'] = MCfloat()
    #         obslines['error'] = MCfloat()
    #         obslines['contflux'] = MCfloat()
    #         obslines['conterror'] = MCfloat()
    #         for i, line in enumerate(obslines):
    #             if np.ma.is_masked(line['wa']):
    #                 obslines['contflux'][i] = 0
    #                 obslines['conterror'][i] = 0
    #                 obslines['flux'][i] = 0
    #                 obslines['error'][i] = 0
    #                 continue
    #             dw = line['wb'] - line['wa']
    #             wmid = np.array([(line['wa'] + line['wb']) / 2])
    #
    #             # continuum flux and uncty
    #             (icont,), = np.where((wmid > intvls[:,0]) & (wmid < intvls[:,1]))
    #             cflux = cfluxes[icont]
    #             cerr = cerrors[icont]
    #             cF = cflux * dw
    #             cE = cerr * dw
    #             obslines['contflux'][i] = cF
    #             obslines['conterror'][i] = cE
    #
    #             # line + cont flux and uncty
    #             m = (spec.wearth_data > line['wa']) & (spec.wearth_data < line['wb'])
    #             F, E = utils.flux_integral(spec.wearth_data[m], spec.f[m], e=spec.emod[m])
    #
    #             # line only flux and uncty
    #             lF = F - cF
    #             lE = np.sqrt(E**2 + cE**2)
    #             obslines['flux'][i] = lF
    #             obslines['error'][i] = lE
    #
    #         with np.errstate(divide='ignore'):
    #             obslines['snr'] = obslines['flux']/obslines['error']
    #             obslines['snr'][obslines['error'] == 0] = 0
    #             obslines['snr'].format = '.1f'
    #
    #
    #         #%% take a gander
    #
    #         obslines.pprint(-1,-1)
    #         utils.query_next_step(batch_mode, care_level, 3, msg='Continue to next step?')
    #
    #
    #         #%% save the table and plot
    #
    #         # save plot with mpl3
    #         if newbands:
    #             htmlfile = str(file).replace('.fits', '.line_band_plot.html')
    #             utils.save_standard_mpld3(fig, htmlfile)
    #
    #         # save table
    #         obslines.sort('wave')
    #         obslines.write(linefile, overwrite=True)
    #
    #     utils.query_next_step(batch_mode, care_level, 2, msg='Continue to next grating?')


    #%% load all the line flux tables, merge cos ones

    line_flux_files = list(data_folder.rglob('*line_fluxes.ecsv'))

    used = []
    fluxtbls = []
    for file in line_flux_files:
        source_files = [file.name]
        config = dbutils.parse_filename(file)['config']
        if file in used:
            continue
        used.append(file)
        line_fluxes = table.Table.read(file)
        line_fluxes.add_index('wave')

        # look for a corresponding g160m/g130m file
        pairs = (('g130m', 'g160m'),
                 ('g160m', 'g130m'))
        otherfile = None
        for this, other in pairs:
            if this in file.name:
                of = [f for f in line_flux_files if other in f.name]
                if of:
                    otherfile, = of
                    break

        if otherfile:
            # merge values from other table into the current table
            used.append(otherfile)
            source_files.append(otherfile.name)
            config += '+' + dbutils.parse_filename(otherfile)['config'][-5:]
            other_fluxes = table.Table.read(otherfile)
            for otherrow in other_fluxes:
                if otherrow['wave'] not in line_fluxes['wave']:
                    line_fluxes.add_row(otherrow)
                else:
                    thisrow = line_fluxes.loc[otherrow['wave']]
                    if (thisrow['flux'] == 0) and (otherrow['flux'] != 0):
                        for key in line_fluxes.colnames[7:]:
                            thisrow[key] = otherrow[key]

        line_fluxes.sort('wave')
        line_fluxes.meta['config'] = config
        line_fluxes.meta['source files'] = source_files
        fluxtbls.append(line_fluxes)


    #%% pick the best

    snr_threshold = 5
    scores, snr_sums = [], []
    for fluxtbl in fluxtbls:
        mask = (fluxtbl['snr'] > snr_threshold)
        if hasattr(fluxtbl['Tform'], 'mask'):
            mask &= ~fluxtbl['Tform'].mask
        if sum(mask) == 0:
            scores.append(0)
            snr_sums.append(0)
            continue
        unq_temps = np.unique(fluxtbl['Tform'][mask].value)
        score = len(unq_temps)
        scores.append(score)
        if score == 0:
            snr_sums.append(0)
            continue
        fluxtbl.add_index('Tform')
        if hasattr(fluxtbl['Tform'], 'mask'):
            _fluxtbl_temp = fluxtbl[~fluxtbl['Tform'].mask]
            unq_temp_rows = _fluxtbl_temp.loc['Tform', unq_temps]
        else:
            unq_temp_rows = fluxtbl.loc['Tform', unq_temps]
        snrs = unq_temp_rows['snr']
        if hasattr(snrs, 'filled'):
            snrs = snrs.filled(0)
        snr_sums.append(np.sum(snrs))
    scores = np.asarray(scores)
    snr_sums = np.asarray(snr_sums)

    # sometimes the same two tables can have the same score, in which case snr_sum will be the decider
    high_score = np.max(scores)
    hiscoring = scores == np.max(scores)
    icontenders, = np.nonzero(hiscoring)
    contender_snr_sums = snr_sums[hiscoring]
    ibest = icontenders[np.argmax(contender_snr_sums)]

    bestfluxes = catutils.add_masks(fluxtbls[ibest], inplace=False)

    bestfluxes.add_index('name')
    bestfluxes['source'] = table.MaskedColumn(dtype=object, length=len(bestfluxes))
    bestfluxes['source'] = bestfluxes.meta['config']
    del bestfluxes.meta['config']


    #%% add in Lya

    lya_recon_file = pcutils.get_lya_recon_file(target)
    Fs = pcutils.get_intrinsic_lya_flux(target, return_16_50_84=True)
    catutils.scrub_indices(bestfluxes)
    catutils.add_masked_row(bestfluxes)
    bestfluxes['key'][-1] = 'lya'
    bestfluxes['name'][-1] = 'Lya'
    bestfluxes['atom'][-1] = 'H'
    bestfluxes['ionztn'][-1] = 0
    bestfluxes['wave'][-1] = 1215.67
    bestfluxes['blend'][-1] = False
    bestfluxes['flux'][-1] = Fs[1]
    bestfluxes['error'][-1] = (Fs[2] - Fs[0])/2
    bestfluxes['source'][-1] = 'reconstruction'
    bestfluxes.meta['Lya source file'] = lya_recon_file.name


    #%% augment with line-line correlations as needed

    basecat = table.Table.read(paths.uv_lines / 'fuv_line_list.ecsv')
    basecat.add_index('name')

    dist = hostprops['sy_dist']

    min_snr = 3

    corrtbl = table.Table.read(paths.uv_lines / 'line-line_correlations.ecsv')
    corrtbl.add_index('x')
    corrtbl.add_index('y')
    corrlines = np.unique(corrtbl['x']).tolist()
    corrlines.remove('Lya')
    corrtemps = [np.mean(basecat.loc[name]['Tform']) for name in corrlines]
    corrlines.append('Lya')
    corrtemps.append(3.5)

    corrlines = np.asarray(corrlines)
    corrtemps = np.asarray(corrtemps)
    tempdict = dict(zip(corrlines, corrtemps))

    bestfluxes.add_index('name')
    def fluxsum(line):
        linetbl = bestfluxes.loc['name', line]
        assert np.max(linetbl['wave']) - np.min(linetbl['wave']) < 10
        F = np.sum(linetbl['flux'])
        E = np.sqrt(np.sum(linetbl['error'] ** 2))
        return F, E

    # identify lines with usable measurements
    usable = []
    for line in corrlines:
        if line == 'Lya':
            usable.append(True)
            continue
        result = False
        if line in bestfluxes['name']:
            F, E = fluxsum(line)
            if E > 0 and F/E > snr_threshold:
                result = True
        usable.append(result)
    usable = np.asarray(usable)

    # fill lines that don't
    usable_lines = corrlines[usable]
    usable_temps = corrtemps[usable]
    for line in corrlines[~usable]:
        # add rows if not present
        if line not in bestfluxes['name']:
            missing_rows = copy(basecat.loc[line])
            if isinstance(missing_rows, table.Table.Row):
                missing_rows = table.Table(rows=missing_rows)
            n = len(missing_rows)
            add_cols = set(bestfluxes.colnames) - set(missing_rows.colnames)
            for col in add_cols:
                missing_rows[col] = table.MaskedColumn(length=n, dtype=bestfluxes[col].dtype, mask=True)
            bestfluxes = table.vstack((bestfluxes, missing_rows))
            bestfluxes.add_index('name')

        # find the usable line with the nearest temp
        dT = np.abs(tempdict[line] - usable_temps)
        iproxy = np.argmin(dT)
        proxy = usable_lines[iproxy]

        # get flux of proxy line
        proxyflux, _ = fluxsum(proxy)
        proxyflux_1au = proxyflux * (dist/u.au)**2
        proxyflux_1au = proxyflux_1au.to_value('')

        # estimate flux
        corr = corrtbl.loc['x', proxy].loc['y', line]
        log10_proxyflux_1au = np.log10(proxyflux_1au)
        log10_estflux_1au = corr['slope'] * log10_proxyflux_1au + corr['intercept']
        estflux_1au = 10**log10_estflux_1au
        estflux = estflux_1au * (u.au/dist)**2
        estflux = estflux.to_value('')

        # put into table
        bestfluxes.sort('wave')
        iline = bestfluxes.loc_indices['name', line]
        for key in bestfluxes.colnames[7:]:
            bestfluxes[key].mask[iline] = True
        bestfluxes['flux'][iline] = 0
        iflux = min(iline) if hasattr(iline, '__iter__') else iline
        bestfluxes['flux'][iflux] = estflux
        bestfluxes['source'][iline] = f'{proxy} correlation'


    #%% groom for outside use

    pass
    # need to scrub indices first
    catutils.scrub_indices(bestfluxes)

    # make sure all tables have the same set of rows
    for row in basecat.copy():
        if row['wave'] not in bestfluxes['wave']:
            catutils.add_masked_row(bestfluxes)
            for col in basecat.colnames:
                bestfluxes[col][-1] = row[col]

    # mask any zero flux rows that don't correspond with a line that has flux
    for row in bestfluxes:
        if row['flux'] == 0:
            likelines = ((bestfluxes['name'] == row['name'])
                         & (np.abs(bestfluxes['wave'] - row['wave']) <= 10))
            if all(bestfluxes['flux'][likelines] == 0):
                for key in bestfluxes.colnames[7:]:
                    bestfluxes[key].mask[likelines] = True

    # get rid of extra or confusing columns, like contflux
    bestfluxes.remove_columns(('contflux', 'conterror', 'Tform', 'key', 'atom', 'ionztn'))
    for key in list(bestfluxes.meta.keys()):
        if key not in ['source files', 'Lya source file']:
            del bestfluxes.meta[key]

    # add some info on provenance to the header
    props = target_table.loc[tic_id]
    def get_prop(key, default):
        val = props[key]
        if np.ma.is_masked(val):
            return default
        else:
            return val
    from math import nan
    bestfluxes.meta['hostname'] = target
    bestfluxes.meta['tic_id'] = tic_id
    bestfluxes.meta['dist'] = float(dist.to_value('pc'))
    bestfluxes.meta['radius'] = float(hostprops['st_rad'].to_value('Rsun').filled(nan))
    bestfluxes.meta['Teff'] = float(hostprops['st_teff'].to_value('K').filled(nan))
    bestfluxes.meta['SpT'] = get_prop('st_spectype', '?')
    bestfluxes.meta['logg'] = float(get_prop('st_logg', nan))
    bestfluxes.meta['mass'] = float(hostprops['st_mass'].to_value('Msun').filled(nan))

    #%% take a gander
    #
    # bestfluxes.sort('wave')
    # pprint(bestfluxes.meta)
    # bestfluxes.pprint(-1,-1)


    #%% save

    bestfluxes.write(data_folder / f'{target}.line-flux-table.ecsv', overwrite=True)


#%% cleanup and end iteration

    plt.close('all')

    utils.query_next_step(batch_mode, care_level, 1, msg='Continue to next target?')

  except StopIteration:
    break


#%% notes for possible future automation
"""
thoughts for possible automation in the future
  account for instrument lsf in bandpass
  account for stellar rotation in bandpass (can be 500 km s-1 for a 0.1 Rsun star rotating 1 d)
  account for spread of blended lines in bandpass
  sy radv in locating band -- can find rough offset based on spec too perhaps
  have some settings for excluding some lines due to airglow or other reasons
  generate plots to check by eye
"""