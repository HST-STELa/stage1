import warnings
from pprint import pprint

import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import patches
from mpld3 import plugins
import mpld3

import database_utilities as dbutils
import paths
import utilities as utils
import catalog_utilities as catutils


#%% general setup

linecat = table.Table.read('reference_files/fuv_line_list.ecsv')
linewaves = linecat['wave'].value

stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')

target_table = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
target_table = catutils.planets2hosts(target_table)
target_table.add_index('tic_id')



#%% target specific

# target = 'TOI-1759'
target = 'HD 149026'
shift_errors = True

tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
targname_file = str(targname_file)
data_folder = paths.data / targname_file

configs = (
    'stis-g140m',
    'stis-g140l',
    'stis-e140m',
    'cos-g130m',
    'cos-g160m',
    'cos-g140l'
)

plt.close('all')

#%% custom coadds?

pass

#%% loop

for config in configs:
    def prompt_nextfile():
        return input('Continue to next file? Enter for yes. Anything else for no.')

#%% find files

    files = dbutils.find_coadd_or_x1ds(targname_file, instruments=config, directory=data_folder)
    iterfiles = iter(files)


#%% next file

    for file in iterfiles:
        def prompt_nextstep():
            return input('Continue to next step? Enter for yes. Anything else for no.')

#%% load data

        spec = fits.getdata(file, 1)
        spec.w = spec['wavelength'].T
        spec.f = spec['flux'].T
        spec.e = spec['error'].T


#%% shift errors -- may not be necessary for a really bright spectrum
        """Also hopefully in the future I will fix coadd script so that it uses poisson errors and avoids the annoying error
        inflation issue."""

        if shift_errors:
            spec.emod = utils.shift_floor_to_zero(spec.e, window_size=50)
        else:
            spec.emod = spec.e


#%% plot data
        fig = plt.figure(figsize=(14,5))
        plt.title(file.name)
        plt.step(spec.w, spec.f, where='mid')
        plt.step(spec.w, spec.emod, where='mid', lw=0.5, color='C0', alpha=0.5)
        plt.tight_layout()
        ax, = fig.get_axes()

        print('Zoom to a good range, then click off the plot.')
        _ = utils.click_coords(fig)
        xhome = plt.xlim()
        yhome = plt.ylim()
        def gohome():
            plt.xlim(xhome)
            plt.ylim(yhome)
            plt.draw()

        def plotspans(x, **kws):
            if len(x) % 2 == 1:
                print('Odd number of clicks. Even number required. Try again.')
                return []
            bands = np.reshape(x, (-1,2))
            artists = []
            for a, b in bands:
                artist = patches.Rectangle((a, 0), (b-a), yhome[1], **kws)
                ax.add_patch(artist)
                artists.append(artist)
            return artists


#%% ranges of viable data

        print('Intervals of viable data:')
        edgeplot = lambda x: [plt.plot([xx]*2, yhome, color='0.5', lw=2)[0] for xx in x]
        edges, _ = utils.interactive_click_loop(fig, edgeplot)
        edges = np.reshape(edges, (-1,2))

        ans = prompt_nextstep()
        if ans != '':
            break


#%% plot the lines that are within

        within = (linewaves[:,None] >= edges[:,0]) & (linewaves[:,None] <= edges[:,1])
        within = within.any(axis=1)
        obslines = linecat[within].copy()
        for line in obslines:
            plt.plot([line['wave']]*2, yhome, color='0.5', lw=0.5)
            plt.annotate(line['name'] + ' ', xy=(line['wave'], 0),
                         rotation='vertical', ha='center', va='top')


#%% pick bands for the lines
        print('Line bands:')
        wbuf = 2 if config.endswith('m') else 15
        bandplot = lambda x: plotspans(x, color='C2', alpha=0.3, lw=0)
        MCfloat = lambda: table.MaskedColumn(length=len(obslines), dtype=float, mask=True, format='.2f')
        obslines['wa'] = MCfloat()
        obslines['wb'] = MCfloat()
        for i, line in enumerate(obslines):
            print(f'{line['name']} {line['wave']}')
            xlim = line['wave'] - wbuf, line['wave'] + wbuf
            plt.xlim(xlim)
            inplt = (spec.w > xlim[0]) & (spec.w < xlim[1])
            ymx = spec.f[inplt].max()
            plt.ylim(-0.05*ymx, 1.5*ymx)
            plt.draw()
            band, _ = utils.interactive_click_loop(fig, bandplot)
            if np.diff(band) < np.diff(spec.w, axis=0)[0]:
                continue
            obslines['wa'][i] = band[0]
            obslines['wb'][i] = band[1]
        gohome()

        ans = prompt_nextstep()
        if ans != '':
            break


#%% pick continuum ranges
        print('Continuum bands:')
        chunksize = 50
        overlap = 10
        cplot = lambda x: plotspans(x, color='C9', alpha=0.3, lw=0)
        cband_list = []
        wa = np.min(edges)
        while wa < np.max(edges):
            wb = wa + chunksize
            plt.xlim(wa, wb)
            plt.draw()
            cbands, _ = utils.interactive_click_loop(fig, cplot)
            cbands = np.reshape(cbands, (-1,2))
            cband_list.append(cbands)
            wa += chunksize - overlap
        cbands = np.vstack(cband_list)

        obslines.meta['continuum bands'] = cbands
        gohome()

        ans = prompt_nextstep()
        if ans != '':
            break


#%% pick fitting intervals
        print('Fitting intervals:')
        chunksize = 200 if 'g140l' in config else 50
        overlap = chunksize / 5
        vplot = lambda x: plotspans(x, color='0.5', alpha=0.1, lw=1)
        intvls_list = []
        wa = np.min(edges)
        while wa < np.max(edges):
            wb = wa + chunksize
            plt.xlim(wa, wb)
            plt.draw()
            intvls, _ = utils.interactive_click_loop(fig, vplot)
            intvls = np.reshape(intvls, (-1,2))
            intvls_list.append(intvls)
            wa += chunksize - overlap
        intvls = np.vstack(intvls_list)

        obslines.meta['continuum fitting intervals'] = intvls
        gohome()

        ans = prompt_nextstep()
        if ans != '':
            break


#%% continuum estimates

        cfluxes, cerrors = [], []
        clns = []
        for intvl in intvls:
            cmask = (cbands[:,0] > intvl[0]) & (cbands[:,0] < intvl[1])
            fitbands = cbands[cmask]

            Fs, Es, dws = [], [], []
            for band in fitbands:
                wave = np.sum(band)/2
                dw = np.diff(band)[0]
                m = (spec.w > band[0]) & (spec.w < band[1])
                n = np.sum(m)
                if n == 0:
                    continue
                elif n == 1:
                    F, E = spec.f[m][0], spec.emod[m][0]
                else:
                    F, E = utils.flux_integral(spec.w[m], spec.f[m], spec.emod[m])
                Fs.append(F)
                Es.append(E)
                dws.append(dw)
            Es = np.array(Es)
            dw_sum = np.sum(dws)
            cflux = np.sum(Fs)/dw_sum
            cerr = np.sqrt(np.sum(Es**2))/dw_sum
            cfluxes.append(cflux)
            cerrors.append(cerr)

            wplt = np.array((fitbands.min(), fitbands.max()))
            cln, = plt.plot(wplt, [cflux]*2, '-', color='C1', alpha=0.5)
            clns.append(cln)
        obslines.meta['continuum fluxes'] = cfluxes
        obslines.meta['continuum flux errors'] = cerrors


#%% compute line fluxes

        MCfloat = lambda: table.MaskedColumn(length=len(obslines), dtype=float, mask=True, format='.2e')
        obslines['flux'] = MCfloat()
        obslines['error'] = MCfloat()
        obslines['contflux'] = MCfloat()
        obslines['conterror'] = MCfloat()
        for i, line in enumerate(obslines):
            if np.ma.is_masked(line['wa']):
                obslines['contflux'][i] = 0
                obslines['conterror'][i] = 0
                obslines['flux'][i] = 0
                obslines['error'][i] = 0
                continue
            dw = line['wb'] - line['wa']
            wmid = np.array([(line['wa'] + line['wb']) / 2])

            # continuum flux and uncty
            (icont,), = np.where((wmid > intvls[:,0]) & (wmid < intvls[:,1]))
            cflux = cfluxes[icont]
            cerr = cerrors[icont]
            cF = cflux * dw
            cE = cerr * dw
            obslines['contflux'][i] = cF
            obslines['conterror'][i] = cE

            # line + cont flux and uncty
            m = (spec.w > line['wa']) & (spec.w < line['wb'])
            flux, err = utils.flux_integral(spec.w[m], spec.f[m], spec.emod[m])
            F = flux * dw
            E = err * dw

            # line only flux and uncty
            lF = F - cF
            lE = np.sqrt(E**2 + cE**2)
            obslines['flux'][i] = lF
            obslines['error'][i] = lE

        with np.errstate(divide='ignore'):
            obslines['snr'] = obslines['flux']/obslines['error']
            obslines['snr'][obslines['error'] == 0] = 0
            obslines['snr'].format = '.1f'


#%% take a gander

        obslines.pprint(-1,-1)
        ans = prompt_nextstep()
        if ans != '':
            break


#%% save the table and plot

        pass
        # save plot with mpl3
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The converter')
            warnings.filterwarnings('ignore', 'Blended')
            dpi = fig.get_dpi()
            fig.set_dpi(150)
            plugins.connect(fig, plugins.MousePosition(fontsize=14))
            htmlfile = str(file).replace('.fits', '.line_band_plot.html')
            mpld3.save_html(fig, htmlfile)
            fig.set_dpi(dpi)

        # save table
        linefile = str(file).replace('.fits', '.line_fluxes.ecsv')
        obslines.sort('wave')
        obslines.write(linefile, overwrite=True)

    ans = prompt_nextfile()
    if ans != '':
        break


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
scores = []
for fluxtbl in fluxtbls:
    mask = fluxtbl['snr'] > snr_threshold
    unq_temps = np.unique(fluxtbl['Tform'][mask])
    score = len(unq_temps)
    scores.append(score)
scores = np.asarray(scores)

if (sum(scores == np.max(scores)) > 1) and (np.max(scores) > 0):
    raise NotImplementedError # should write some code in this case to pick the one with higher snrs
ibest = np.argmax(scores)

bestfluxes = catutils.add_masks(fluxtbls[ibest], inplace=False)

bestfluxes.add_index('name')
bestfluxes['source'] = table.MaskedColumn(dtype=object, length=len(bestfluxes))
bestfluxes['source'] = bestfluxes.meta['config']
del bestfluxes.meta['config']


#%% add in Lya

lyarecon_file, = data_folder.rglob('*lya_recon.csv')
lyarecon = table.Table.read(lyarecon_file)
Fs = []
for suffix in ['low_1sig', 'median', 'high_1sig']:
    F = np.trapz(lyarecon[f'lya_intrinsic unconvolved_{suffix}'], lyarecon['wave_lya'])
    Fs.append(F)
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
bestfluxes.meta['Lya source file'] = lyarecon_file.name


#%% augment with line-line correlations as needed

basecat = table.Table.read('reference_files/fuv_line_list.ecsv')
basecat.add_index('name')

dist = target_table.loc[tic_id]['sy_dist'] * u.pc

min_snr = 3

corrtbl = table.Table.read('reference_files/line-line_correlations.ecsv')
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
        if F/E > snr_threshold:
            result = True
    usable.append(result)
usable = np.asarray(usable)

# fill lines that don't
usable_lines = corrlines[usable]
usable_temps = corrtemps[usable]
for line in corrlines[~usable]:
    # add rows if not present
    if line not in bestfluxes['name']:
        missing_rows = basecat.loc[line].copy()
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
bestfluxes.meta['dist'] = dist.to_value('pc')
bestfluxes.meta['radius'] = get_prop('st_rad', nan)
bestfluxes.meta['Teff'] = get_prop('st_teff', nan)
bestfluxes.meta['SpT'] = get_prop('st_spectype', '?')
bestfluxes.meta['logg'] = get_prop('st_logg', nan)
bestfluxes.meta['mass'] = get_prop('st_mass', nan)


#%% take a gander

bestfluxes.sort('wave')
pprint(bestfluxes.meta)
bestfluxes.pprint(-1,-1)


#%% save

bestfluxes.write(data_folder / f'{targname_file}.line-flux-table.ecsv', overwrite=True)


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