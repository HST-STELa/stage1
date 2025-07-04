import warnings

import numpy as np
from astropy import table
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import patches
from mpld3 import plugins
import mpld3

import database_utilities as dbutils
import paths
import utilities as utils


#%% general setup

linecat = table.Table.read('reference_files/fuv_line_list.ecsv')
linewaves = linecat['wave'].value

stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')


def interactive_click_loop(fig, plot_fn):
    def get_and_plot():
        xy = utils.click_coords(fig)
        if len(xy):
            x, y = zip(*xy)
            plotted_artists = plot_fn(x)
            plt.draw()
            return x, plotted_artists
        else:
            return [], []

    print('Collecting points. Click off the plot when done.')
    x, artists = get_and_plot()
    while True:
        print('Click off the plot if satisfied. Click new points if not.')
        xnew, newartists = get_and_plot()
        if xnew:
            # Remove plotted artists before repeating
            for artist in artists:
                artist.remove()
            x, artists = xnew, newartists
        else:
            break
    return x

#%% target specific

target = 'GJ 357'
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

    files = dbutils.find_data_files('coadd', instruments=config, directory=data_folder)
    if not files:
        files = dbutils.find_data_files('x1d', instruments=config, directory=data_folder)
        if not files:
            print(f'No {config} files found.')

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
        edges = interactive_click_loop(fig, edgeplot)
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
            band = interactive_click_loop(fig, bandplot)
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
            cbands = interactive_click_loop(fig, cplot)
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
            intvls = interactive_click_loop(fig, vplot)
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
    
#%% notes
"""
thoughts for possible automation in the future
  account for instrument lsf in bandpass
  account for stellar rotation in bandpass (can be 500 km s-1 for a 0.1 Rsun star rotating 1 d)
  account for spread of blended lines in bandpass
  sy radv in locating band -- can find rough offset based on spec too perhaps
  have some settings for excluding some lines due to airglow or other reasons
  generate plots to check by eye
"""