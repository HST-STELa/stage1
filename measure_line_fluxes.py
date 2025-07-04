import numpy as np
from astropy import table
from astropy.io import fits
from matplotlib import pyplot as plt

import database_utilities as dbutils
import paths
import utilities as utils

#%% general setup

linecat = table.Table.read('reference_files/fuv_line_list.ecsv')
linewaves = linecat['wave'].value

configs = (
    'stis_g140m',
    'stis-g140l',
    'stis-e140m',
    'cos-g130m',
    'cos-g160m',
    'cos-g140l'
)
iterconfigs = iter(configs)

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
    return np.sort(x)

#%% target specific
target = 'GJ 486'
tic_id = stela_name_tbl.loc['hostname', target]['tic_id']
targname_file, = dbutils.target_names_stela2file([target])
targname_file = str(targname_file)
data_folder = paths.data / targname_file

#%% custom coadds?

pass

#%% next config

config = next(iterconfigs)


#%% find files

files = dbutils.find_data_files('coadd', instruments=config, directory=data_folder)
if not files:
    files = dbutils.find_data_files('x1d', instruments=config, directory=data_folder)
    if not files:
        print(f'No {config} files found.')

iterfiles = iter(files)


#%% next file

file = next(iterfiles)


#%% plot

spec = fits.getdata(file, 1)
spec.w = spec['wavelength'].T
spec.f = spec['flux'].T
spec.e= spec['error'].T

fig = plt.figure(figsize=(15,5))
plt.title(file.name)
plt.step(spec.w, spec.f, where='mid')
plt.step(spec.w, spec.f, where='mid', lw=0.5, color='C0', alpha=0.5)
plt.tight_layout()

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
        artist = plt.axvspan(a, b, **kws)
        artists.append(artist)
    return artists


#%% ranges of viable data

print('Intervals of viable data:')
edgeplot = lambda x: [plt.axvline(xx, color='0.5', lw=2) for xx in x]
edges = interactive_click_loop(fig, edgeplot)
edges = np.reshape(edges, (-1,2))


#%% plot the lines that are within

within = (linewaves[:,None] >= edges[:,0]) & (linewaves[:,None] <= edges[:,1])
within = within.any(axis=1)
obslines = linecat[within].copy()
for line in obslines:
    plt.axvline(line['wave'], color='0.5', lw=0.5)
    plt.annotate(line['name'], xy=(line['wave'], 0.99), xycoords=('data', 'axes fraction'),
                 rotation='vertical', ha='center', va='top')

#%% pick bands for the lines
print('Line bands:')
wbuf = 2 if config.endswith('m') else 15
bandplot = lambda x: plotspans(x, color='C2', alpha=0.3, lw=0)
obslines['wa'] = 0.
obslines['wa'].format = '.2f'
obslines['wb'] = 0.
obslines['wb'].format = '.2f'
for i, line in enumerate(obslines):
    print(f'{line['name']} {line['wave']}')
    xlim = line['wave'] - wbuf, line['wave'] + wbuf
    plt.xlim(xlim)
    inplt = (spec.w > xlim[0]) & (spec.w < xlim[1])
    ymx = spec.f[inplt].max()
    plt.ylim(-0.05*ymx, 1.5*ymx)
    plt.draw()
    band = interactive_click_loop(fig, bandplot)
    if np.diff(band) < np.diff(spec.w[0]):
        continue
    obslines['wa'][i] = band[0]
    obslines['wb'][i] = band[1]
gohome()

#%% pick continuum ranges
print('Continuum bands:')
cplot = lambda x: plotspans(x, color='C9', alpha=0.3, lw=0)
cbands = interactive_click_loop(fig, cplot)
cbands = np.reshape(cbands, (-1,2))
obslines.meta['continuum bands'] = cbands
gohome()

# pick fitting intervals
print('Fitting intervals:')
vplot = lambda x: plotspans(x, color='0.5', alpha=0.1, lw=1)
intvls = interactive_click_loop(fig, vplot)
intvls = np.reshape(intvls, (-1,2))
obslines.meta['continuum fitting intervals'] = intvls
gohome()


#%% continuum fits

ps, covs = [], []
for intvl in intvls:
    cmask = (cbands[:,0] > intvl[0]) & (cbands[:,1] < intvl[1])
    fitbands = cbands[cmask]

    if 1220 in intvl:
        order = 3
    elif intvl[1] - intvl[0] < 20:
        order = 0
    else:
        order = 1

    waves, fluxes, errors = [], [], []
    for band in fitbands:
        wave = np.sum(band)/2
        m = (spec.w > band[0]) & (spec.w < band[1])
        flux, err = utils.flux_integral(spec.w[m], spec.f[m], spec.e[m])
        waves.append(wave)
        fluxes.append(flux)
        errors.append(err)
    errors = np.asarray(errors)

    p, cov = np.polyfit(waves, fluxes, order, w=1/errors, cov='unscaled')
    ps.append(p)
    covs.append(cov)

    n = 2 if order <= 1 else 100
    wplt = np.linspace(fitbands.min(), fitbands.max(), num=n)
    fplt = np.polyval(p, wplt)
    plt.plot(wplt, fplt, '-', color='C1', alpha=0.5)
obslines.meta['polynomial fits'] = ps
obslines.meta['polynomial covariances'] = covs


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
    dw = line['wb'] - line['wa']
    wmid = np.array([(line['wa'] + line['wb']) / 2])

    # continuum flux and uncty
    (ifit,), = np.where((wmid > intvls[:,0]) & (wmid < intvls[:,1]))
    p = ps[ifit]
    cov = covs[ifit]
    deg = len(p) - 1
    cflux = np.polyval(p, wmid)
    J = np.vstack([wmid ** i for i in range(deg, -1, -1)]).T  # shape (N, deg+1)
    cerr = np.sqrt(np.sum(J @ cov * J, axis=1))  # shape (N,)
    cF = cflux * dw
    cE = cerr * dw
    obslines['contflux'][i] = cF
    obslines['conterror'][i] = cE

    # line + cont flux and uncty
    m = (spec.w > line['wa']) & (spec.w < line['wb'])
    flux, err = utils.flux_integral(spec.w[m], spec.f[m], spec.e[m])
    F = flux * dw
    E = err * dw

    # line only flux and uncty
    lF = F - cF
    lE = np.sqrt(E**2 + cE**2)
    obslines['flux'][i] = lF
    obslines['error'][i] = lE


#%% save the table and the plot

# as of 2025-07 using mpld3 didn't work well becuase it can't hand the axes transforms that go into the
# annotations and axvspans of vlines
pdffile = str(file).replace('.fits', 'line_bands.pdf')
fig.savefig(pdffile)
pngfile = str(file).replace('.fits', 'line_bands.png')
fig.savefig(pngfile, dpi=100)

# save table
linefile = str(file).replace('.fits', 'line_fluxes.ecsv')
obslines.sort('wave')
obslines.write(linefile, overwrite=True)


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