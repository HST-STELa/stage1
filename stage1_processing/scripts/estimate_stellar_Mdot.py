import warnings
from math import pi

from astropy.table import Table, join
from astropy import units as u
from astropy import constants as const

import empirical
import paths
import utilities as utils
import catalog_utilities as catutils
from stage1_processing import target_lists
from stage1_processing import preloads

#%% host catalog

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    hosts = preloads.hosts.copy()
hosts.add_index('tic_id')

#%% target list

targets = target_lists.eval_no(2) + target_lists.eval_no(1)
tic_ids = preloads.stela_names.loc['hostname_file', targets]['tic_id']
proptbl = hosts.loc[tic_ids]


#%% get Mdots from X-ray fluxes

Eband = (0.1, 2.4) * u.keV
wband = const.h*const.c/Eband
wband = wband.to('AA')[::-1]

Fxs, Mdots = [], []
for target, props  in zip(targets, proptbl):
    xfile, = list(paths.data_targets.rglob(f'{target}*xray*.fits'))
    xray = Table.read(xfile)
    w = xray['wavelength'].quantity
    dw = xray['bin_width'].quantity
    if w[-1] < wband[1]:
        print(f'{target} spectrum only extends to {w[-1]} vs the Fx band of {wband}.')
        _wband = (wband[0], w[-1] + dw[-1]/2)
    else:
        _wband = wband
    f = xray['flux'].quantity
    Fx = utils.flux_integral(w, f, range=_wband, bin_widths=dw)

    getprop = lambda key: catutils.get_quantity_flexible(key, props, proptbl)
    R = getprop('st_rad')
    d = getprop('sy_dist')

    Fx_sfc = Fx * d**2/R**2
    Fx_sfc = Fx_sfc.to('erg s-1 cm-2')

    if (Fx_sfc < 1e3 * u.Unit('erg s-1 cm-2')) or (Fx_sfc > 2e8 * u.Unit('erg s-1 cm-2')):
        warnings.warn(f'{target} surface x-ray flux of {Fx_sfc:.1e} is beyond the range of Wood+ 21.')

    Mdot_sfc = empirical.stellar_Mdot_from_Xray_wood21(Fx_sfc)
    Mdot = Mdot_sfc * (4 * pi * R**2)

    Fxs.append(Fx)
    Mdots.append(Mdot)

Mdots = u.Quantity(Mdots).to('g s-1')
Fxs = u.Quantity(Fxs)


#%% put into a table and save

Mdot_tbl = Table((proptbl['hostname'], proptbl['tic_id'], targets, Fxs, Mdots),
                 names='hostname tic_id hostname_file Fx Mdot'.split())
Mdot_tbl['Fx'].description = 'X-ray flux at Earth in the 0.1-2.4 keV ROSAT band'

path = paths.catalogs / 'host_wind_estimates.ecsv'
if path.exists():
    past_Mdot_tbl = Table.read(path)
    merged_Mdot_tbl = catutils.merge_tables_with_update(past_Mdot_tbl, Mdot_tbl, 'tic_id')
    print('If this is the first use of this merging, compare the old table with the updated table and '
          'check that nothing crazy happend because this uses an AI-generated function.')
    # todo uncomment write code once you've seen a case that verifies this works
    # merged_Mdot_tbl.write(path, overwrite=True)
else:
    Mdot_tbl.write(path)