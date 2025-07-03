from pathlib import Path
import re

from astropy import table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from target_selection_tools import query

#%% convert .txt tables to .csv

"""You actually just used ChatGPT for this."""

#%% load in the tables

folder = Path('single_use_data')
linsky = table.Table.read(folder / 'linsky2020_table1.csv')
france3 = table.Table.read(folder / 'france2018_table3.csv')
france4 = table.Table.read(folder / 'france2018_table4.csv')
france = table.vstack((france3, france4))


#%% use simbad to get the tic ids of each star and then cross match the tables

france_simbad_map = {'HD 114729 A':'HD 114729',
                     'LHS-26':'LHS 26'}
france_names = [france_simbad_map.get(name, name) for name in france['name'].tolist()]
france_ids = query.query_simbad_for_tic_ids(france_names)
for i, id in enumerate(france_ids):
    if ' ' in id:
        france_ids[i] = id.split()[0]
france['tic_id'] = france_ids

linsky_names = linsky['alternate name'].tolist()
for i, name in enumerate(linsky_names):
    if not name or len(re.findall('GJ|HD|HIP', name)) == 0:
        linsky_names[i] = linsky['name'][i]
linsky_simbad_map = {'HD 79210J':'GJ 338A'}
linsky_names = [linsky_simbad_map.get(name, name) for name in linsky_names]
linsky_ids = query.query_simbad_for_tic_ids(linsky_names)
for i, id in enumerate(linsky_ids):
    if ' ' in id:
        linsky_ids[i] = id.split()[0]
linsky['tic_id'] = linsky_ids

cat = table.join(france, linsky, 'tic_id')


#%% load melbourne table

melbourne = table.Table.read(folder / 'melbourne2020_data.txt', format='ascii.mrt')

melbourne_names = melbourne['Name'].tolist()
mel_ids = query.query_simbad_for_tic_ids(melbourne_names)
for i, id in enumerate(mel_ids):
    if ' ' in id:
        mel_ids[i] = id.split()[0]
melbourne['tic_id'] = mel_ids

cat = table.join(cat, melbourne, 'tic_id', 'outer')

#%% convert all france fluxes to 1 au

backup = cat.copy()

line_names = 'C II,Si III,Si IV,N V'.split(',')
line_names = np.char.add(line_names, ' (1e-15)')
col_names = line_names.tolist() + 'Fbol (1e-7),Feuv (1e-14)'.split(',')
dfac = (cat['distance']*u.pc/u.au)**2
dfac = dfac.to_value('')
for name in col_names:
    fac, = re.findall(r'\de-\d+', name)
    fac = float(fac)
    cat[name] *= fac * dfac
    newname = re.sub(r' \(\de-\d+\)', '', name)
    cat.rename_column(name, newname)


#%% convert all melbourne luminosities to 1 au

dfac = 1/(4*np.pi*u.au**2)
dfac = dfac.to_value('cm-2')

line_names = 'CII,MgII,HeII,SiII,SiIII,CIV,NV,Lya'.split(',')
col_names = np.char.add('L-', line_names)

for name in col_names:
    fluxes = cat[name] * dfac
    newname = name.replace('L-','F-')
    cat[newname] = fluxes


#%% create correlations for the lines

line_names = 'C II,Si III,Si IV,N V'.split(',')
# some shenanigans to combine fluxes from different sources without duplicates
Flya = cat['Flya']
Flya[Flya.mask] = cat['F-Lya'][Flya.mask]

for line in line_names:
    # some shenanigans to combine fluxes from different sources without duplicates
    fcol = line
    mcol = 'F-' + line.replace(' ', '')
    Fline = cat[fcol]
    if mcol in cat.colnames:
        Fline[Fline.mask] = cat[mcol][Fline.mask]

    plt.figure()
    plt.ylabel(line)
    plt.xlabel('Lya')
    for SpT in 'FGKM':
        mask = ((np.char.count(cat['SpT'].filled(''), SpT) > 0)
                | (np.char.count(cat['SpType'].filled(''), SpT) > 0))
        plt.loglog(Flya[mask], Fline[mask], 'o', label=SpT)

    plt.title(line)

    usable = ~Fline.mask & ~Flya.mask
    x = np.log10(Flya[usable])
    y = np.log10(Fline[usable])
    p = np.polyfit(x, y, 1)
    y_predicted = np.polyval(p, x)
    scatter = np.std(y - y_predicted)
    plt.plot(10**x, 10**y_predicted, 'k-')
    eqn = f'log10(F-{line}) = {p[0]:.3f}*log10(F-Lya) + {p[1]:.3f} [{scatter:.2f} dex scatter]'
    print(eqn)

    plt.annotate(eqn, xy=(0.98,0.02), xycoords='axes fraction', ha='right')
    plt.legend()
    plt.savefig(f'../../scratch/lya_fuv-line_correlations_2025-07/{line}.pdf')

