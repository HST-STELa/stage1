import warnings
import re

import matplotlib
from matplotlib import pyplot as plt
from astropy import table
from astropy import coordinates
from astropy import units as u
from astropy import time
from astropy.io import fits
import numpy as np

import paths
import database_utilities as dbutils
import ephemeris

from processing import target_lists
from processing import preloads
from processing.observation_table import ObsTable

matplotlib.use('agg')

#%% settings

# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

target = 'toi-776'


#%% catalog setup

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    hostcat = preloads.hosts.copy()
    planets = preloads.planets.copy()
hostcat.add_index('tic_id')
planets.add_index('tic_id')
planets['pl_id'] = dbutils.planet_suffixes(planets)
planets.add_index('pl_id')


#%% do it

# load observation table
data_dir = paths.target_hst_data(target)
obs_tbl = ObsTable.load_from_targname(target)
configs = np.unique(obs_tbl['science config'])


#%% add an exposure time column if one does not already exist

if 'exptime' not in obs_tbl.colnames:
    exptimes = []
    for fs in obs_tbl['key science files']:
        f = fs[0]
        fpath, = dbutils.find_stela_files_from_hst_filenames(f, data_dir)
        exptime = fits.getval(fpath, 'exptime', 1)
        exptimes.append(exptime)
    i_start = obs_tbl.colnames.index('start')
    obs_tbl.add_column(exptimes, name='exptime', index=i_start+1)
    obs_tbl['exptime'].format = '.1f'
    obs_tbl['exptime'].unit = 's'


#%% obs times

# get target coordinates
ids = preloads.stela_names.loc['hostname_file', target]
tic_id = ids['tic_id']
name = ids['hostname']
props = hostcat.loc[tic_id]
coords = coordinates.SkyCoord(props['ra'], props['dec'])

# adjust obs start times to barycentric frame
tmid = time.Time(obs_tbl['start']) + obs_tbl['exptime'].quantity
offset = ephemeris.bary_offset(tmid, coords)
tmid_bary = tmid + offset


#%% phases

# find all ephemeris tables
eph_dir = paths.target_epehemerides(target)
eph_paths = list(eph_dir.glob('*ephemerides.ecsv'))
assert len(eph_paths) > 0

# search string for extracting planet letter from pathname
pattern = f'{target}-([a-z]|\\d+)?\\.'

# get system planets
tic_id = preloads.stela_names.loc['hostname_file', target]['tic_id']
targplanets = planets.loc['tic_id', tic_id]
if isinstance(targplanets, table.Row):
    targplanets = planets[[targplanets.index]]

for eph_path in eph_paths:
    letter, = re.findall(pattern, eph_path.name)
    if letter in 'ef':
        continue

    planet = targplanets.loc['pl_id', letter]
    T = planet['pl_trandur']

    eph_tbl = table.Table.read(eph_path)
    eph_row, = eph_tbl[eph_tbl['picked']]
    ephem = ephemeris.Ephemeris.from_table_row(eph_row)

    phase, phase_err = ephem.transit_offset(tmid_bary)
    phasecolname = f'phase_{letter}'
    obs_tbl[phasecolname] = phase.to('h')
    obs_tbl[phasecolname].format = '.2f'
    obs_tbl[f'{phasecolname}_err'] = phase_err.to('min')
    obs_tbl[f'{phasecolname}_err'].format = '.1f'

    plt.figure()
    for i, config in enumerate(configs):
        slim = obs_tbl[obs_tbl['science config'] == config]
        phases = slim[phasecolname]
        starts = time.Time(slim['start'])
        dates = [start.isot[:10] for start in starts]
        plt.axhline(i+1, color='0.5', lw=0.5)
        plt.scatter(phases, [i+1]*len(slim), c=starts.jd)
        _, arg_unq_dates = np.unique(dates, return_index=True)
        for j in arg_unq_dates:
            plt.annotate('  ' + dates[j], xy=(phases[j],i+1), rotation='vertical',
                         ha='center', va='bottom', fontsize='x-small')
    labels = np.char.replace(configs.astype(str), 'hst-', '')
    plt.ylim(0, i+2)
    plt.yticks(ticks=np.arange(i+1)+1, labels=labels)
    plt.xlabel(f'time from mid-transit')

    xlim = plt.xlim()
    xlim_phase = xlim*u.h/ephem.P
    plt.twiny()
    plt.xlim(xlim_phase.to_value(''))
    plt.xlabel('orbital phase')

    plt.suptitle(f'{name} {letter}')
    plt.tight_layout()

    filename = f'{target}-{letter}.observation-timing.pdf'
    plt.savefig(data_dir / filename)

plt.close('all')

obs_tbl.write(ObsTable.get_path(target), overwrite=True)