import warnings
import re

import matplotlib
from matplotlib import pyplot as plt
from astropy import table
from astropy import coordinates
from astropy import units as u
from astropy import time
import numpy as np

import paths
import database_utilities as dbutils
import ephemeris

from stage1_processing import target_lists
from stage1_processing import preloads

#%% choose whether to show plots

matplotlib.use('Qt5Agg')
plt.ioff()


#%% catalog setup

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    hostcat = preloads.hosts.copy()
    planets = preloads.planets.copy()
hostcat.add_index('tic_id')
planets.add_index('tic_id')
planets['pl_id'] = dbutils.planet_suffixes(planets)
planets.add_index('pl_id')


#%% targets

# targets = target_lists.observed_since('2025-07-14')
targets = target_lists.eval_no(2)
# targets = target_lists.everything_in_progress_table()


#%% do it

for target in targets:

    # load observation table
    data_dir = paths.target_hst_data(target)
    obs_tbl_paths = list(data_dir.glob('*observation-table.ecsv'))
    if len(obs_tbl_paths) == 0:
        print(f'No observation table found for {target}. Skipping.')
        continue
    obs_tbl_path, = obs_tbl_paths
    obs_tbl = table.Table.read(obs_tbl_path)
    configs = np.unique(obs_tbl['science config'])

    # get target coordinates
    ids = preloads.stela_names.loc['hostname_file', target]
    tic_id = ids['tic_id']
    name = ids['hostname']
    props = hostcat.loc[tic_id]
    coords = coordinates.SkyCoord(props['ra'], props['dec'])

    # adjust obs start times to barycentric frame
    start = time.Time(obs_tbl['start'])
    offset = ephemeris.bary_offset(start, coords)
    start_bary = start + offset

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

        planet = targplanets.loc['pl_id', letter]
        T = planet['pl_trandur']

        eph_tbl = table.Table.read(eph_path)
        eph_row, = eph_tbl[eph_tbl['picked']]
        ephem = ephemeris.Ephemeris.from_table_row(eph_row)

        phase, phase_err = ephem.transit_offset(start_bary)
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

    obs_tbl.write(obs_tbl_path, overwrite=True)