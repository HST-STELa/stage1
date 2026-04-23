from the_usuals import *

import paths
import database_utilities as dbutils

#%% inspect line flux tables

fd = paths.scratch / '2026-04-23 batch 3 suspect line and lya cases/rerun_b2'
targets = ['toi-4336a', 'hd21520', 'toi-6992', 'wasp-84']
for target in targets:
    f, = fd.rglob(f'{target}*table.ecsv')
    tbl = table.Table.read(f)
    print(target)
    tbl.pprint(-1,-1)
    print('\n\n')


#%% inspect lya fits

targets = ['toi-4336a', 'hd21520', 'toi-6992', 'wasp-84']
for target in targets:
    fd = paths.target_data(target)
    rf, = fd.rglob('*lya-recon.csv')
    lya = table.Table.read(rf)
    plt.figure()
    plt.step(lya['wave_lya'], lya['flux_lya'], where='mid', color='0.6')
    plt.plot(lya['wave_lya'], lya['lya_model_median'], 'C0')
    plt.plot(lya['wave_lya'], lya['lya_model_low_1sig'], 'C0:')
    plt.plot(lya['wave_lya'], lya['lya_model_high_1sig'], 'C0:')
    plt.title(target)
    plt.xlabel('wave')
    plt.ylabel('flux')