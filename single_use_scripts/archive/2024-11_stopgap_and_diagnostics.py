#%% imports and constants
from math import nan

from astropy import table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import paths
import catalog_utilities as catutils
from lya_prediction_tools import heritage, transit

#%% load second cut
cut2 = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'cut2_planet_and_host_type.ecsv')

#%% load SNRs and parameters from first phase II submission

old_snrs = table.Table.read(paths.other / 'initial_phase2_values.ecsv')
old_snrs['id'] = old_snrs['id'].astype('object')
old_snrs = old_snrs[~old_snrs['id'].mask]

#%% histogram snr differences
"""
The differences between the two are the use of a flexible integration range for comparing in vs out of transit spectra 
rather than [-150, 100] km s-1 and a fix of a bug in the error estimate that caused some values to be zeroed out. 
"""

matching_ids = table.join(old_snrs[['id']], cut2[['id']])
catutils.add_index(cut2, 'id')
i = cut2.loc_indices[matching_ids['id']]
new_snrs = cut2[i]

Flya_at_earth = new_snrs['Flya_1AU_adopted'].quantity * (u.AU/new_snrs['sy_dist'].quantity)**2
new_snrs['Flya_at_earth'] = Flya_at_earth.to('erg s-1 cm-2')
def get_heritage_snrs(cat):
    snrs = []
    for row in tqdm(cat):
        snr = heritage.transit_snr(row, expt_out=3500, expt_in=6000, optimistic=True)
        snrs.append(snr)
    return u.Quantity(snrs)
new_snrs['snr_new_cat'] = get_heritage_snrs(new_snrs)
old_snrs['snr_rerun'] = get_heritage_snrs(old_snrs)

params = dict(expt_out=3500, expt_in=6000,
              default_rv=0, tail_shift='blue wing max',
              n_H_percentile=16, lya_percentile=84)
snr_bugfix = transit.opaque_tail_transit_SNR(new_snrs, integrate=(-150, 100), show_progress=True, **params)
new_snrs['snr_new_cat+fun'] = snr_bugfix
snr_optimal_integration = transit.opaque_tail_transit_SNR(new_snrs, integrate='best', show_progress=True, **params)
new_snrs['snr_new_cat+fun+range'] = snr_optimal_integration

params['n_H_percentile'] = 84
params['lya_percentile'] = 16
new_snrs['snr_pessimistic'] = transit.opaque_tail_transit_SNR(new_snrs, integrate='best', show_progress=True, **params)

comparison = table.join(old_snrs, new_snrs, 'id')

old_snr_values = comparison['lya_transit_snr_crude_optimistic']

# make sure the heritage function is succesfully reproducing heritage values when given previous parameters
assert np.all(np.isclose(comparison['snr_rerun'], old_snr_values, equal_nan=True))

d_newcat = (comparison['snr_new_cat'] - old_snr_values)/old_snr_values
d_bugfix = (comparison['snr_new_cat+fun'] - old_snr_values)/old_snr_values
d_newint = (comparison['snr_new_cat+fun+range'] - old_snr_values)/old_snr_values

plt.figure()
hist_kws = dict(bins=50, alpha=0.5)
_ = plt.hist(d_newcat, label='+ updated parameters', **hist_kws)
_ = plt.hist(d_bugfix, label='++ bug fix in error calc', **hist_kws)
_ = plt.hist(d_newint, label='+++ optimal integration', **hist_kws)
plt.xlabel('(new SNR - old SNR) / old SNR')
plt.legend()
plt.tight_layout()


#%% plot some differences in parameters

def diff_new_old(col):
    diff = comparison[f'{col}_2'] - comparison[f'{col}_1']
    return diff.filled(nan)
plt_kws = dict(ls='', marker='.', alpha=0.5)
hist_kws = dict(bins=50, alpha=0.5)

old_lya = comparison['Flya_at_1AU']
d_lya = (comparison['Flya_1AU_adopted'] - old_lya)/old_lya
d_lya.mask = ~np.isfinite(d_lya.filled(nan))
d_lya = d_lya.filled(nan)
d_lya_teff = (comparison['Flya_1AU_Teff_schneider'] - old_lya)/old_lya
d_lya_teff.mask = ~np.isfinite(d_lya_teff.filled(nan))
d_lya_teff = d_lya_teff.filled(nan)
plt.figure()
_ = plt.hist(d_lya, label='log(new/old)', **hist_kws)
_ = plt.hist(d_lya, label='log(new/old)', bins=np.linspace(-1, 1, 300))
_ = plt.hist(d_lya_teff, label='log(new_Teff/old)', **hist_kws)

plt.figure()
plt.plot(d_lya, d_newcat, **plt_kws)
plt.xlabel('∆Flya')
plt.ylabel('∆SNR')

plt.figure()
plt.plot(diff_new_old('st_teff'), d_lya, **plt_kws)
plt.xlabel('∆Teff')
plt.ylabel('∆Flya')

plt.figure()
plt.plot(diff_new_old('sy_fuvmag'), d_lya, **plt_kws)
plt.xlabel('∆FUV')
plt.ylabel('∆Flya')

plt.figure()
plt.plot(diff_new_old('sy_nuvmag'), d_lya, **plt_kws)
plt.xlabel('∆NUV')
plt.ylabel('∆Flya')


#%% fancy table to track changes

# note that the "1" values are from the old table, "2" from the new table
comparison['snr_old'] = comparison['lya_transit_snr_crude_optimistic']
col_fmt_pairs = (
    ('id', 's'),
    ('snr_old', '.2g'),
    ('snr_rerun', '.2g'),
    ('snr_new_cat', '.2g'),
    ('snr_new_cat+fun', '.2g'),
    ('snr_new_cat+fun+range', '.2g'),
    ('Flya_at_1AU', '.1e'),
    ('Flya_1AU_adopted', '.1e'),
    ('Flya_1AU_Teff_linsky', '.1e'),
    ('Flya_1AU_Teff_schneider', '.1e'),
    ('Flya_1AU_nuv', '.1e'),
    ('Flya_1AU_fuv', '.1e'),
    ('sy_fuvmag_1', '.1f'),
    ('sy_fuvmaglim_1', '.1f'),
    ('sy_fuvmag_2', '.1f'),
    ('sy_fuvmaglim_2', '.1f'),
    ('sy_nuvmag_1', '.1f'),
    ('sy_nuvmaglim_1', '.1f'),
    ('sy_nuvmag_2', '.1f'),
    ('sy_nuvmaglim_2', '.1f'),
    ('st_teff_1', '.0f'),
    ('st_teff_2', '.0f'),
    ('st_mass_1', '.1f'),
    ('st_mass_2', '.1f'),
    ('st_rad_1', '.2f'),
    ('st_rad_2', '.2f'),
    ('pl_bmasse_1', '.1f'),
    ('pl_bmasse_2', '.1f'),
    ('pl_orbsmax_1', '.2f'),
    ('pl_orbsmax_2', '.2f'),
    ('st_radv_1', '.1f'),
    ('st_radv_2', '.1f'),
    ('sy_dist_1', '.1f'),
    ('sy_dist_2', '.1f'))

cols, _ = zip(*col_fmt_pairs)
pretty_table = comparison[cols]
for col, fmt in col_fmt_pairs:
    pretty_table[col].format = fmt
pretty_table.write(paths.selection_outputs / 'diff_tbl_submissions_1_2.csv', overwrite=True)

# now this can be opened in spreadsheet software for easier viewing



#%% top targets based on having good SNRs in both iterations
"""
Pick top 20 targets as stopgap to help with HST long range planning.
"""
good_before = comparison['lya_transit_snr_crude_optimistic'] > 10
still_good = comparison['snr_new_cat+fun+range'] > 10
consider = comparison[good_before & still_good]
consider.sort('snr_new_cat+fun+range', reverse=True)

