import sys;

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/parke/Google Drive/Research/STELa/public github repo/stage1'])
from importlib import reload
import string
import sys
import re
import warnings
from math import pi, nan
from datetime import datetime
from astropy import table
from astropy import units as u
from astropy import constants as const
from astropy import coordinates as coord
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import catalog_utilities
import paths
import empirical
import catalog_utilities as catutils
from lya_prediction_tools import transit, lya
from target_selection_tools import galex_query
from target_selection_tools import duplication_checking as dc
from target_selection_tools import reference_tables as ref
from target_selection_tools import query, columns, apt

import database_utilities as dbutils
erg_s_cm2 = u.Unit('erg s-1 cm-2')
allocated_orbits = 204
# if true, pulls the latest versions of these files from the progress review folder:
# - visit list from status xml
# - manually udpated observation status excel sheet
# then makes sure all planned visits are kept and lemons are dropped
# whether an fuv visit is kept or dropped depends on the "pass to stage 1b?" column in the excel sheet
toggle_mid_cycle_update = True
# use thes to specify visits we are going to add back in or remove for whatever reason
# for example, in the May update I wanted to remove the K2-72 fuv visit Z2 because of an error in the clearance sheet
# and add in NB for TOI-2015 due to revised pass through cut
hand_remove_visits = 'J7'.split()
hand_add_visits = 'W1 W6 OO OP'.split()
# adds this many orbits-worth of backup targets if desired so you can get ahead on vetting them
backup_orbits = 5
# backup_orbits = int(round(0.2 * 204))
toggle_plots = True
toggle_save_outputs = True
toggle_save_galex = True
toggle_save_difftbl = True
toggle_save_visit_labels = True
toggle_redo_all_galex = False # only needed if galex search methodology modified
toggle_remake_filtered_hst_archive = True  # only needed if archive file redownloaded
diff_label = 'target-backfill-2025-06'
toggle_checkpoint_saves = True
toggle_target_removal_test = False # removes targets to see if sort order changes as a test for bugs
assumed_transit_range = [-100, 50] # based on typical ranges from actual transit observations
default_sys_rv = "ism"
if toggle_checkpoint_saves:
    cat = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt4__fill-basic_properties.ecsv')
#%% load and match existing galex mags
"""
These are helpful for improved Lya flux estimates and estimating buffer times for G140L spectra. 
This is slowww because handling proper motions requires a target-by-target query. Expect 1 h per 1000 targets.
Results saved will be reloaded and matched in in order to save time. 
"""
# try loading and cross-matching galex first
filepath = paths.selection_intermediates / 'galex_magntiudes.ecsv'
if filepath.exists():
    galex_tbl = catutils.load_and_mask_ecsv(filepath)
    cat = table.join(cat, galex_tbl, 'id', join_type='left')
#%% new/missing galex
pass
# create empty columns for galex mags if needed
for band in ['fuv', 'nuv']:
    basename = f'sy_{band}mag'
    if basename not in cat.colnames:
        catutils.add_masked_columns(cat, basename, 'float', suffixes=('', 'err1', 'err2', 'lim'))
    obsname = basename + 'obs'
    if obsname not in cat.colnames:
        cat[obsname] = table.MaskedColumn(length=len(cat), mask=True, dtype=bool)
def add_mag(cps, cpserr, band, i):
    if cps == -999: # either lower limit or no data
        if cpserr == -999: # no data
            cat[f'sy_{band}magobs'][i] = False
        else: # lower limit, limit cps given in cpserr
            result = galex_query.galex_cps2mag(cpserr, band)
            assert np.isfinite(result)
            cat[f'sy_{band}mag'][i] = result
            cat[f'sy_{band}maglim'][i] = -1
            cat[f'sy_{band}magobs'][i] = True
    elif cps > 0:
        result = galex_query.galex_cps2mag(cps, band)
        assert np.isfinite(result)
        cat[f'sy_{band}mag'][i] = result
        magerr = galex_query.cps2magerr(cps, cpserr)
        cat[f'sy_{band}magerr1'][i] = magerr
        cat[f'sy_{band}magerr2'][i] = magerr
        cat[f'sy_{band}maglim'][i] = 0
        cat[f'sy_{band}magobs'][i] = True
    elif np.isnan(cps):
        # no data, this is returned when one bad was observed and the other was not
        cat[f'sy_{band}magobs'][i] = False
    else:
        raise ValueError
pos = catutils.J2000_positions(cat)
missing_values = cat['sy_nuvmagobs'].mask | cat['sy_fuvmagobs'].mask
while any(missing_values):
    i_failed, = np.nonzero(missing_values)
    for i in tqdm(i_failed):
        try:
            assert np.isfinite(pos.ra[i])
            posi = pos[i]
            ra= posi.ra.to_value('deg')
            dec = posi.dec.to_value('deg')
            pm_ra = posi.pm_ra_cosdec.to_value('mas yr-1')
            pm_dec = posi.pm_dec.to_value('mas yr-1')
            result = galex_query.extract_and_coadd(ra, dec, pm_ra, pm_dec, match_radius=16. / 3600, query_timeout=30.)
            (nuv, nerr), (fuv, ferr) = result
            add_mag(nuv, nerr, 'nuv', i)
            add_mag(fuv, ferr, 'fuv', i)
            missing_values[i] = False
        except ValueError as e:
            msg = str(e)
            if 'Request not filled due to connection errors' in msg:
                continue
            if 'Response ended prematurely' in msg:
                continue
assert np.all(np.isfinite(cat['sy_fuvmag']))
toggle_save_galex
if toggle_save_galex:
    assert not np.any(cat['id'].mask)
    galex_cols = ['id']
    for band in ['nuv', 'fuv']:
        for suffix in ['', 'err1', 'err2', 'lim', 'obs']:
            name = f'sy_{band}mag{suffix}'
            galex_cols.append(name)
    galex_tbl = cat[galex_cols]
    galex_tbl.write(paths.selection_intermediates / 'galex_magntiudes.ecsv', overwrite=True)
#%% estimate Lya based on galex
for band in ['nuv', 'fuv']:
    # values must not be masked or an upper or lower limit
    mag = cat[f'sy_{band}mag'].filled(nan)
    lim_flag = cat[f'sy_{band}maglim'].filled(nan) != 0
    MK = cat['st_teff'].filled(1e4) < 4500. # because schneider GALEX-Lya relationship is for late Ks and Ms only
    d = cat['sy_dist'].filled(nan).quantity
    Flya_1AU = empirical.Lya_from_galex_schneider19(mag, d.to_value('pc'), band=band)
    Flya_1AU = Flya_1AU * erg_s_cm2
    bad_galex_lya = ~np.isfinite(Flya_1AU) | lim_flag | ~MK
    cat[f'Flya_1AU_{band}'] = table.MaskedColumn(Flya_1AU, mask=bad_galex_lya)
#%% estimate Lya based on Teff, distance
keepers = cat['stage1'].filled(True)
Teff = cat['st_teff'].filled(nan)
# assume stars without measured period or known youth fall on the "normal" track from Linsky+ 2013
Prot_adopt = cat['st_rotp'].filled(nan)
noP = ~(Prot_adopt > 0)
young = cat['flag_young']
Prot_adopt[young & noP] = 1
Prot_adopt[~young & noP] = 20
Flya_linsky = empirical.Lya_from_Teff_linsky13(Teff, Prot_adopt) * erg_s_cm2
valid_lya = Flya_linsky > 0
assert np.all(valid_lya[keepers]) # sanity check
cat['Flya_1AU_Teff_linsky'] = table.MaskedColumn(Flya_linsky, mask=~valid_lya,
                                                 description="Lya flux estimated based on Teff and Prot using Lisnky+ 2013.")
Flya_schneider = empirical.Lya_from_Teff_schneider19(Teff) * erg_s_cm2
valid_lya = Flya_schneider > 0
assert np.all(valid_lya[keepers])
cat['Flya_1AU_Teff_schneider'] = table.MaskedColumn(Flya_schneider, mask=~valid_lya,
                                                    description="Lya flux estimated based on Teff using Schneider+ 2019.")
#%% pick best Lya estimate
catutils.add_masked_columns(cat, 'Flya_1AU_adopted', 'float', suffixes=('',), unit='erg s-1 cm-2')
F = cat['Flya_1AU_adopted'] # changes to F will appear in the cat table
fuv = cat['sy_fuvmag'].filled(nan)
fuvlim = cat['sy_fuvmaglim'].filled(nan)
nuv = cat['sy_nuvmag'].filled(nan)
nuvlim = cat['sy_nuvmaglim'].filled(nan)
Teff = cat['st_teff'].filled(nan)
Prot = cat['st_rotp'].filled(nan)
# based on Pineda 2021 for M star saturated activity level
# expected to apply to all given the similarity in saturation levels across
# spectral types for other activity generated flux
max_plausible_Llya_Lbol = 10**-3.0
max_plausible_Llya = max_plausible_Llya_Lbol * 10**cat['st_lum'] * u.Lsun
max_plausible_Flya_1AU = max_plausible_Llya / (4 * np.pi * (1*u.AU)**2)
max_plausible_Flya_1AU = max_plausible_Flya_1AU.to('erg s-1 cm-2')
use_fuv = (np.isfinite(fuv) & (fuvlim == 0) & (cat['Flya_1AU_fuv'].filled(nan) > 0)
           & (cat['Flya_1AU_fuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using FUV-estimated Lya (Schneider+ 19) for {np.sum(use_fuv)} targets.")
F[use_fuv] = cat['Flya_1AU_fuv'][use_fuv]
use_nuv = (F.mask & np.isfinite(nuv) & (nuvlim == 0) & (cat['Flya_1AU_nuv'].filled(nan) > 0)
           & (cat['Flya_1AU_nuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using NUV-estimated Lya (Schneider+ 19) for {np.sum(use_nuv)} targets.")
F[use_nuv] = cat['Flya_1AU_nuv'][use_nuv]
use_Teff1 = F.mask & (Teff > 0) & (Prot > 0) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff+Prot-estimated Lya (Linsky+ 13) for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]
use_Teff1 = F.mask & (Teff > 0) & ~(Prot > 0) & (cat['flag_young'].filled(False)) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff estimated Lya from Linsky+ 13 assuming fast rotation due to known youth for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]
use_Teff2 = F.mask & (cat['Flya_1AU_Teff_schneider'].filled(nan) > 0)
print(f"Using Teff-estimated Lya (Schneider+ 19) for remaining {np.sum(use_Teff2)} targets.")
F[use_Teff2] = cat['Flya_1AU_Teff_schneider'][use_Teff2]
# make sure there aren't any masked or bad values left in F
assert np.all(F.filled(nan) > 0)
#%% flag Lya flux cut, pre-ISM
"""
In the future, if we update to a 3D ISM model, this should be updated to cut without considering the ISM. 
"""
# scale to earth
d = cat['sy_dist'].filled(np.nan)
Flya_earth_no_ISM = cat['Flya_1AU_adopted'].quantity * (1*u.AU/d)**2
Flya_earth_no_ISM = Flya_earth_no_ISM.to('erg s-1 cm-2')
valid_lya_earth = Flya_earth_no_ISM > 0
# check values are all good
keepers = cat['stage1'].filled(True)
valid_lya_earth = Flya_earth_no_ISM > 0
assert np.all(valid_lya_earth[keepers])
cat['Flya_earth_no_ISM'] = table.MaskedColumn(Flya_earth_no_ISM, mask=~valid_lya_earth,
                                              description='Flya_1AU_adopted scaled to earth distance, no ISM abosprtion.')
Flya_earth_min = 1e-15 * erg_s_cm2 # about 10x fainter than the K2-18 spectrum
FLya_basic_too_faint = cat['Flya_earth_no_ISM'].filled(0) < Flya_earth_min
FLya_basic_too_faint_str = (f'Cut because a basic, no ISM estimate of the Lya flux at earth is below {Flya_earth_min}'
                            f'erg s-1 cm-2.')
catutils.flag_cut(cat, FLya_basic_too_faint, FLya_basic_too_faint_str)
#%% transit SNR estimates
"""
This "crude" estimate takes an ad hoc transit profile, scales it to the a maximum depth equivalent to
an opaque rectangle representing an outflow tail that fills the planet's hill sphere vertically and spans
the diameter of the star in length, and shifts it blueward.
There is obvious optimism in an estimate like this. However, it has a lower 
false negative rate than a more sophisticated estimate based on an outflow model in predicting the 
transit detectability for existing known transits. Because we want to encourage detections, we make it the
basis of our target selection.
"""
params = dict(expt_out=3500, expt_in=6000, default_rv=default_sys_rv, transit_range=assumed_transit_range, integrate_range='best', show_progress=True)
SNR_optimistic = transit.opaque_tail_transit_SNR(cat, n_H_percentile=16, lya_percentile=84, **params)
assert np.all(SNR_optimistic > 0)
cat['transit_snr_optimistic'] = table.MaskedColumn(SNR_optimistic)
SNR_nominal = transit.opaque_tail_transit_SNR(cat, n_H_percentile=50, lya_percentile=50, **params)
assert np.all(SNR_nominal > 0)
cat['transit_snr_nominal'] = table.MaskedColumn(SNR_nominal)
#%% basic transit SNR cut
min_SNR = 2
low_SNR = cat['transit_snr_optimistic'].filled(-1) < min_SNR
low_SNR_str = f'Cut because an optimistic estimate of the transit SNR is below {min_SNR}.'
catutils.flag_cut(cat, low_SNR, low_SNR_str)
#%% remove manually IDed bad targets
"""
note that you will need the confirmed and toi tables for this so that the code can check that each target was
initially present and was subsequently cut if it is not found to be sure that if the planet isn't found it isn't
because its name has changed (or some other problematic reason)
"""
# first make sure the ids in the hand checked lists are all accounted for in the initial input tables
# to be certain none will be missed because of a mismatched id
if 'tois_uncut' not in locals():
    tois_uncut = catutils.load_and_mask_ecsv(paths.ipac / 'tois_uncut.ecsv')
if 'confirmed' not in locals():
    confirmed = catutils.load_and_mask_ecsv(paths.ipac / 'confirmed_uncut.ecsv')
all_checked_ids = catutils.read_hand_checked_planets(('remove', 'vetted', 'no-exofop'))
in_tois = np.in1d(all_checked_ids, tois_uncut['toi'].astype(str))
in_confirmed = np.in1d(all_checked_ids, confirmed['pl_name'])
in_input_tables = in_tois | in_confirmed
assert np.all(in_input_tables)
# if this raises an error, figure out why some targets weren't matched and fix it!
ids_to_remove = catutils.read_hand_checked_planets(('remove', 'no-exofop'))
mask_remove = np.in1d(cat['id'], ids_to_remove) | np.in1d(cat['toi'], ids_to_remove)
remove_str = "Removed because planet candidate failed manual vetting."
catutils.flag_cut(cat, mask_remove, remove_str)
for band in ['nuv', 'fuv']:
    # values must not be masked or an upper or lower limit
    mag = cat[f'sy_{band}mag'].filled(nan)
    lim_flag = cat[f'sy_{band}maglim'].filled(nan) != 0
    MK = cat['st_teff'].filled(1e4) < 4500. # because schneider GALEX-Lya relationship is for late Ks and Ms only
    d = cat['sy_dist'].filled(nan).quantity
    Flya_1AU = empirical.Lya_from_galex_schneider19(mag, d.to_value('pc'), band=band)
    Flya_1AU = Flya_1AU * erg_s_cm2
    bad_galex_lya = ~np.isfinite(Flya_1AU) | lim_flag | ~MK
    cat[f'Flya_1AU_{band}'] = table.MaskedColumn(Flya_1AU, mask=bad_galex_lya)
keepers = cat['stage1'].filled(True)
Teff = cat['st_teff'].filled(nan)
# assume stars without measured period or known youth fall on the "normal" track from Linsky+ 2013
Prot_adopt = cat['st_rotp'].filled(nan)
noP = ~(Prot_adopt > 0)
young = cat['flag_young']
Prot_adopt[young & noP] = 1
Prot_adopt[~young & noP] = 20
Flya_linsky = empirical.Lya_from_Teff_linsky13(Teff, Prot_adopt) * erg_s_cm2
valid_lya = Flya_linsky > 0
assert np.all(valid_lya[keepers]) # sanity check
cat['Flya_1AU_Teff_linsky'] = table.MaskedColumn(Flya_linsky, mask=~valid_lya,
                                                 description="Lya flux estimated based on Teff and Prot using Lisnky+ 2013.")
Flya_schneider = empirical.Lya_from_Teff_schneider19(Teff) * erg_s_cm2
valid_lya = Flya_schneider > 0
assert np.all(valid_lya[keepers])
cat['Flya_1AU_Teff_schneider'] = table.MaskedColumn(Flya_schneider, mask=~valid_lya,
                                                    description="Lya flux estimated based on Teff using Schneider+ 2019.")
catutils.add_masked_columns(cat, 'Flya_1AU_adopted', 'float', suffixes=('',), unit='erg s-1 cm-2')
F = cat['Flya_1AU_adopted'] # changes to F will appear in the cat table
fuv = cat['sy_fuvmag'].filled(nan)
fuvlim = cat['sy_fuvmaglim'].filled(nan)
nuv = cat['sy_nuvmag'].filled(nan)
nuvlim = cat['sy_nuvmaglim'].filled(nan)
Teff = cat['st_teff'].filled(nan)
Prot = cat['st_rotp'].filled(nan)
# based on Pineda 2021 for M star saturated activity level
# expected to apply to all given the similarity in saturation levels across
# spectral types for other activity generated flux
max_plausible_Llya_Lbol = 10**-3.0
max_plausible_Llya = max_plausible_Llya_Lbol * 10**cat['st_lum'] * u.Lsun
max_plausible_Flya_1AU = max_plausible_Llya / (4 * np.pi * (1*u.AU)**2)
max_plausible_Flya_1AU = max_plausible_Flya_1AU.to('erg s-1 cm-2')
use_fuv = (np.isfinite(fuv) & (fuvlim == 0) & (cat['Flya_1AU_fuv'].filled(nan) > 0)
           & (cat['Flya_1AU_fuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using FUV-estimated Lya (Schneider+ 19) for {np.sum(use_fuv)} targets.")
F[use_fuv] = cat['Flya_1AU_fuv'][use_fuv]
use_nuv = (F.mask & np.isfinite(nuv) & (nuvlim == 0) & (cat['Flya_1AU_nuv'].filled(nan) > 0)
           & (cat['Flya_1AU_nuv'].filled(0) < max_plausible_Flya_1AU))
print(f"Using NUV-estimated Lya (Schneider+ 19) for {np.sum(use_nuv)} targets.")
F[use_nuv] = cat['Flya_1AU_nuv'][use_nuv]
use_Teff1 = F.mask & (Teff > 0) & (Prot > 0) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff+Prot-estimated Lya (Linsky+ 13) for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]
use_Teff1 = F.mask & (Teff > 0) & ~(Prot > 0) & (cat['flag_young'].filled(False)) & (cat['Flya_1AU_Teff_linsky'].filled(nan) > 0)
print(f"Using Teff estimated Lya from Linsky+ 13 assuming fast rotation due to known youth for {np.sum(use_Teff1)} targets.")
F[use_Teff1] = cat['Flya_1AU_Teff_linsky'][use_Teff1]
use_Teff2 = F.mask & (cat['Flya_1AU_Teff_schneider'].filled(nan) > 0)
print(f"Using Teff-estimated Lya (Schneider+ 19) for remaining {np.sum(use_Teff2)} targets.")
F[use_Teff2] = cat['Flya_1AU_Teff_schneider'][use_Teff2]
# make sure there aren't any masked or bad values left in F
assert np.all(F.filled(nan) > 0)
np.sum(F.mask)
np.sum(cat['st_teff'].mask)
"""
In the future, if we update to a 3D ISM model, this should be updated to cut without considering the ISM. 
"""
# scale to earth
d = cat['sy_dist'].filled(np.nan)
Flya_earth_no_ISM = cat['Flya_1AU_adopted'].quantity * (1*u.AU/d)**2
Flya_earth_no_ISM = Flya_earth_no_ISM.to('erg s-1 cm-2')
valid_lya_earth = Flya_earth_no_ISM > 0
# check values are all good
keepers = cat['stage1'].filled(True)
valid_lya_earth = Flya_earth_no_ISM > 0
assert np.all(valid_lya_earth[keepers])
cat['Flya_earth_no_ISM'] = table.MaskedColumn(Flya_earth_no_ISM, mask=~valid_lya_earth,
                                              description='Flya_1AU_adopted scaled to earth distance, no ISM abosprtion.')
Flya_earth_min = 1e-15 * erg_s_cm2 # about 10x fainter than the K2-18 spectrum
FLya_basic_too_faint = cat['Flya_earth_no_ISM'].filled(0) < Flya_earth_min
FLya_basic_too_faint_str = (f'Cut because a basic, no ISM estimate of the Lya flux at earth is below {Flya_earth_min}'
                            f'erg s-1 cm-2.')
catutils.flag_cut(cat, FLya_basic_too_faint, FLya_basic_too_faint_str)
"""
This "crude" estimate takes an ad hoc transit profile, scales it to the a maximum depth equivalent to
an opaque rectangle representing an outflow tail that fills the planet's hill sphere vertically and spans
the diameter of the star in length, and shifts it blueward.
There is obvious optimism in an estimate like this. However, it has a lower 
false negative rate than a more sophisticated estimate based on an outflow model in predicting the 
transit detectability for existing known transits. Because we want to encourage detections, we make it the
basis of our target selection.
"""
params = dict(expt_out=3500, expt_in=6000, default_rv=default_sys_rv, transit_range=assumed_transit_range, integrate_range='best', show_progress=True)
SNR_optimistic = transit.opaque_tail_transit_SNR(cat, n_H_percentile=16, lya_percentile=84, **params)
assert np.all(SNR_optimistic > 0)
cat['transit_snr_optimistic'] = table.MaskedColumn(SNR_optimistic)
SNR_nominal = transit.opaque_tail_transit_SNR(cat, n_H_percentile=50, lya_percentile=50, **params)
assert np.all(SNR_nominal > 0)
cat['transit_snr_nominal'] = table.MaskedColumn(SNR_nominal)
cat['transit_snr_optimistic'] = table.MaskedColumn(SNR_optimistic)
SNR_nominal = transit.opaque_tail_transit_SNR(cat, n_H_percentile=50, lya_percentile=50, **params)
cat['transit_snr_nominal'] = table.MaskedColumn(SNR_nominal)
#%% basic transit SNR cut
min_SNR = 2
low_SNR = cat['transit_snr_optimistic'].filled(-1) < min_SNR
low_SNR_str = f'Cut because an optimistic estimate of the transit SNR is below {min_SNR}.'
catutils.flag_cut(cat, low_SNR, low_SNR_str)
#%% remove manually IDed bad targets
"""
note that you will need the confirmed and toi tables for this so that the code can check that each target was
initially present and was subsequently cut if it is not found to be sure that if the planet isn't found it isn't
because its name has changed (or some other problematic reason)
"""
# first make sure the ids in the hand checked lists are all accounted for in the initial input tables
# to be certain none will be missed because of a mismatched id
if 'tois_uncut' not in locals():
    tois_uncut = catutils.load_and_mask_ecsv(paths.ipac / 'tois_uncut.ecsv')
if 'confirmed' not in locals():
    confirmed = catutils.load_and_mask_ecsv(paths.ipac / 'confirmed_uncut.ecsv')
all_checked_ids = catutils.read_hand_checked_planets(('remove', 'vetted', 'no-exofop'))
in_tois = np.in1d(all_checked_ids, tois_uncut['toi'].astype(str))
in_confirmed = np.in1d(all_checked_ids, confirmed['pl_name'])
in_input_tables = in_tois | in_confirmed
assert np.all(in_input_tables)
# if this raises an error, figure out why some targets weren't matched and fix it!
ids_to_remove = catutils.read_hand_checked_planets(('remove', 'no-exofop'))
mask_remove = np.in1d(cat['id'], ids_to_remove) | np.in1d(cat['toi'], ids_to_remove)
remove_str = "Removed because planet candidate failed manual vetting."
catutils.flag_cut(cat, mask_remove, remove_str)
Flya_1au = cat['Flya_1AU_adopted'].filled(nan).data
Teff = cat['st_teff'].filled(nan).data
Feuv_1au = empirical.EUV_Linsky14(Flya_1au, Teff)
assert np.all(Feuv_1au > 0)
Feuv_1au *= erg_s_cm2
cat['Feuv_1AU'] = table.MaskedColumn(Feuv_1au)
a = cat['pl_orbsmax'].filled(nan).quantity
Feuv_at_planet = Feuv_1au * (1*u.AU/a)**2
assert np.all(Feuv_at_planet > 0)
cat['Feuv_at_planet'] = table.MaskedColumn(Feuv_at_planet)
keep = ~cat['st_teff'].mask
sum(keep)
len(cat)
cat = cat[keep]
Flya_1au = cat['Flya_1AU_adopted'].filled(nan).data
Teff = cat['st_teff'].filled(nan).data
Feuv_1au = empirical.EUV_Linsky14(Flya_1au, Teff)
assert np.all(Feuv_1au > 0)
Feuv_1au *= erg_s_cm2
cat['Feuv_1AU'] = table.MaskedColumn(Feuv_1au)
a = cat['pl_orbsmax'].filled(nan).quantity
Feuv_at_planet = Feuv_1au * (1*u.AU/a)**2
assert np.all(Feuv_at_planet > 0)
cat['Feuv_at_planet'] = table.MaskedColumn(Feuv_at_planet)
"""planets near gap"""
Rgap = empirical.ho_gap_lowlim(cat['pl_orbper'], cat['st_mass'])
dR_gap_dex = np.log10(cat['pl_rade'] / Rgap)
cat['flag_gap_upper_cusp'] = (dR_gap_dex > 0) & (dR_gap_dex < 0.1)
# water worlds
# consider them water world compatible if they are within 1 sigma of the Luque & Palle 2022 water world line"""
lp22 = ref.lp22
pp = lp22['m'] * u.Mearth / (4 / 3 * np.pi * (lp22['r'] * u.Rearth) ** 3)
pp = pp.to_value('g cm-3')
p_water = np.interp(cat['pl_bmasse'], lp22['m'], pp, left=np.nan, right=np.nan)
rho_diff_sig = np.abs((cat['pl_dens'] - p_water) / cat['pl_denserr1'])
water_world = (rho_diff_sig < 1) & (cat['pl_bmasse'] < 30) & (cat['pl_rade'] < 3)
cat['flag_water_world'] = water_world
# flag escape on/off multis
above_euv_hydro_threshold = cat['Feuv_at_planet'] > 1 # lower xuv threshold to catch stars where code likely underpredicts xuv flux
has_HHe = cat['flag_gaseous']
has_outflow = has_HHe & above_euv_hydro_threshold
no_outflow = has_HHe & ~above_euv_hydro_threshold
print(f'{sum(no_outflow)} planets in catalog are gaseous but too weakly irradiated to support an outflow.')
cat['flag_outflow'] = table.MaskedColumn(has_outflow)
positions_cut3 = coord.SkyCoord(cat['ra'], cat['dec'])
positions_outflow = positions_cut3[has_outflow.filled(False)]
positions_no_outflow = positions_cut3[no_outflow.filled(False)]
i_matches_no_outflow, i_matches_outflow, _, _ = positions_outflow.search_around_sky(positions_no_outflow, seplimit=0.000001*u.deg)
i_outflow, = np.nonzero(has_outflow.filled(False))
i_no_outflow, = np.nonzero(no_outflow.filled(False))
on_off_multis = np.hstack((i_outflow[i_matches_outflow], i_no_outflow[i_matches_no_outflow]))
on_off_multis = np.unique(on_off_multis)
_description = 'systems contains planets with and without H/He envelopes'
cat['flag_outflow_and_not_in_sys'] = table.MaskedColumn([False] * len(cat), fill_value=False,
                                                        description=_description)
cat['flag_outflow_and_not_in_sys'][on_off_multis] = True
# HHe + no HHe multis
positions_cut3 = coord.SkyCoord(cat['ra'], cat['dec'])
has_HHe = cat['flag_gaseous']
positions_HHe = positions_cut3[has_HHe.filled(False)]
positions_noHHe = positions_cut3[(~has_HHe).filled(False)]
i_matches_noHHe, i_matches_HHe, _, _ = positions_HHe.search_around_sky(positions_noHHe, seplimit=0.000001*u.deg)
i_HHe, = np.nonzero(has_HHe.filled(False))
i_noHHe, = np.nonzero((~has_HHe).filled(False))
H_no_H_multis = np.hstack((i_HHe[i_matches_HHe], i_noHHe[i_matches_noHHe]))
H_no_H_multis = np.unique(H_no_H_multis)
_description = 'systems contains planets with and without H/He envelopes'
cat['flag_gas_and_rocky_in_sys'] = table.MaskedColumn([False] * len(cat), fill_value=False,
                                                      description=_description)
cat['flag_gas_and_rocky_in_sys'][H_no_H_multis] = True
#  super puffs
p_sp = 0.3  # based on Libby-Roberts 2020
super_puff = (cat['pl_dens'] < p_sp) & (cat['pl_bmasse'] < 30)
cat['flag_super_puff'] = super_puff
# high TSM
Rp = cat['pl_rade']#.quantity.to_value('Rearth')
Teq = cat['pl_eqt']#.quantity.to_value('K')
Mp = cat['pl_bmasse']#.quantity.to_value('Mearth')
Rs = cat['st_rad']#.quantity.to_value('Rsun')
J = cat['sy_jmag']#.quantity.to_value('')
R1 = (Rp < 1.5)
R2 = (Rp >= 1.5) & (Rp < 2.)
R3 = (Rp >= 2) & (Rp < 2.75)
R4 = (Rp >= 2.75) & (Rp < 4)
R5 = (Rp >= 4) & (Rp < 10.)
TSM_raw = Rp ** 3 * Teq / Mp / Rs ** 2 * 10 ** (-J / 5)
cat['TSM'] = TSM_raw
cat['TSM'][R1] = TSM_raw[R1] * 0.19
cat['TSM'][R2] = TSM_raw[R2] * 1.26 * 2.3 / 18
cat['TSM'][R3] = TSM_raw[R3] * 1.26
cat['TSM'][R4] = TSM_raw[R4] * 1.28
cat['TSM'][R5] = TSM_raw[R5] * 1.15
cat['flag_high_TSM'] = (((cat['pl_rade'] <= 1.5) & (cat['TSM'] > 10))
                        | ((cat['pl_rade'] > 1.5) & (cat['TSM'] > 90)))
# flag measured mass
cat['flag_measured_mass'] = cat['pl_bmassprov'] == 'Mass'
# flag Lya or FUV already
cat['flag_has_FUV_or_Lya_already'] = (cat['external_lya'].astype(float) + cat['external_fuv'].astype(float)) == 1
cat['score'] = cat['transit_snr_nominal']
# zero out score for planets where atm escape known to be unlikely
outflow_unlikely = ~cat['flag_gaseous'] | ~cat['flag_outflow']
zero_out = outflow_unlikely.filled(False)
cat['score'][zero_out] = 0
# pick top score for each host
catutils.pick_planet_parameters(cat, 'score', np.max, 'score_host')
_, unq_inverse = np.unique(cat['score_host'], return_inverse=True)
rank = unq_inverse.max() - unq_inverse + 1
cat['stage1_rank'] = table.MaskedColumn(rank)
mask = cat['external_lya'].filled(False)
sum(mask)
archive = cat[mask]
archive_hosts = catutils.planets2hosts(archive)
len(archive_hosts)
archive_lowsnr = archive[archive['transit_snr_optimistic'].filled(0) < 2]
len(archive_lowsnr)
archive_lowsnr_no_fuv = archive_lowsnr[~archive_lowsnr['external_fuv'].filled(False)]
np.unique(archive_lowsnr_no_fuv['hostname'])
archive.write(paths.selection_outputs / '2025-06-13_catalog_of_any_planets_with_external_lya.ecsv')