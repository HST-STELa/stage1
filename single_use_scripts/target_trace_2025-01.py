
from pathlib import Path

import numpy as np
from astropy import table
from matplotlib import pyplot as plt

import empirical
import paths
import catalog_utilities as catutils
from lya_prediction_tools import heritage, transit



#%% load old and new catalogs

def reformat_tic_id(cat):
    ids = cat['tic_id']
    ids = np.char.replace(ids, 'TIC ', '')
    ids = ids.astype(int)
    cat['tic_id'] = ids

# phase I
catdir = Path('/Users/parke/Google Drive/Research/STELa/proposals/HST proposal/target search/2024-02 search')
files = [min(catdir.glob(f'{i}_*.ecsv')) for i in range(1,7) if i != 3]
can1cuts = [catutils.load_and_mask_ecsv(file) for file in files]
path1 = '/Users/parke/Google Drive/Research/STELa/phase IIs/cycle 32/input_catalogs/preliminary_targets_2024-02.ecsv'
candidates1 = catutils.load_and_mask_ecsv(path1)
for cat in can1cuts:
    reformat_tic_id(cat)
reformat_tic_id(candidates1)

# first phase II submission
path2 = '/Users/parke/Google Drive/Research/STELa/phase IIs/cycle 32/output_catalogs/cycle32_processed_targets.ecsv'
chosen2 = catutils.load_and_mask_ecsv(path2)
candidates2 = catutils.load_and_mask_ecsv(paths.other / 'initial_phase2_values.ecsv')
chosen2 = chosen2[~chosen2['tic_id'].mask]
candidates2 = candidates2[~candidates2['tic_id'].mask]
reformat_tic_id(chosen2)
reformat_tic_id(candidates2)

# latest candidates and selections
can3cut1 = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt3__fill-basic_properties.ecsv')
can3cut2 = catutils.load_and_mask_ecsv(
    paths.selection_intermediates / 'chkpt4__cut-planet_host_types__add-galex_lya_transit_snr.ecsv')
can3cut3 = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt6__add-categories_scores.ecsv')
chosen3 = can3cut3[can3cut3['stage1']]

#%% make tic id the index of all catalogs

catalogs = [candidates1, candidates2, chosen2, can3cut1, can3cut2, can3cut3, chosen3] + can1cuts
for cat in catalogs:
    cat.add_index('tic_id')


#%% function to deal with differing SNRs in multiplanet systems

def max_snr_in_system(cat, id, snrcol):
    snrs = cat.loc[id][snrcol]
    snrs = np.atleast_1d(snrs)
    return max(snrs)

snr_threshold = min(max_snr_in_system(chosen3, id, 'transit_snr_nominal') for id in chosen3['tic_id'])


#%% examine what happened to targets no longer selected

dropped = ~np.isin(chosen2['tic_id'], chosen3['tic_id'])
dropped_ids = np.unique(chosen2['tic_id'][dropped])

flag_dictionary = {1: 'First pass SNR below latest threshold.',
                   2: 'Second pass SNR below latest threshold.',
                   3: 'Final SNR below latest threshold.',
                   4: 'Target cut early in final pass for reason printed during flagging.'}

flags = []
for id in dropped_ids:
    if id in candidates1['tic_id']:
        snr1 = max_snr_in_system(candidates1, id, 'pl_lya_tnst_snr_nominal')
        if snr1 < snr_threshold:
            flags.append(1)
            continue
    snr2 = max_snr_in_system(candidates2, id, 'lya_transit_snr_crude')
    if snr2 < snr_threshold:
        flags.append(2)
        continue
    if id in can3cut3['tic_id']:
        snr3 = max_snr_in_system(can3cut3, id, 'transit_snr_nominal')
        if snr3 < snr_threshold:
            flags.append(3)
            continue
    assert id in can3cut1['tic_id']
    for cut in [can3cut3, can3cut2, can3cut1]:
        if id in cut['tic_id']:
            print()
            print(id)
            try:
                cut.loc[id][['id', 'decision']].pprint(-1,-1)
            except AttributeError:
                print(cut.loc[id][['id', 'decision']])
            flags.append(4)
            break
flags = np.array(flags)


#%% examine targets not present in initial phase ii

added = ~np.isin(chosen3['tic_id'], chosen2['tic_id'])
added_ids = np.unique(chosen3['tic_id'].filled(0)[added])

cat = can1cuts[-1]
Rgap = empirical.ho_gap_lowlim(cat['pl_orbper'], cat['st_mass'])
cat['flag_abovegap'] = cat['pl_rade'] > Rgap.filled(1.8)

flag_dictionary = {1: 'Target not known at proposal submission.',
                   2: 'Target did not make it to SNR calculation stage in proposal search (see printout for where it was cut).',
                   3: 'Target did not make short list in proposal search because optimistic SNR < 3.',
                   4: 'Target dropped after SNR calculation but not because of SNR.',
                   5: 'SNR increased from pass 1.',
                   6: 'SNR decreased from pass 1, but target rose in ranking due to other targets falling.'}

flags = []
for id in added_ids:
    if id not in can1cuts[0]['tic_id']:
        flags.append(1)
        continue
    if id not in candidates1['tic_id']:
        cat = can1cuts[-1]
        if id not in cat['tic_id']:
            flags.append(2)
            print()
            print(id)
            for i, cat in enumerate(can1cuts):
                if id not in cat['tic_id']:
                    print(f'Target cut prior to {files[i].name}')
                    break
            continue
        else:
            snr1 = max_snr_in_system(cat, id, 'pl_lya_tnst_snr_nominal')
            snr3 = max_snr_in_system(can3cut3, id, 'transit_snr_nominal')
            if snr1<= 3:
                print('')
                print(f'Nominal SNR for {id} increased from {snr1:.1f} to {snr3:.1f}')
                flags.append(3)
            else:
                HHe = cat.loc[id]['flag_abovegap'].tolist()
                TOIs = cat.loc[id]['toi'].tolist()
                print('')
                print(f'{id} SNR > 3, TOIs {TOIs} above gap {HHe}')
                flags.append(4)
            continue
    snr1 = max_snr_in_system(candidates1, id, 'pl_lya_tnst_snr_nominal')
    snr3 = max_snr_in_system(can3cut3, id, 'transit_snr_nominal')
    if snr3 < snr1:
        flags.append(5)
        continue
    flags.append(6)
flags = np.array(flags)


#%% comparing SNR calcs


def snr_trace(ids, SNRfunc):

    can1 = can1cuts[-1]
    catutils.set_index(can1, 'tic_id')
    ids_in_1 = np.in1d(ids, can1['tic_id'])
    ids_in_3 = np.in1d(ids, can3cut2['tic_id'])
    ids = ids[ids_in_1 & ids_in_3]
    matches1 = can1.loc[ids]
    pl_ids = matches1['id']
    in3 = np.in1d(pl_ids, can3cut2['id'])
    pl_ids = pl_ids[in3]

    catutils.set_index(matches1, 'id')
    catutils.set_index(can3cut2, 'id')
    matches1 = matches1.loc[pl_ids]
    matches3 = can3cut2.loc[pl_ids]

    # make sure the heritage SNR calculator is working
    checkSNRs = heritage.transit_snr_proposal(matches1)
    assert np.all(np.isclose(checkSNRs, matches1['pl_lya_tnst_snr_nominal']))

    mod = matches1.copy()
    SNRs = [checkSNRs]
    labels = []
    def newSNR(label):
        labels.append(label)
        SNRs.append(SNRfunc(mod))

    mod['Flya_1AU_adopted'] = mod['Flya_at_1AU']
    newSNR('SNR function')

    mod['Flya_1AU_adopted'] = matches3['Flya_1AU_adopted']
    newSNR('Flya')

    for param in 'pl_orbsmax pl_bmasse st_mass st_radv st_rad'.split()[::-1]:
        mod[param] = matches3[param]
        newSNR(param)

    SNRs = np.array(SNRs)
    SNRdiffs = np.diff(SNRs, axis=0) / (SNRs[-1,:] - SNRs[0,:])

    difftbl = table.Table(SNRdiffs.T, names=labels)
    for label in labels:
        difftbl[label].format = '.2f'
    difftbl['id'] = mod['id']
    difftbl = difftbl[['id'] + labels]
    difftbl.pprint(-1, -1)

    main_diff = np.argmax(SNRdiffs, axis=0)
    for i, label in enumerate(labels):
        print(f'{sum(main_diff == i)} largest change due to {label}')


def SNRfunc(cat):
    params = dict(expt_out=3500, expt_in=6000, default_rv=0, tail_shift=-50., integrate='best', show_progress=True)
    return transit.opaque_tail_transit_SNR(cat, lya_1AU_colname='Flya_1AU_adopted', **params)

# you might need to fiddle with FLya_at_1AU vs Flya_1AU_adopted to do what you want with this
# def SNRfunc(cat):
#     return heritage.transit_snr_proposal(cat)


#%% histogram SNRs

ids2 = chosen2['tic_id']
in2 = np.isin(ids2, can3cut2['tic_id'])
ids2 = ids2[in2]
SNR2 = can3cut2.loc[ids2]['transit_snr_nominal']
SNR3 = catutils.planets2hosts(chosen3)['score_host']

plt.figure()
bins = np.arange(0, 30, 1/4)
_ = plt.hist(SNR2, bins, alpha=0.5, label='Initial Phase II')
_ = plt.hist(SNR3, bins, alpha=0.5, label='2024-01-06 Build')