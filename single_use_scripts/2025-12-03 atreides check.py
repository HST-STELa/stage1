from the_usuals import *

import catalog_utilities as catutils

#%%

atr_str = """HAT-P-26 b
HATS-12 b
HATS-37 A b
HATS-38 b
HATS-7 b
HD 93963 A c
K2-100 b
K2-108 b
K2-334 b
K2-370 b
K2-39 b
K2-405 b
K2-60 b
NGTS-14 A b
TOI-132 b
TOI-181 b
TOI-2374 b
TOI-2498 b
TOI-3071 b
TOI-620 b
TOI-3261 b
TOI-942 b
TOI-1853 b
TOI-2196 b
LP 714-47 b"""
atr_targets = atr_str.split('\n')

#%%


def load_cat(path):
    cat = catutils.load_and_mask_ecsv(path)
    keep = ~cat['pl_name'].mask
    cat = cat[keep]
    cat['pl_name'] = cat['pl_name'].astype(str)
    cat.add_index('pl_name')
    return cat


def get_info(cat, cols):
    avail_cols = [col for col in cols if col in cat.colnames]
    present = np.isin(atr_targets, cat['pl_name'])
    avail_targs = np.array(atr_targets)[present]
    slim = cat.loc[avail_targs]
    return slim[avail_cols]


def combine(newcat, oldcat, cols):
    i = oldcat.loc_indices['pl_name', newcat['pl_name'].tolist()]
    for col in cols:
        if col == 'pl_name':
            continue
        if col not in oldcat.colnames:
            if col not in newcat.colnames:
                continue
            else:
                oldcat[col] = table.MaskedColumn(length=len(oldcat), mask=True, dtype=newcat[col].dtype)

        newvals = newcat[col]
        oldcat[col][i] = newvals


#%%
cols = 'pl_name Flya_earth_no_ISM transit_snr_nominal stage1 stage1_rank decision'.split()
root = Path('/Users/parke/Repos/STELa/stage1/target_selection_data/intermediates')


for i in range(1,9):
    path, = root.glob(f'chkpt{i}_*.ecsv')
    cat = load_cat(path)
    slim = get_info(cat, cols)

    if i == 1:
        result = slim
        continue

    combine(slim, result, cols)
