from matplotlib import pyplot as plt
from astropy import table
import numpy as np

import paths

from processing import target_lists

#%%

fd = paths.inbox / '2026-04-05 fuv line fluxes'

tbl_files = sorted(fd.rglob('*line-flux-table.ecsv'))

#%% plot all line fluxes

fig, ax = plt.subplots(1,1)
ax.set_xscale('linear')
ax.set_yscale('log')

for tbl_file in tbl_files:
    tgt = tbl_file.name.split('.')[0]
    tbl = table.Table.read(tbl_file)
    both_valid = ~tbl['flux'].mask
    snr = tbl['snr']
    if np.any(tbl['snr'][both_valid] < -3):
        print(f"{tgt} has line fluxes with < -3σ values")
    if np.any(~np.isfinite(tbl['flux'])):
        print(f"{tgt} has non-finite fluxes")
    m_scaling = np.char.count(tbl['source'].filled('').astype(str), 'correlation') > 0
    ln, = plt.plot(tbl['wave'][both_valid], tbl['flux'][both_valid], '-')
    color = ln.get_color()
    closed = both_valid & ~m_scaling
    open = both_valid & m_scaling
    plt.plot(tbl['wave'][closed], tbl['flux'][closed], 'o', fillstyle='full', color=color)
    plt.plot(tbl['wave'][open], tbl['flux'][open], 'o', fillstyle='none', color=color)

    tbl.add_index('name')
    ilya = tbl.loc_indices['Lya']
    ax.annotate(tgt + ' ', xy=(tbl['wave'][ilya], tbl['flux'][ilya]),
                ha='right', va='center', fontsize='x-small')
    
fig.tight_layout()


#%% inspect tables with low fluxes

# tgt = 'hd22946'
# tgt = 'hr858'
tgt = 'toi-1898'

f, = [f for f in tbl_files if tgt in f.name]
tbl = table.Table.read(f)

tbl.pprint(-1, -1)


"""Determined they're all fine. It's either Si II or Al II near the end of the range that are < 3 sigma
and things are noisy there with edge effects."""


#%% check that all of the eval 3 targets have a line flux table

tgts3 = target_lists.eval_no(3)
file_tgts = [f.name.split('.')[0] for f in tbl_files]

missing = set(tgts3) - set(file_tgts)

"""GJ 3929 is missing."""

#%% utility to combine fluxes of consecutive rows with the same line name (multiplets)
def combine_consecutive_multiplets(tbl):
    """
    Combine fluxes (and errors) in a line flux table if consecutive rows have the same 'name'.

    For each group of consecutive rows with the same 'name':
      - Add fluxes and errors (quadrature) into the *top* row of the group.
      - Flux and error in the lower rows of the group are masked.
      - All other columns are left unchanged.

    Returns a new astropy Table. The output table has the original structure, with updated/masked flux/error.
    """
    import numpy as np
    from astropy.table import Table, MaskedColumn

    # Copy to avoid editing input
    t = tbl.copy()

    # Ensure flux and error are MaskedColumn for masking
    if not isinstance(t['flux'], MaskedColumn):
        t['flux'] = MaskedColumn(t['flux'], mask=np.zeros(len(t), dtype=bool))
    if not isinstance(t['error'], MaskedColumn):
        t['error'] = MaskedColumn(t['error'], mask=np.zeros(len(t), dtype=bool))

    n = len(t)
    i = 0
    while i < n:
        this_name = t['name'][i]
        # Find how many consecutive rows share this name
        j = i + 1
        while j < n and t['name'][j] == this_name:
            j += 1

        group_idx = np.arange(i, j)
        if len(group_idx) > 1:
            # Only operate on the flux/error columns
            fluxes = t['flux'][group_idx]
            errors = t['error'][group_idx]
            valid = ~fluxes.mask & np.isfinite(fluxes)
            valid_err = ~errors.mask & np.isfinite(errors)
            # For sum: treat masked/invalid as zero
            flux_sum = np.sum(fluxes[valid]) if np.any(valid) else np.ma.masked
            error_sum = np.sqrt(np.sum(errors[valid_err] ** 2)) if np.any(valid_err) else np.ma.masked

            # Write sum into the top row, and mask in all subsequent rows
            t['flux'][i] = flux_sum
            t['error'][i] = error_sum
            if len(group_idx) > 1:
                t['flux'][group_idx[1:]] = np.ma.masked
                t['error'][group_idx[1:]] = np.ma.masked

        i = j

    return t

# Example usage:
# new_tbl = combine_consecutive_multiplets(tbl)

#%% look for major changes in line fluxes

file_tgts = [f.name.split('.')[0] for f in tbl_files]
rel_diff_thresh = 0.5
frac_lines_thres = 0.5
ion_state_rng_thresh = 1

itertgts = iter(file_tgts)
i = 1
substantial_change_list = []

viewcols = [
    'name',
    'wave',
    'flux_a',
    'flux_p',
    'error_a',
    'error_p',
    'snr_a',
    'snr_p',
    'source_a',
    'source_p',
    'diff',
    'diff rel'
]

ion_state_map = {
    'I': 0,
    'II': 1,
    'III': 2,
    'IV': 3,
    'V': 4,
    'VI': 5,
    'VII': 6,
    'VIII': 7,
    'IX': 8,
    'X': 9,
}


for tgt in file_tgts:
    i = file_tgts.index(tgt)
    print(f"{i+1}/{len(file_tgts)}: {tgt}")
    srch_name = f"{tgt}*line-flux-table.ecsv"
    fa, = fd.rglob(srch_name)
    fdp = paths.target_data(tgt)
    try:
        fp, = fdp.rglob(srch_name)
    except ValueError:
        print(f"No pre-existing table for {tgt}.")
        print()
        print('=' * 30)
        continue

    subst_change = False

    tbls = []
    for f in (fa, fp):
        t = table.Table.read(f)
        t = combine_consecutive_multiplets(t)
        tbls.append(t)
    ta, tp = tbls
    tj = table.join(ta, tp, keys=['name', 'wave'], table_names=['a', 'p'])

    stats = []
    for suffix in ('_a', '_p'):
        F = tj['flux' + suffix].filled(np.nan)
        E = tj['error' + suffix].filled(np.nan)
        src = tj['source' + suffix].filled('').astype(str)
        snr = F/E
        valid = np.isfinite(F) & (F != 0)
        corr = np.char.count(src, 'correlation') > 0
        ion_state = [ion_state_map[name.split()[1]] if name != 'Lya' else 0 for name in tj['name']]
        ion_state = np.array(ion_state)
        usable = (
                    ((np.isfinite(snr) & (snr > 3)) |
                     corr) &
                  valid
        )
        ion_rng = max(ion_state[usable]) - min(ion_state[usable])
        x = dict(suffix=suffix, F=F, valid=valid, ion_rng=ion_rng)
        stats.append(x)
    a, p = stats

    # Find lines present in p but disappear in a:
    disappeared_mask = p['valid'] & (~a['valid'])
    if np.any(disappeared_mask):
        print('Lines present in previous _p table but disappear from current _a table:')
        tj[disappeared_mask].pprint(-1, -1)
        print()
    else:
        print('No lines disappeared from p to a.')
        print()

    both_valid = a['valid'] & p['valid']
    one_valid = a['valid'] | p['valid']

    print(f"Ion state range changed from {p['ion_rng']} to {a['ion_rng']}")
    if a['ion_rng'] - p['ion_rng'] >= ion_state_rng_thresh:
        print('\tMarked substantial.')
        subst_change = True
    else:
        print('\tNot marked substantial.')
    print()

    with np.errstate(divide='ignore', invalid='ignore'):
        diff = a['F'] - p['F']
        diff_rel = diff/np.abs(p['F'])
        tj['diff'] = diff
        tj['diff'].format = '.1e'
        tj['diff rel'] = diff_rel
        tj['diff rel'].format = '.2f'
        subst_diffs = np.abs(diff_rel[both_valid]) > rel_diff_thresh
    if np.any(subst_diffs):
        ns, nt = np.sum(subst_diffs), np.sum(both_valid)
        frac_lines = ns/nt
        print(f'{ns}/{nt} = {frac_lines:.2f} lines have > {rel_diff_thresh} change')
        if frac_lines > frac_lines_thres:
            print('\tMarked substantial.')
            subst_change = True
        else:
            print('\tNot marked substantial.')
    else:
        print(f'No lines have > {rel_diff_thresh} change')
    print()

    if subst_change:
        substantial_change_list.append(tgt)
        tj[viewcols][one_valid].pprint(-1, -1)
        print()
        print('Added to substantial change list.')
    else:
        print('No substantial changes noted.')

    print()
    print('='*30)


"""
sometimes ines are filled with correlations when I don't expect it, but that's just because they had low snr
and the script fills those with correlation values

otherwise all look reasonable except that O V in ds-tuc is too high
"""


#%% mask O V in ds-tuc table

tgt = 'ds-tuc-a'
i = file_tgts.index(tgt)
srch_name = f"{tgt}*line-flux-table.ecsv"
fa, = fd.rglob(srch_name)
tbl = table.Table.read(fa)
tbl.add_index('name')
i = tbl.loc_indices['O V']
cols = ['wa', 'wb', 'flux', 'error', 'snr', 'source']
for name in cols:
    tbl[name].mask[i] = True

tbl.write(fa, overwrite=True)

