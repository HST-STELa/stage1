
from tqdm import tqdm

from stage1_processing import target_lists
from stage1_processing import observation_table as obt

#%%
dry_run = False
targets = target_lists.everything_in_database()

#%%

ignore_substrings = [
    'flux within central tile',
    'telemetry indicates that',
    'the flux in the third image',
    'despite acquisition issues',
]

for target in targets:
    obstbl = obt.load_obs_tbl(target)
    fix_mask = []
    for row in obstbl:
        fix = False
        flags = row.get('flags', [])
        notes = row.get('notes', [])
        if len(flags) == 1 and flags[0] == 'acquisition untrustworthy':
            acq_notes = []
            human_pass = False
            for note in notes:
                note = note.lower()
                if not ('acq' in note or 'final centering' in note):
                    continue
                if any(ss in note for ss in ignore_substrings):
                    continue
                if 'could not identify target' in note:
                    human_pass = False
                elif 'was able to identify target' in note:
                    human_pass = True
                else:
                    acq_notes.append(note)
            if len(acq_notes) == 0 and human_pass:
                fix = True
                flags.remove('acquisition untrustworthy')
                if flags == []:
                    obstbl['flags'][row.index] = None
                    obstbl['flags'].mask[row.index] = True
                else:
                    obstbl['flags'][row.index] = flags
        fix_mask.append(fix)

    if any(fix_mask):
        obstbl.update_usability(fix_mask, 'all clear')
        if dry_run:
            print(target.upper())
            print(obstbl[fix_mask].pretty_string_with_flags_notes())
        else:
            tblpth = obstbl.get_path(target)
            obstbl.write(tblpth, overwrite=True)