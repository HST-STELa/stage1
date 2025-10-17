from pathlib import Path
import os
import re
import warnings
from collections import defaultdict
from itertools import cycle

import numpy as np
from astropy.io import fits
from astropy import table

import paths
import utilities as utils
from target_selection_tools import query


hst2simbad_map = table.Table.read(paths.checked / 'odd_name_map.ecsv')
hst2simbad_map.add_index('odd')
stela_name_tbl = table.Table.read(paths.locked / 'stela_names.csv')
stela_name_tbl.add_index('tic_id')
stela_name_tbl.add_index('hostname')
stela_name_tbl.add_index('hostname_file')
stela_name_tbl.add_index('hostname_hst')


def rename_replace(files, pattern, repl, dry_run):
    for f in files:
        newname = re.sub(pattern, repl, f.name)
        if dry_run:
            print(f"{f.name} --> {newname}")
        else:
            os.rename(f, f.parent / newname)


def resolve_stela_name_w_simbad(names_for_simbad):
    tic_ids = query.query_simbad_for_tic_ids(names_for_simbad)
    stela_names = []
    no_tids_in_simbad = []
    tids_not_in_stela = []
    for name, tids in zip(names_for_simbad, tic_ids):
        tids = list(map(int, tids.split()))
        if tids:
            valid = np.in1d(tids, stela_name_tbl['tic_id'])
            nvalid = sum(valid)
            if nvalid > 1:
                raise ValueError('A single target had multiple TIC IDs in SIMBAD, '
                                 'and they matched to different STELa targets.')
            elif nvalid == 1:
                valid_id, = np.asarray(tids)[valid]
                stela_name = stela_name_tbl.loc[valid_id]['hostname']
                stela_names.append(stela_name)
            else:
                tids_not_in_stela.append(name)
                stela_names.append('no match')
        else:
            no_tids_in_simbad.append(name)
            stela_names.append('no match')

        if tids_not_in_stela:
            msg = 'The TIC ID(s) for these targets had no matches in the STELa names table:\n\t'
            msg += '\n\t'.join(tids_not_in_stela)
            warnings.warn(msg)
        if no_tids_in_simbad:
            msg = ('These targets were not resovled by SIMBAD (see warning above for list) '
                   'or had no TIC IDs in SIMBAD:\n\t')
            msg += '\n\t'.join(no_tids_in_simbad)
            warnings.warn(msg)

    return stela_names


def resolve_stela_name_flexible(names):
    names = np.asarray(names)
    found = np.zeros(len(names), dtype=bool)
    stela_names = np.empty(len(names), dtype='object')

    # search for the name in stela columns
    for colname in stela_name_tbl.colnames:
        incol = np.isin(names, stela_name_tbl[colname])
        if np.any(incol):
            found |= incol
            stela_names[incol] = stela_name_tbl.loc[colname, names[incol]]['hostname']

    # search for the name in simbad
    try:
        names_from_simbad_search = resolve_stela_name_w_simbad(names[~found])
    except ValueError as e:
        msg = str(e)
        if 'No SIMBAD matches' in msg:
            namelist = msg.split('\n')[1:]
            newmsg = 'These names not found in the STELa name table or SIMBAD:\n'
            newmsg += '\n'.join(namelist)
            raise ValueError(newmsg)
        else:
            raise

    stela_names[~found] = names_from_simbad_search

    return stela_names.astype(str)


def groom_hst_names_for_simbad(hst_names):
    simbad_names = []
    for hn in hst_names:
        if hn in hst2simbad_map['odd']:
            sn = hst2simbad_map.loc[hn]['simbad']
        elif hn.upper() in hst2simbad_map['odd']:
            sn = hst2simbad_map.loc[hn.upper()]['simbad']
        elif hn.startswith('BD') or hn.startswith('CD'):
            sn = re.sub(r'([BC])D', r'\1D ', hn)
            sn = re.sub(r'(\d)[-D](\d)', r'\1 \2', sn)
        elif hn.startswith('K2') or hn.startswith('HATS') or hn.startswith('TOI-') or hn.startswith('MASCARA-'):
            sn = hn
        elif hn.startswith('L-'):
            sn = hn.replace('L-', 'L ')
        elif hn.startswith('UCAC'):
            sn = re.sub(r'(UCAC\d)-', r'\1 ', hn)
        elif hn.startswith('V-'):
            sn = hn.replace('V-', 'V* ')
            sn = sn.replace('-', ' ')
        else:
            sn = re.sub(r'([A-Z])-([A-Z\d])', r'\1 \2', hn)
            sn = sn.replace('-UPDATED', '')
            sn = sn.replace('NEWCOORDS', '')
            sn = sn.lower()
        simbad_names.append(sn)
    return simbad_names


def target_names_hst2stela(hst_names):
    simbad_names = groom_hst_names_for_simbad(hst_names)
    return resolve_stela_name_w_simbad(simbad_names)


def target_names_stela2hst(stela_names):
    names = np.char.upper(stela_names)
    names = [re.sub(r'([A-Z]) ([A-Z])', r'\1-\2', name) for name in names]
    names = [re.sub(r'(\d) (\d)', r'\1-\2', name) for name in names]
    names = np.char.replace(names, ' ', '')
    return names


def target_names_stela2tic(stela_names):
    return stela_name_tbl.loc['hostname', stela_names]['tic_id']


def target_names_tic2stela(tic_ids):
    return stela_name_tbl.loc['tic_id', tic_ids]['hostname']


def target_names_stela2file(stela_names):
    names = target_names_stela2hst(stela_names)
    return np.char.lower(names)


def planet_suffixes(catalog):
    tois = catalog['toi']
    has_letter = catalog['pl_letter'].filled('') != ''
    has_toi = catalog['toi'].filled('') != ''
    toi_number = [toi.split('.')[1] for toi in tois[has_toi]]
    special = ~has_letter & ~has_toi
    specials = catalog[special]
    specials.add_index('tic_id')
    special_letters = np.empty(len(specials), dtype=object)
    for tic_id in np.unique(specials['tic_id'].tolist()):
        hostmask = specials['tic_id'] == tic_id
        n = sum(hostmask)
        for i, letter in zip(range(n), cycle('xyz')):
            special_letters[i] = letter
    suffixes = np.empty(len(catalog), dtype=object)
    suffixes[special] = special_letters
    suffixes[has_toi] = toi_number
    suffixes[has_letter] = catalog['pl_letter'][has_letter]
    return suffixes



def hst_filename2stela(path, target='from_header'):
    h = fits.open(path)
    h0, h1 = h[0].header, h[1].header
    if h0['telescop'] != 'HST':
        raise NotImplementedError(f'{path.name} is not an HST file.')
    if target in ['from_header', 'infer']: # keep 'infer' for backwards compatability
        target = h0['targname'].lower()
    instrument = h0['instrume'].lower()
    config = ['hst', instrument]
    if 'opt_elem' in h0:
        config.append(h0['opt_elem'].lower())
    config = '-'.join(config)
    pid = h0['proposid']
    try:
        if instrument == 'cos':
            date = h1['date-obs']
            time = h1['time-obs']
        if instrument == 'stis':
            date = h0['tdateobs']
            time = h0['ttimeobs']
        time = time.replace(':', '')
        time = time[:6]
        datetime = f'{date}T{time}'
    except:
        datetime = 'no-time'
    suffix, = re.findall(r'\w{9}_\w+\.[a-z]+$', path.name)
    name = (f"{target}.{config}.{datetime}.pgm{pid}.{suffix}")
    return name


def parse_filename(path):
    path = Path(path)
    name = path.name
    pieces = name.split('.')[:-1]
    id, ftype = re.search(r'(\w{9})_(\w+_?[ab]?)', pieces[4]).groups()
    parse_dict = dict(target=pieces[0],
                      config=pieces[1],
                      datetime=pieces[2],
                      program=pieces[3],
                      locator=pieces[4])
    if 'hst-' in pieces[1]:
        parse_dict['id'] = id
        parse_dict['type'] = ftype
    if len(pieces) > 5:
        parse_dict['derivative'] = pieces[5]
    return parse_dict


def get_target_name(path):
    return fits.getval(path, 'targname')


def find_visit_primary_target(path, target_dir):
    id = fits.getval(path, 'asn_id').lower()
    files = []
    for folder in (path.parent, target_dir):
        files += list(folder.rglob(f'*{id}*_tag*'))
        files += list(folder.rglob(f'*{id}*_rawtag.fits'))
        files += list(folder.rglob(f'*{id}*_rawtag_a.fits'))
    if len(files) == 0:
        raise ValueError('There must be a [raw]tag file from the same visit as the file below in the same directory '
                         'or the target directory in order to infer the primary visit target'
                         f'\n{path.name}')
    file, = files
    return get_target_name(file)


def modify_file_label(file, new_label):
    """Used to replace, e.g., _x1d with _tag"""
    file = Path(file)
    newname = re.sub(r'_\w{3}\.', f'_{new_label}.', file.name)
    return Path(file.parent) / newname


def find_data_files(
        extension='*',
        targets='any',
        instruments='any',
        after='0000-00-00',
        before='9999-99-99',
        ids='any',
        directory='.'
):
    """
    targets needs to be 'any' or a list of targets like ['k2-9', 'toi-1204']
    instruments can be 'any'; a single string capturing just observatory, isntrument, or spectrograph
    like 'hst', 'hst-stis', or 'hst-stis-g140m'; or a list of full instrument strings like
    ['hst-stis-g140m', 'hst_stis-e140m']
    """
    directory = Path(directory)
    files = list(directory.rglob(f'*_{extension}.fits'))

    if targets != 'any':
        if utils.is_list_like(targets):
            file_targets = [f.name.split('.')[0] for f in files]
            files = [f for f,target in zip(files, file_targets) if target in targets]
        else:
            raise ValueError('Targets must be "any" or a list of target names.')

    if instruments != 'any':
        if isinstance(instruments, str):
            instrument = instruments
            files = [f for f in files if instrument in f.name]
        elif utils.is_list_like(instruments):
            file_insts = [parse_filename(f)['config'] for f in files]
            inst_match_str = rf'{'|'.join(instruments)}'
            files = [f for f,inst in zip(files, file_insts) if re.findall(inst_match_str, inst)]
        else:
            raise ValueError('Targets must be "any", a string, or a list of target names.')

    if ids != 'any':
        if isinstance(ids, str):
            id = ids
            files = [f for f in files if id in f.name]
        elif utils.is_list_like(ids):
            file_ids = [parse_filename(f)['id'] for f in files]
            id_match_str = rf'{'|'.join(ids)}'
            files = [f for f, id in zip(files, file_ids) if re.findall(id_match_str, id)]
        else:
            raise ValueError('Targets must be "any", a string, or a list of target names.')

    file_dates = [f.name.split('.')[2] for f in files]
    files = [f for f,date in zip(files, file_dates) if (date >= after) and (date <= before)]

    return files


def find_stela_files_from_hst_filenames(hst_filenames, directory='.'):
    directory = Path(directory)
    if len(hst_filenames[0]) == 1:
        hst_filenames = hst_filenames,
    files = [list(directory.rglob(f'*{f}')) for f in hst_filenames]
    files = sum(files, [])
    assert len(files) == len(hst_filenames)
    return files


def pathname_max(folder, glob_str):
    folder = Path(folder)
    paths = list(folder.glob(glob_str))
    return max(paths)


def rename_and_organize_hst_files(
        source_dir,
        target_dir='source_dir',
        resolve_stela_name=False,
        overwrite=False,
        validate_names=True,
        target_name='from files',
        into_target_folders=True,
        confirm=True,
):
    # nicknames for directories
    src = Path(source_dir)
    if target_dir == 'source_dir':
        target_dir = source_dir
    tgt = Path(target_dir)

    # find all fits files
    fitspaths = tuple(src.rglob('*.fits'))
    fitspaths = sorted(fitspaths)
    if len(fitspaths) == 0:
        print('No fits files found.')
        return

    # get the STELa-HST names for all targets
    # need to do this first because resolving names requires a SIMBAD query that I should only do once
    # a loop will cause SIMBAD to lock me out
    if target_name == 'from files':
        obs_names = []
        for path in fitspaths:
            hdr = fits.getheader(path)
            targname = hdr['targname']
            if targname in ['WAVEHITM', 'WAVELINE']:
                if hdr['instrume'] == 'STIS':
                    targname = find_visit_primary_target(path, tgt)

            obs_names.append(targname)
        obs_names = np.asarray(obs_names)
        if resolve_stela_name:
            simbad_names = groom_hst_names_for_simbad(obs_names)
            unq_names, i_mapback = np.unique(simbad_names, return_inverse=True)
            stela_names_unq = resolve_stela_name_w_simbad(unq_names)
            stela_hst_names_unq = target_names_stela2file(stela_names_unq)
            targnames = stela_hst_names_unq[i_mapback]
        else:
            targnames = np.char.lower(obs_names)
    else:
        targnames = [target_name]*len(fitspaths)

    # check that all names are what we want, throw error if not
    if validate_names:
        all_stela_file_names = target_names_stela2file(stela_name_tbl['hostname'])
        valid = np.isin(targnames, all_stela_file_names)
        if np.any(~valid):
            if target_name == 'from files':
                temp_tbl = table.Table((np.array(obs_names)[~valid], targnames[~valid]),
                                       names='name_in_file name_to_be_used'.split())
                msg = "These names are not in the STELa database. Something is wrong and you will have to dig, sorry.\n"
                msg += '\n\t'.join(temp_tbl.pformat(-1,-1))
                raise ValueError(msg)
            else:
                raise ValueError(f'{target_name} is not in the STELa database.')

    # generate new file names
    newnames = []
    for path, targname in zip(fitspaths, targnames):
        newname = hst_filename2stela(path, targname)
        newnames.append(newname)

    # for some files, replace path with the equivalent for the associated tag or raw file
    replace_types = ['spt']
    for i, name in enumerate(newnames):
        id, suffix = re.search(r'(\w{9})_([_\w]+)\.fits', name).groups()
        if suffix in replace_types:
            pattern = r'_(tag|raw|rawtag_a|rawacq)\.fits'
            alternate_names = (list(filter(lambda name: len(re.findall(id + pattern, name)) > 0, newnames))
                               + list(tgt.glob(f'*{id}_*tag*.fits'))
                               + list(tgt.glob(f'*{id}_raw*.fits')))
            if len(alternate_names) == 0:
                raise ValueError(f'No matching file found for {name}. You might need to downlaod it.')
            alternate_name = alternate_names[0]
            alternate_name = Path(alternate_name).name
            newnames[i] = re.sub(pattern, f'_{suffix}.fits', alternate_name)

    newpaths = []
    for newname, targname in zip(newnames, targnames):
        newpath = tgt / targname / newname / 'hst' if into_target_folders else tgt / newname
        if newpath != path:
            newpaths.append(newpath)
        else:
            newpaths.append(None)

    # show moves and renames are about to happen and ask user to confirm
    if confirm:
        print(f"You are about to execute this moving and renaming of these files from the directory"
              f"\n\t{src}"
              f"\nto the directory"
              f"\n\t{tgt}\n")
        for old, new in zip(fitspaths, newpaths):
            oldstr = str(old).replace(str(src), '')
            newstr = 'No change.' if new is None else str(new).replace(str(tgt), '')
            print(f"\t{oldstr} --> {newstr}")
        proceed = input('Note that files will be moved and renamed, not copied. Proceed (y/n)?\n')
    else:
        proceed = 'y'

    if proceed == 'y':
        for old, new in zip(fitspaths, newpaths):
            if new is None:
                continue
            if new.exists():
                if overwrite:
                    os.remove(new)
                else:
                    raise IOError(f'{new.name} exists, set overwrite=True if desired.')
            if not new.parent.exists():
                os.mkdir(new.parent)
            os.rename(old, new)


def delete_files_by_hst_id(ids, directory='.'):
    files = find_data_files('*', ids=ids, directory=directory)
    msg = f'Proceed with the permanent deletion of these files? (y/n)'
    for file in files:
        msg += f'\t\n{file.name}'
    answer = input(msg)
    if answer == 'y':
        for file in files:
            os.remove(file)


def filter_observations(
        obs_table,
        config_substrings=None,
        usable=None,
        usability_fill_value=True,
        exclude_flags=None
):
    """
    Filter an astropy table based on config substring matches, usability, and exclusion flags.

    Parameters
    ----------
    obs_table : astropy.table.Table
        The input table.
    config_substrings : list of str, optional
        Substrings to match in the 'config' column.
    usable : bool, optional
        Whether to keep rows marked usable (True) or not (False).
    usability_fill_value : bool, optional
        Value to fill in for masked usability entries.
    exclude_flags : list of str, optional
        Any row with one of these flags in the 'flags' column will be excluded.

    Returns
    -------
    filtered_table : astropy.table.Table
        The filtered table.
    """
    tbl = obs_table.copy()

    # Step 1: Fill masked usability values
    if isinstance(tbl['usable'], obs_table.MaskedColumn):
        tbl['usable'] = tbl['usable'].filled(usability_fill_value)

    # Step 2: Filter by config substrings
    if config_substrings:
        config_mask = np.zeros(len(tbl), dtype=bool)
        for substr in config_substrings:
            config_mask |= [substr in cfg for cfg in tbl['science config']]
        tbl = tbl[config_mask]

    # Step 3: Filter by usability
    if usable is not None:
        tbl = tbl[tbl['usable'] == usable]

    # Step 4: Filter out rows with any matching flag
    if exclude_flags:
        def has_excluded_flag(flag_list):
            return any(flag in flag_list for flag in exclude_flags)

        flag_mask = [not has_excluded_flag(flags) for flags in tbl['flags']]
        tbl = tbl[flag_mask]

    return tbl


def delete_files_for_unusable_observations(
        obs_table,
        usability_fill_value=True,
        dry_run=True,
        verbose=True,
        directory='.'
):
    """
    Delete science and supporting files for unusable observations.

    Science files are always deleted.
    Supporting files are deleted only if they are not used in any usable observation.

    Parameters
    ----------
    obs_table : astropy.table.Table
        Table containing observation metadata with 'usable', 'key science files', and 'supporting files' columns.
    usability_fill_value : bool, optional
        Value to fill in for masked entries in the 'usable' column.
    dry_run : bool, optional
        If True, just print what would be deleted. If False, actually delete the files.
    verbose : bool, optional
        If True, print what is being deleted or kept.
    """
    from astropy.table import MaskedColumn

    directory = Path(directory)

    tbl = obs_table.copy()

    # Fill in masked usability values if needed
    if isinstance(tbl['usable'], MaskedColumn):
        tbl['usable'] = tbl['usable'].filled(usability_fill_value)

    # Collect all supporting files used by usable observations
    supporting_usage = defaultdict(int)
    for row in tbl[tbl['usable']]:
        if row['supporting files']:
            for f in row['supporting files'].values():
                supporting_usage[f] += 1

    # Go through unusable observations
    to_be_deleted = []
    for row in tbl[~tbl['usable']]:
        # Delete science files
        id = row['archive id']
        fpaths = find_data_files('*', ids=id, directory=directory)
        to_be_deleted.extend(fpaths)
        for fpath in fpaths:
            if verbose or dry_run:
                print(f"{'[DRY-RUN] ' if dry_run else ''}Deleting science file: {fpath.name}")
            if not dry_run and os.path.exists(fpath):
                os.remove(fpath)

        # Delete supporting files if no other usable obs uses them
        if row['supporting files']:
            for label, f in row['supporting files'].items():
                if supporting_usage[f] == 0:
                    fpaths = list(directory.glob(f'*{f}'))
                    to_be_deleted.extend(fpaths)
                    if fpaths:
                        fpath, = fpaths
                        if verbose or dry_run:
                            print(f"{'[DRY-RUN] ' if dry_run else ''}Deleting unused supporting file: {fpath.name}")
                        if not dry_run and os.path.exists(fpath):
                            os.remove(fpath)
                elif verbose:
                    print(f"Keeping shared supporting file: {f}")

    return to_be_deleted

def find_coadd_or_x1ds(target, **file_srch_kws):
    files = find_data_files('coadd', targets=[target], **file_srch_kws)
    if not files:
        files = find_data_files('x1d', **file_srch_kws)
    return files


def clear_usability_values(obs_tbl, id_substr=None, reason_substr=None, other_columns_to_clear=None):
    """Does what the name suggests. Use reason_substr if you only want to clear, e.g., rows where the reason includes
    "acquisition"."""
    cleared_tbl = obs_tbl.copy()
    if other_columns_to_clear is None:
        other_columns_to_clear = []
    if id_substr is None:
        id_substr = ''
    if reason_substr is None:
        reason_substr = ''
    def get_substr_mask(colname, sub):
        str_col = obs_tbl[colname].filled('').astype(str)
        return np.char.count(str_col, sub) > 0
    mask = get_substr_mask('archive id', id_substr) & get_substr_mask('reason unusable', reason_substr)
    colanmes_to_clear = ['usable', 'reason unusable'] + other_columns_to_clear
    for name in colanmes_to_clear:
        cleared_tbl[name].mask |= mask
    return cleared_tbl