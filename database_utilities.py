from pathlib import Path
import os
import re
import warnings

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
        elif hn.startswith('K2') or hn.startswith('HATS') or hn.startswith('TOI-'):
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


def target_names_stela2file(stela_names):
    names = target_names_stela2hst(stela_names)
    return np.char.lower(names)


def filename2stela(path, target='from_header'):
    h = fits.open(path)
    h0, h1 = h[0].header, h[1].header
    if h0['telescop'] != 'HST':
        raise NotImplementedError(f'{path.name} is not an HST file.')
    if target in ['from_header', 'infer']: # keep 'infer' for backwards compatability
        target = h0['targname'].lower()
    instrument = h0['instrume'].lower()
    grating = h0['opt_elem'].lower()
    pid = h0['proposid']
    if instrument == 'cos':
        date = h1['date-obs']
        time = h1['time-obs'].replace(':', '')
    if instrument == 'stis':
        date = h0['tdateobs']
        time = h0['ttimeobs'].replace(':', '')
    time = time[:6]
    suffix, = re.findall(r'\w{9}_\w+\.[a-z]+', h0['filename'])
    name = (f"{target}.hst-{instrument}-{grating}.{date}T{time}.pgm{pid}.{suffix}")
    return name


def parse_filename(path):
    path = Path(path)
    name = path.name
    pieces = name.split('.')[:-1]
    stsci_name = pieces[-1].split('_')
    parse_dict = dict(target=pieces[0],
                      config=pieces[1],
                      datetime=pieces[2],
                      id=stsci_name[0],
                      type=stsci_name[-1])
    return parse_dict


def get_target_name(path):
    return fits.getval(path, 'targname')


def find_visit_primary_target(path):
    id = fits.getval(path, 'asn_id').lower()
    files = (list(path.parent.rglob(f'*{id}*_tag*'))
             + list(path.parent.rglob(f'*{id}*_rawtag.fits'))
             + list(path.parent.rglob(f'*{id}*_rawtag_a.fits')))
    if len(files) == 0:
        raise ValueError('There must be a [raw]tag file from the same visit as the file below in the same directory '
                         'in order to infer the primary visit target'
                         f'\n{path.name}')
    file, = files
    return get_target_name(file)


def modify_file_label(file, new_label):
    """Used to replace, e.g., _x1d with _tag"""
    file = Path(file)
    newname = re.sub(r'_\w{3}\.', f'_{new_label}.', file.name)
    return Path(file.parent) / newname


def find_data_files(extension='*', targets='any', instruments='any', after='0000-00-00', before='9999-99-99', directory='.'):
    """
    targets needs to be 'any' or a list of targets like ['k2-9', 'toi-1204']
    instruments can be 'any'; a single string capturing just observatory, isntrument, or spectrograph
    like 'hst', 'hst-stis', or 'hst-stis-g140m'; or a list of full instrument strings like
    ['hst-stis-g140m', 'hst_stis-e140m']
    """
    directory = Path(directory)
    files = list(directory.glob(f'*_{extension}.fits'))

    if targets != 'any':
        if utils.is_list_like(targets):
            file_targets = [f.name.split('.')[0] for f in files]
            files = [f for f,target in zip(files, file_targets) if target in targets]
        else:
            raise ValueError('Targets must be "any" or a list of target names.')

    if instruments != 'any':
        if isinstance(instruments, str):
            files = [f for f in files if instruments in f.name]
        elif utils.is_list_like(instruments):
            file_insts = [f.name.split('.')[1] for f in files]
            files = [f for f,inst in zip(files, file_insts) if inst in instruments]
        else:
            raise ValueError('Targets must be "any", a string, or a list of target names.')

    file_dates = [f.name.split('.')[2] for f in files]
    files = [f for f,date in zip(files, file_dates) if (date >= after) and (date <= before)]

    return files


def pathname_max(folder, glob_str):
    folder = Path(folder)
    paths = list(folder.glob(glob_str))
    return max(paths)


def rename_and_organize_hst_files(source_dir, target_dir='source_dir', resolve_stela_name=False, overwrite=False):
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

    # get the STELa-HST names for all targets
    # need to do this first because resolving names requires a SIMBAD query that I should only do once
    # a loop will cause SIMBAD to lock me out
    obs_names = []
    for path in fitspaths:
        hdr = fits.getheader(path)
        targname = hdr['targname']
        mode = hdr['obsmode'] # both STIS and COS have this kwd
        mode += hdr['exptype'] if 'exptype' in hdr else '' # only COS has this kwd, and it is where ACQ is specified
        exptype = fits.getval(path, 'exptype')
        # these files might are part of the observations but didn't target the science target
        if targname in ['WAVEHITM', 'WAVELINE'] : # FIXME remove
        # if (targname in ['WAVEHITM', 'WAVELINE']) or ('ACQ' in exptype):
            targname = find_visit_primary_target(path)
        obs_names.append(targname)
    obs_names = np.asarray(obs_names)
    if resolve_stela_name:
        simbad_names = groom_hst_names_for_simbad(obs_names)
        unq_names, i_mapback = np.unique(simbad_names, return_inverse=True)
        stela_names_unq = resolve_stela_name_w_simbad(unq_names)
        stela_hst_names_unq = target_names_stela2file(stela_names_unq)
        stela_hst_names = stela_hst_names_unq[i_mapback]
    else:
        stela_hst_names = np.char.lower(obs_names)

    # check that all names are what we want, throw error if not
    all_stela_file_names = target_names_stela2file(stela_name_tbl['hostname'])
    valid = np.in1d(stela_hst_names, all_stela_file_names)
    if np.any(~valid):
        temp_tbl = table.Table((obs_names[~valid], stela_hst_names[~valid]),
                               names='name_in_file stela_name_guess'.split())
        msg = "These names are in the STELa database. Something is wrong and you will have to dig, sorry.\n"
        msg += '\n\t'.join(temp_tbl.pformat(-1,-1))
        raise ValueError(msg)

    # generate new file paths
    newpaths = []
    for path, targname in zip(fitspaths, stela_hst_names):
        newfilename = filename2stela(path, targname)
        newpath = tgt / targname / newfilename
        if newpath != path:
            newpaths.append(newpath)
        else:
            newpaths.append(None)

    # show moves and renames are about to happen and ask user to confirm
    print(f"You are about to execute this moving and renaming of these files from the directory"
          f"\n\t{src}"
          f"\nto the directory"
          f"\n\t{tgt}\n")
    for old, new in zip(fitspaths, newpaths):
        oldstr = str(old).replace(str(src), '')
        newstr = 'No change.' if new is None else str(new).replace(str(tgt), '')
        print(f"\t{oldstr} --> {newstr}")
    proceed = input('Note that files will be moved and renamed, not copied. Proceed (y/n)?\n')

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