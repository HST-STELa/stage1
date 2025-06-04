from pathlib import Path
import os
import re

from astropy.io import fits

import utilities as utils

def rename_files(directory, overwrite=False):
    directory = Path(directory)
    paths = list(directory.glob('*.fits'))
    if len(paths) == 0:
        print('No fits files found.')

    newpaths = []
    for path in paths:
        target = get_target(path)
        if target in ['wavehitm', 'waveline']:
            target = find_visit_primary_target(path, directory)
        newname = new_name(path, target)

        newpath = directory / newname
        if newpath != path:
            newpaths.append(newpath)
        else:
            newpaths.append(None)

    print(f"You are about to execute this renaming of files in {directory}:")
    for old, new in zip(paths, newpaths):
        newname = 'Will not be renamed.' if new is None else new.name
        print(f"\t{old.name} --> {newname}")
    proceed = input('Note that files will be renamed, not copied. Proceed (y/n)?\n')

    if proceed == 'y':
        for old, new in zip(paths, newpaths):
            if new is None:
                continue
            if new.exists():
                if overwrite:
                    os.remove(new)
                else:
                    raise IOError(f'{new.name} exists, set overwrite=True if desired.')
            os.rename(old, new)


def new_name(path, target='infer'):
    h = fits.open(path)
    h0, h1 = h[0].header, h[1].header
    if h0['telescop'] != 'HST':
        raise NotImplementedError(f'{path.name} is not an HST file.')

    if target == 'infer':
        target = h0['targname'].lower()
    instrument = h0['instrume'].lower()
    grating = h0['opt_elem'].lower()
    if instrument == 'cos':
        date = h1['date-obs']
        time = h1['time-obs'].replace(':', '')
    if instrument == 'stis':
        date = h0['tdateobs']
        time = h0['ttimeobs'].replace(':', '')
    time = time[:6]
    suffix, = re.findall(r'\w{9}_\w{3}.[a-z]+', h0['filename'])
    name = (f"{target}.hst-{instrument}-{grating}.{date}T{time}.{suffix}")
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


def get_target(path):
    return fits.getval(path, 'targname').lower()


def find_visit_primary_target(path, directory):
    id = fits.getval(path, 'asn_id').lower()
    files = list(directory.glob(f'*{id}*_tag*'))
    if len(files) == 0:
        raise ValueError('There must be a tag file from the same visit as the file below in the directory '
                         'in order to infer the primary visit target'
                         f'\n{path.name}')
    file, = files
    return get_target(file)


def modify_file_label(file, new_label):
    """Used to replace, e.g., _x1d with _tag"""
    file = Path(file)
    newname = re.sub(r'_\w{3}\.', f'_{new_label}.', file.name)
    return Path(file.parent) / newname


def find_data_files(extension='*', targets='any', instruments='any', after='2020', before='inf', directory='.'):
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
    files = [f for f,date in zip(files, file_dates) if date > after]
    files = [f for f,date in zip(files, file_dates) if date < before]

    return files


def pathname_max(folder, glob_str):
    folder = Path(folder)
    paths = list(folder.glob(glob_str))
    return max(paths)