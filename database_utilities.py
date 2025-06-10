from pathlib import Path
import os
import re

from astropy.io import fits

import utilities as utils

# copied in from the observation progress sheet
program_target_names = ("TRAPPIST-1, HD 136352, AU Mic, HD 63433, GJ 143, HD 95338, HD 60779, HD 260655, LHS 1140, "
                        "TOI-1468, TOI-2443, TOI-1759, TOI-2134, HD 189733, TOI-2194, L 98-59, TOI-5789, TOI-6965, "
                        "HAT-P-20, TOI-776, HD 110067, TOI-836, HD 207496, TOI-1231, TOI-406, TOI-2076, TOI-1774, "
                        "HD 63935, TOI-6973, TOI-1710, HD 73344, HD 207897, GJ 436, HIP 9618, HD 97658, TOI-1224, "
                        "HD 39091, HD 73583, TOI-712, HR 858, LTT 3780, TOI-444, DS Tuc A, TOI-1203, TOI-1898, "
                        "HAT-P-11, HD 18599, HD 135694, HD 21520, WASP-59, HD 5278, HD 22946, TOI-561, TOI-815, "
                        "K2-72, TOI-5554, TOI-1266, TOI-2285, TOI-700, TOI-4576, TOI-4632, TOI-260, TOI-1467, "
                        "TOI-1434, Kepler-37, GJ 1214, HD 15906, HD 15337, TOI-431, WASP-84, TOI-270, TOI-2094, "
                        "LP 714-47, TOI-178, TOI-2018, HD 235088, TOI-2095, HD 209458, K2-3, HD 17156, TOI-1730, "
                        "TOI-198, HD 183579, TOI-2015, TOI-1728, TOI-233, TOI-1452, TOI-6078, TOI-4438, TOI-4336 A, "
                        "TOI-904, TOI-687, TOI-1696, TOI-1751, TOI-480, TOI-1691, TOI-2459, GJ 3090, TOI-2276, "
                        "TOI-1718, HD 332231, TOI-4189, TOI-1742, TOI-1268, TOI-1451, KELT-2 A, TOI-286, TOI-1801, "
                        "TOI-1235, HD 3167, TOI-6850, TOI-257, TOI-620, TOI-122, K2-9, TOI-6992, TOI-2136, HD 23472, "
                        "TOI-870, TOI-133, Wolf 503, TOI-2079, TOI-5388, Kepler-10, HIP 94235, K2-174, TOI-1695, "
                        "TOI-2128, TOI-771, TOI-5788, TOI-727, HD 42813, TOI-6871, TOI-5169, HIP 113103, TOI-2287, "
                        "HD 118203, TOI-421, TOI-4185, Kepler-138, HD 191939, TOI-1643, TOI-1752, TOI-4643, TOI-2158, "
                        "WASP-69, TOI-214, TOI-4556, K2-18, V1298 Tau, GJ 3470, K2-136, GJ 9827, WASP-77 A, K2-233, "
                        "HIP 116454, HD 86226, WASP-107, K2-25, HD 149026, HIP 67522, LTT 1445 A, Kepler-444, "
                        "HD 219134, TOI-4307, LP 791-18, TOI-5521, TOI-244, TOI-6054, WASP-140, TOI-4451, TOI-4529, "
                        "TOI-5076, WASP-8, TOI-4364, TOI-715, TOI-1180, TOI-7052, TOI-3485, TOI-5531, TOI-553")
program_target_names = program_target_names.split(', ')
program_target_names = [x.replace(' ', '').lower() for x in program_target_names]
name_resolver = {
    'hd-85426': 'toi-1774',
    'hd-97507': 'toi-1203'
}

def rename_files(directory, overwrite=False):
    directory = Path(directory)
    paths = list(directory.glob('*.fits'))
    if len(paths) == 0:
        print('No fits files found.')

    newpaths = []
    unresolved_names = []
    for path in paths:
        target = get_target(path)
        if target in ['wavehitm', 'waveline']:
            target = find_visit_primary_target(path, directory)
        if target not in program_target_names:
            try:
                target = name_resolver[target].lower()
            except KeyError:
                unresolved_names.append(target)
                newpaths.append(None)
                continue
        newname = new_name(path, target)

        newpath = directory / newname
        if newpath != path:
            newpaths.append(newpath)
        else:
            newpaths.append(None)

    if unresolved_names:
        print("These target names did not match to a program target. You will need to add an appropriate pair to"
              "the database_utilities.name_resolver dictionary for their files to be renamed.")
        for name in unresolved_names:
            print(f'\t{name}')
        print('')

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