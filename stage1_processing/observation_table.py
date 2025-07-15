import re

from astropy import table
from astropy.io import fits

import paths
import catalog_utilities as catutils
import database_utilities as dbutils

standard_columns = {
    'observatory',
    'science config',
    'start',
    'program',
    'pi',
    'archive id',
    'key science files',
    'supporting files',
    'usable',
    'reason unusable',
    'flags',
    'notes'
}
non_object_columns = {
    'start',
    'program',
    'usable'
}
object_columns = standard_columns - non_object_columns


def get_path(target):
    data_dir = paths.target_hst_data(target)
    path_obs_tbl = data_dir / f'{target}.observation-table.ecsv'
    return path_obs_tbl


def load_obs_tbl(target):
    path_obs_tbl = get_path(target)
    obs_tbl = catutils.load_and_mask_ecsv(path_obs_tbl)

    newcols = standard_columns - set(obs_tbl.colnames)
    for col in newcols:
        obs_tbl[col] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')

    for col in object_columns:
        obs_tbl[col] = obs_tbl[col].astype('object')

    return obs_tbl


def initialize(science_files=()):
    data_dir = science_files[0].parent if science_files else None
    science_files = science_files
    key_science_files = []
    files_science_copy = science_files[:]
    while files_science_copy:
        file = files_science_copy.pop(0)
        pieces = dbutils.parse_filename(file)
        associated_files = [file.name]
        ftp = pieces['type']
        pairs = (('_a', '_b'), ('_b', '_a'))
        for s1, s2 in pairs:
            if ftp.endswith(s1):
                file2 = file.parent / file.name.replace(f'{s1}.fits', f'{s2}.fits')
                if file2 in files_science_copy:
                    files_science_copy.remove(file2)
                    associated_files.append(file.name)
        key_science_files.append(associated_files)

    n = len(key_science_files)
    columns = [
        table.MaskedColumn(data=['hst'] * n, name='observatory', dtype='object'),
        table.MaskedColumn(length=n, name='science config', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='start', dtype='S20', mask=True),
        table.MaskedColumn(length=n, name='program', dtype='int', mask=True),
        table.MaskedColumn(length=n, name='pi', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='archive id', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='key science files', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='supporting files', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='usable', dtype='bool', mask=True),
        table.MaskedColumn(length=n, name='reason unusable', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='flags', dtype='object', mask=True),
        table.MaskedColumn(length=n, name='notes', dtype='object', mask=True)
    ]
    obs_tbl = table.Table(columns)
    for i, asc_files in enumerate(key_science_files):
        pi = fits.getval(data_dir / asc_files[0], 'PR_INV_L')
        pieces = dbutils.parse_filename(asc_files[0])
        obs_tbl['archive id'][i] = pieces['id']
        obs_tbl['science config'][i] = pieces['config']
        obs_tbl['start'][i] = re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime'])
        obs_tbl['program'][i] = pieces['program'].replace('pgm', '')
        obs_tbl['pi'][i] = pi
        obs_tbl['key science files'][i] = ['.'.join(name.split('.')[-2:]) for name in asc_files]

    return obs_tbl
