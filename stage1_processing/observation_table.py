import re
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits

import paths
import catalog_utilities as catutils
import database_utilities as dbutils


class ObsTable(table.Table):
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
        'usability status',
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

    @classmethod
    def initialize_blank(cls, science_files=()):
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
            table.MaskedColumn(length=n, name='usability status', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='reason unusable', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='flags', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='notes', dtype='object', mask=True)
        ]

        tbl = ObsTable(columns)

        for i, asc_files in enumerate(key_science_files):
            pi = fits.getval(data_dir / asc_files[0], 'PR_INV_L')
            pieces = dbutils.parse_filename(asc_files[0])
            tbl['archive id'][i] = pieces['id']
            tbl['science config'][i] = pieces['config']
            tbl['start'][i] = re.sub(r'([\d-]+T\d{2})(\d{2})(\d{2})', r'\1:\2:\3', pieces['datetime'])
            tbl['program'][i] = pieces['program'].replace('pgm', '')
            tbl['pi'][i] = pi
            tbl['key science files'][i] = ['.'.join(name.split('.')[-2:]) for name in asc_files]

        return tbl

    @classmethod
    def get_path(cls, target):
        data_dir = paths.target_hst_data(target)
        path_obs_tbl = data_dir / f'{target}.observation-table.ecsv'
        return path_obs_tbl

    @classmethod
    def load_from_targname(cls, target):
        path_obs_tbl = ObsTable.get_path(target)
        return ObsTable.read(path_obs_tbl)

    @classmethod
    def read(cls, path):
        obs_tbl = catutils.load_and_mask_ecsv(path)

        newcols = ObsTable.standard_columns - set(obs_tbl.colnames)
        for col in newcols:
            obs_tbl[col] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')

        for col in ObsTable.object_columns:
            obs_tbl[col] = obs_tbl[col].astype('object')

        return ObsTable(obs_tbl)

    def add_flag(self, row_idx, flag):
        catutils.append_to_column_of_lists(self, 'flags', row_idx, flag, raise_nonlist_error=False)
        self.clean_duplicate_flags()

    def clean_duplicate_flags(self):
        flagcol = self['flags']
        for i, val in enumerate(flagcol):
            if not np.ma.is_masked(val):
                flagcol[i] = np.unique(val).tolist()

    def filter_files_by_usability_status(self, paths, allow=('has issues', 'all clear', 'unchecked', 'masked')):
        file_ids = [dbutils.parse_filename(p)['id'] for p in paths]
        mask = np.zeros(len(self), bool)
        usability = self['usability status']
        if hasattr(usability, 'filled'):
            usability = usability.filled('masked')
        for status in allow:
            mask |= usability == status
        good_obs_ids = self['archive id'][mask].copy()
        mask = np.isin(file_ids, good_obs_ids)
        filtered_paths = np.array(paths)[mask]
        return list(filtered_paths)


# for backwards compatability
initialize = ObsTable.initialize_blank
get_path = ObsTable.get_path
load_obs_tbl = ObsTable.load_from_targname