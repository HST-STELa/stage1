import re
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits

import paths
import catalog_utilities as catutils
import database_utilities as dbutils
import utilities as utils


class ObsTable(table.Table):
    standard_columns = (
        'observatory',
        'science config',
        'start',
        'program',
        'pi',
        'archive id',
        'usable',
        'usability status',
        'reason unusable',
        'flags',
        'notes',
        'key science files',
        'supporting files',
    )
    non_object_columns = (
        'start',
        'program',
        'usable'
    )
    object_columns = set(standard_columns) - set(non_object_columns)
    object_columns = list(object_columns)

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
        obs_tbl = ObsTable(obs_tbl)

        missing_cols = set(ObsTable.standard_columns) - set(obs_tbl.colnames)
        for col in missing_cols:
            obs_tbl[col] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype='object')

        for col in ObsTable.object_columns:
            obs_tbl[col] = obs_tbl[col].astype('object')

        # organize columns in default order
        new_cols = set(obs_tbl.colnames) - set(ObsTable.standard_columns)
        new_cols_original_order = [c for c in obs_tbl.colnames if c in new_cols]
        col_order = list(ObsTable.standard_columns) + new_cols_original_order
        obs_tbl = obs_tbl[col_order]

        return obs_tbl

    def add_flags(self, row_idx, comma_sep_flags):
        self.add_comma_sep_str_to_list_col('flags', row_idx, comma_sep_flags)

    def add_notes(self, row_idx, commas_sep_notes):
        self.add_comma_sep_str_to_list_col('notes', row_idx, commas_sep_notes)

    def add_comma_sep_str_to_list_col(self, colname, row_idx, comma_sep_str):
        items = re.split(', *', comma_sep_str)
        for item in items:
            if not item.isspace():
                catutils.append_to_column_of_lists(self, colname, row_idx, item, raise_nonlist_error=False)

    def add_flag(self, row_idx, flag):
        catutils.append_to_column_of_lists(self, 'flags', row_idx, flag, raise_nonlist_error=False)
        self.clean_duplicates_col_of_lists('flags')

    def clean_nulls_col_of_lists(self, colname="flags"):
        old = self[colname]

        new_data = []
        new_mask = []

        # existing mask if present, otherwise all False
        old_mask = getattr(old, "mask", np.zeros(len(old), dtype=bool))

        for i, val in enumerate(old):
            if old_mask[i]:
                new_data.append(None)
                new_mask.append(True)
                continue

            # treat scalar/string as one-item list
            if isinstance(val, str) or not hasattr(val, "__iter__"):
                items = [val]
            else:
                items = list(val)

            cleaned = [item for item in items if not utils.is_null_item(item)]

            if len(cleaned) == 0:
                new_data.append(None)
                new_mask.append(True)
            else:
                new_data.append(cleaned)
                new_mask.append(False)

        new_col = table.MaskedColumn(
            data=new_data,
            mask=new_mask,
            name=colname,
            dtype=object,
        )

        self.replace_column(colname, new_col)

    def clean_duplicates_col_of_lists(self, colname='flags'):
        self.clean_nulls_col_of_lists(colname)
        col = self[colname]
        for i, val in enumerate(col):
            if not np.ma.is_masked(val):
                col[i] = np.unique(val).tolist()

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

    def write(self, path='default', overwrite=False):
        if path == 'default':
            target = self.meta['target']
            path = self.get_path(target)
        super().write(path, serialize_method='data_mask', overwrite=overwrite)


# for backwards compatability
initialize = ObsTable.initialize_blank
get_path = ObsTable.get_path
load_obs_tbl = ObsTable.load_from_targname