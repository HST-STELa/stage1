import json
import re
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits

import paths
import catalog_utilities as catutils
import database_utilities as dbutils
import utilities as utils


class ObsRow(table.Row):
    """Row view for `ObsTable` with mask-aware ``get`` and ``usable``."""

    @staticmethod
    def _cell_masked_for_index(col, idx):
        mask = getattr(col, "mask", None)
        if mask is None:
            return False
        if mask is True:
            return True
        try:
            return bool(np.asarray(mask, dtype=bool).reshape(-1)[idx])
        except Exception:
            return False

    @staticmethod
    def _structure_entirely_absent(val):
        if isinstance(val, str):
            return utils.is_null_item(val)
        if ObsTable._is_null_like(val):
            return True
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return True
            if np.ma.isMaskedArray(val):
                return ObsRow._structure_entirely_absent(val.tolist())
            return ObsRow._structure_entirely_absent(val.tolist())
        if isinstance(val, dict):
            if len(val) == 0:
                return True
            return all(ObsRow._structure_entirely_absent(v) for v in val.values())
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return True
            return all(ObsRow._structure_entirely_absent(v) for v in val)
        return False

    def get(self, key, default=None, /):
        if key not in self._table.columns:
            return default
        col = self._table.columns[key]
        if self._cell_masked_for_index(col, self._index):
            return default
        val = table.Row.__getitem__(self, key)
        if self._structure_entirely_absent(val):
            return default
        return val

    def usable(self, fill=True):
        return self.get("usable", fill)


class ObsTable(table.Table):
    Row = ObsRow
    standard_column_specs = (
        # name, dtype, semantic subtype for object columns
        ('observatory', 'O', str),
        ('science config', 'O', str),
        ('start', 'U', None),
        ('program', 'int64', None),
        ('pi', 'O', str),
        ('archive id', 'O', str),
        ('usable', bool, None),
        ('usability status', 'O', str),
        ('reason unusable', 'O', str),
        ('flags', 'O', list),
        ('notes', 'O', list),
        ('key science files', 'O', list),
        ('supporting files', 'O', dict),
    )

    _OBJECT_SUBTYPE_METAKEY = 'object_column_semantic_subtypes'

    standard_columns = [x[0] for x in standard_column_specs]
    column_spec_map = {name: (dtype, subtype) for name, dtype, subtype in standard_column_specs}
    object_column_subtypes = {
            name: subtype
            for name, dtype, subtype in standard_column_specs
            if dtype in ('O', object, 'object')
        }
    object_columns = list(object_column_subtypes.keys())

    @staticmethod
    def _is_masked_scalar(x):
        return x is np.ma.masked

    @staticmethod
    def _is_null_like(x):
        if x is np.ma.masked or x is None:
            return True
        if np.ma.isMaskedArray(x) and x.size == 0:
            return True
        return False

    @classmethod
    def _coerce_object_value_for_write(cls, val, expected_subtype, colname):
        """
        Convert object-column payloads into plain Python JSON-serializable values.
        Return None for null/masked values.
        """
        if cls._is_null_like(val):
            return None

        # Convert NumPy things to plain Python
        if np.ma.isMaskedArray(val):
            # non-empty masked arrays should become ordinary Python containers/scalars
            val = val.tolist()
        elif isinstance(val, np.ndarray):
            val = val.tolist()

        if expected_subtype is str:
            # Repair old bad saves where string columns became single-element lists
            if isinstance(val, list):
                if len(val) == 0:
                    return None
                if len(val) == 1 and isinstance(val[0], str):
                    return val[0]
                raise TypeError(
                    f"Column {colname!r} should contain strings; got list value {val!r}"
                )
            if isinstance(val, str):
                return val
            raise TypeError(
                f"Column {colname!r} should contain strings; got {type(val).__name__}: {val!r}"
            )

        if expected_subtype is list:
            # Repair old bad saves where list columns became bare strings
            if isinstance(val, str):
                return [val]
            if isinstance(val, tuple):
                return list(val)
            if isinstance(val, list):
                return val
            raise TypeError(
                f"Column {colname!r} should contain lists; got {type(val).__name__}: {val!r}"
            )

        if expected_subtype is dict:
            if isinstance(val, dict):
                return val
            raise TypeError(
                f"Column {colname!r} should contain dicts; got {type(val).__name__}: {val!r}"
            )

        # fallback: plain Python conversion if possible
        return val

    @classmethod
    def _repair_object_value_on_read(cls, val, expected_subtype, colname):
        """
        Repair values read back from ECSV according to the semantic subtype.
        """
        if cls._is_null_like(val):
            return None

        if np.ma.isMaskedArray(val):
            if val.size == 0:
                return None
            val = val.tolist()
        elif isinstance(val, np.ndarray):
            val = val.tolist()

        if expected_subtype is str:
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                if len(val) == 0:
                    return None
                if len(val) == 1 and isinstance(val[0], str):
                    return val[0]
            raise TypeError(
                f"While reading, column {colname!r} should be string-like but got {type(val).__name__}: {val!r}"
            )

        if expected_subtype is list:
            if isinstance(val, list):
                return val
            if isinstance(val, tuple):
                return list(val)
            if isinstance(val, str):
                return [val]
            raise TypeError(
                f"While reading, column {colname!r} should be list-like but got {type(val).__name__}: {val!r}"
            )

        if expected_subtype is dict:
            if isinstance(val, dict):
                return val
            raise TypeError(
                f"While reading, column {colname!r} should be dict-like but got {type(val).__name__}: {val!r}"
            )

        return val

    @classmethod
    def _make_masked_object_column(cls, values, name):
        mask = [v is None for v in values]
        data = [None if m else v for v, m in zip(values, mask)]
        return table.MaskedColumn(
            data=np.array(data, dtype=object),
            mask=np.array(mask, dtype=bool),
            name=name,
            dtype=object,
        )

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
            table.MaskedColumn(length=n, name='start', dtype='U20', mask=True),
            table.MaskedColumn(length=n, name='program', dtype='int64', mask=True),
            table.MaskedColumn(length=n, name='pi', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='archive id', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='key science files', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='supporting files', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='usable', dtype='bool', mask=True),
            table.MaskedColumn(length=n, name='usability status', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='reason unusable', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='flags', dtype='object', mask=True),
            table.MaskedColumn(length=n, name='notes', dtype='object', mask=True),
        ]

        tbl = cls(columns)

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
        path_obs_tbl = cls.get_path(target)
        return cls.read(path_obs_tbl)

    @classmethod
    def read(cls, path, *args, **kwargs):
        obs_tbl = catutils.load_and_mask_ecsv(path)
        obs_tbl = cls(obs_tbl)

        missing_cols = set(cls.standard_columns) - set(obs_tbl.colnames)
        for col in missing_cols:
            dtype, subtype = cls.column_spec_map[col]
            obs_tbl[col] = table.MaskedColumn(length=len(obs_tbl), mask=True, dtype=dtype)

        # Determine expected semantic subtype for object columns.
        # Prefer explicit class spec; optionally merge with metadata from file.
        saved_subtypes = obs_tbl.meta.get(cls._OBJECT_SUBTYPE_METAKEY, {})
        expected_subtypes = dict(saved_subtypes)
        expected_subtypes.update(cls.object_column_subtypes)

        for colname, expected_subtype in expected_subtypes.items():
            if colname not in obs_tbl.colnames:
                continue

            old = obs_tbl[colname]
            old_mask = np.array(getattr(old, 'mask', np.zeros(len(old), dtype=bool)), dtype=bool)

            repaired = []
            repaired_mask = []

            for i, val in enumerate(old):
                masked = np.any(old_mask[i])

                if masked or cls._is_null_like(val):
                    repaired.append(None)
                    repaired_mask.append(True)
                    continue

                new_val = cls._repair_object_value_on_read(val, expected_subtype, colname)
                if new_val is None:
                    repaired.append(None)
                    repaired_mask.append(True)
                else:
                    repaired.append(new_val)
                    repaired_mask.append(False)

            obs_tbl.replace_column(
                colname,
                table.MaskedColumn(
                    data=np.array(repaired, dtype=object),
                    mask=np.array(repaired_mask, dtype=bool),
                    name=colname,
                    dtype=object,
                )
            )

        # organize columns in default order
        new_cols = set(obs_tbl.colnames) - set(cls.standard_columns)
        new_cols_original_order = [c for c in obs_tbl.colnames if c in new_cols]
        col_order = list(cls.standard_columns) + new_cols_original_order
        obs_tbl = cls(obs_tbl[col_order])

        return obs_tbl

    def write(self, *args, **kwargs):
        """
        Write a temporary cleaned copy so object columns round-trip robustly.
        """
        catutils.scrub_indices(self)

        tbl = self.copy(copy_data=True)
        subtype_meta = {}

        for colname, expected_subtype in self.object_column_subtypes.items():
            if colname not in tbl.colnames:
                continue

            old = tbl[colname]
            old_mask = np.array(getattr(old, 'mask', np.zeros(len(old), dtype=bool)), dtype=bool)

            cleaned = []
            cleaned_mask = []

            for i, val in enumerate(old):
                if old_mask[i] or self._is_null_like(val):
                    cleaned.append(None)
                    cleaned_mask.append(True)
                    continue

                new_val = self._coerce_object_value_for_write(val, expected_subtype, colname)
                if new_val is None:
                    cleaned.append(None)
                    cleaned_mask.append(True)
                else:
                    cleaned.append(new_val)
                    cleaned_mask.append(False)

            tbl.replace_column(
                colname,
                table.MaskedColumn(
                    data=np.array(cleaned, dtype=object),
                    mask=np.array(cleaned_mask, dtype=bool),
                    name=colname,
                    dtype=object,
                )
            )
            subtype_meta[colname] = expected_subtype.__name__

        tbl.meta[self._OBJECT_SUBTYPE_METAKEY] = subtype_meta

        kwargs.setdefault('format', 'ascii.ecsv')
        kwargs.setdefault('overwrite', False)

        return super(ObsTable, tbl).write(*args, **kwargs)

    def add_flags(self, row_idx, comma_sep_flags):
        self.add_comma_sep_str_to_list_col('flags', row_idx, comma_sep_flags)

    def add_notes(self, row_idx, commas_sep_notes):
        self.add_comma_sep_str_to_list_col('notes', row_idx, commas_sep_notes)

    def add_comma_sep_str_to_list_col(self, colname, row_idx, comma_sep_str):
        items = re.split(', *', comma_sep_str)
        for item in items:
            if not item.strip() == '':
                catutils.append_to_column_of_lists(self, colname, row_idx, item, raise_nonlist_error=False)

    def add_flag(self, row_idx, flag):
        catutils.append_to_column_of_lists(self, 'flags', row_idx, flag, raise_nonlist_error=False)
        self.clean_duplicates_col_of_lists('flags')

    def clean_nulls_col_of_lists(self, colname="flags"):
        old = self[colname]

        new_data = []
        new_mask = []

        old_mask = getattr(old, "mask", np.zeros(len(old), dtype=bool))

        for i, val in enumerate(old):
            if old_mask[i]:
                new_data.append(None)
                new_mask.append(True)
                continue

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
            data=np.array(new_data, dtype=object),
            mask=np.array(new_mask, dtype=bool),
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

    @staticmethod
    def _pretty_diagnostic_format_value(val):
        if val is None or ObsTable._is_null_like(val):
            return '—'
        if isinstance(val, dict):
            if not val:
                return '{}'
            return '{' + ', '.join(f'{k}: {v}' for k, v in val.items()) + '}'
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return '[]'
            return ', '.join(str(x) for x in val)
        return str(val)

    def _pretty_diagnostic_list_lines(self, colname, row_index, bullet_indent):
        col = self[colname]
        if ObsRow._cell_masked_for_index(col, row_index):
            return []
        val = col[row_index]
        if val is None or ObsTable._is_null_like(val):
            return []
        if isinstance(val, str):
            if utils.is_null_item(val):
                return []
            return [f'{bullet_indent}- {val}']
        if isinstance(val, (list, tuple)):
            lines = []
            for item in val:
                if utils.is_null_item(item):
                    continue
                lines.append(f'{bullet_indent}- {item}')
            return lines
        return [f'{bullet_indent}- {val}']

    def pretty_string_with_flags_notes(self, indent='  ', sub_indent='    '):
        """
        Plain-text layout: each row shows all columns except ``flags`` and ``notes`` as
        ``name: value`` lines, then ``flags`` and ``notes`` as indented bullet lists.
        """
        lines = []
        skip_main = {'flags', 'notes'}
        n = len(self)
        for i in range(n):
            elmts = []
            for name in self.colnames:
                if name in skip_main:
                    continue
                masked = ObsRow._cell_masked_for_index(self[name], i)
                if masked:
                    text = '—'
                else:
                    text = self._pretty_diagnostic_format_value(self[name][i])
                elmts.append(f'text')
                lines.append(' | '.join(elmts))
            for label, colname in (('flags', 'flags'), ('notes', 'notes')):
                if colname not in self.colnames:
                    continue
                lines.append(f'{indent}{label}:')
                blines = self._pretty_diagnostic_list_lines(colname, i, sub_indent)
                if not blines:
                    lines.append(f'{sub_indent}(none)')
                else:
                    lines.extend(blines)
            lines.append('')
        return '\n'.join(lines).rstrip() + '\n'

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

    def clear_usability_values(self, id_substr=None, reason_substr=None, other_columns_to_clear=None):
        """Does what the name suggests. Use reason_substr if you only want to clear, e.g., rows where the reason includes
        "acquisition"."""
        cleared_tbl = self.copy()
        if other_columns_to_clear is None:
            other_columns_to_clear = []
        if id_substr is None:
            id_substr = ''
        if reason_substr is None:
            reason_substr = ''

        def get_substr_mask(colname, sub):
            str_col = self[colname].filled('').astype(str)
            return np.char.count(str_col, sub) > 0

        mask = get_substr_mask('archive id', id_substr) & get_substr_mask('reason unusable', reason_substr)
        colanmes_to_clear = ['usable', 'reason unusable'] + other_columns_to_clear
        for name in colanmes_to_clear:
            cleared_tbl[name].mask |= mask
        return cleared_tbl

    def update_usability(self, row_idx, usability_status, reason_unusable=None):
        """
        Set ``usability status`` (and derived fields) for the selected rows. The status
        string drives everything (case-insensitive except for ``reason_unusable`` text).

        * ``'unusable'`` — ``usable`` is ``False`` (unmasked). ``reason_unusable`` is
          required and must be a :data:`reasons_menu` key or its canonical value.
        * ``'has issues'`` — ``usable`` and ``reason unusable`` are masked;
          ``reason_unusable`` must not be passed.
        * ``'all clear'`` (alias ``'usable'``) — ``usable`` is ``True``, ``reason unusable``
          masked; ``reason_unusable`` must not be passed.
        * ``'mask'`` — all three columns are masked; ``reason_unusable`` must not be passed.
        reason_unusable may also be a :data:`reasons_menu` key or its canonical value.
        """
        idx = row_idx
        st = usability_status
        if not isinstance(st, str):
            raise TypeError('usability_status must be a string')
        key = st.strip().lower()
        if key in ('usable', 'clear'):
            key = 'all clear'

        if key == 'mask':
            if reason_unusable is not None:
                raise ValueError('reason_unusable must not be given when usability_status is "mask"')
            self['usability status'][idx] = None
            self['usability status'].mask[idx] = True
            self['reason unusable'][idx] = None
            self['reason unusable'].mask[idx] = True
            self['usable'][idx] = None
            self['usable'].mask[idx] = True
            return

        if key == 'unusable':
            if reason_unusable is None:
                raise ValueError('reason_unusable is required when usability_status is "unusable"')
            if reason_unusable not in reasons_menu.values():
                try:
                    reason_unusable = reasons_menu[reason_unusable]
                except KeyError:
                    raise ValueError(f'Invalid reason unusable {reason_unusable!r}; pass a reasons_menu key or one of: {list(reasons_menu.values())}')
            assert reason_unusable in reasons_menu.values(), f'Invalid reason unusable {reason_unusable!r}; pass a reasons_menu key or one of: {list(reasons_menu.values())}'
            self['usability status'][idx] = 'unusable'
            self['usability status'].mask[idx] = False
            self['usable'][idx] = False
            self['usable'].mask[idx] = False
            self['reason unusable'][idx] = reason_unusable
            self['reason unusable'].mask[idx] = False
            return

        if key == 'has issues':
            if reason_unusable is not None:
                raise ValueError('reason_unusable must not be given when usability_status is "has issues"')
            self['usability status'][idx] = 'has issues'
            self['usability status'].mask[idx] = False
            self['usable'].mask[idx] = True
            self['reason unusable'][idx] = None
            self['reason unusable'].mask[idx] = True
            return

        if key == 'all clear':
            if reason_unusable is not None:
                raise ValueError('reason_unusable must not be given when usability_status is "all clear"')
            self['usability status'][idx] = 'all clear'
            self['usability status'].mask[idx] = False
            self['usable'][idx] = True
            self['usable'].mask[idx] = False
            self['reason unusable'][idx] = None
            self['reason unusable'].mask[idx] = True
            return

        raise ValueError(
            f'Unknown usability_status {usability_status!r}; use one of: '
            f'"unusable", "has issues", "all clear", "mask" (aliases: usable, clear)'
        )

    @classmethod
    def _iter_nonnull_cell_items(cls, val):
        """Yield atomic, non-null items from a cell (scalar or iterable)."""
        if cls._is_null_like(val):
            return
        if isinstance(val, str) or not hasattr(val, '__iter__') or isinstance(val, (bytes, bytearray)):
            items = (val,)
        elif isinstance(val, np.ndarray):
            items = val.tolist()
        else:
            items = tuple(val)
        for item in items:
            if item is None or cls._is_null_like(item):
                continue
            yield item

    def substring_match_mask(self, colname, substr):
        """
        Return a 1-D boolean array: True where any string in ``colname`` contains ``substr``.

        Each cell may be a string, or an iterable of strings (e.g. list). Non-string
        elements are converted with ``str()`` for the test. Masked and null-like cells
        are False in the result. Matching is case-sensitive (``substr in text``).
        """
        col = self[colname]
        row_mask = np.array(getattr(col, 'mask', np.zeros(len(col), dtype=bool)), dtype=bool)
        out = np.zeros(len(col), dtype=bool)

        for i in range(len(col)):
            if row_mask[i]:
                continue
            val = col[i]
            for item in self._iter_nonnull_cell_items(val):
                text = item if isinstance(item, str) else str(item)
                if substr in text:
                    out[i] = True
                    break

        return out


# for backwards compatability
initialize = ObsTable.initialize_blank
get_path = ObsTable.get_path
load_obs_tbl = ObsTable.load_from_targname


def inspect_values_of_all_tables(colname='notes'):
    """
    Load every ``*observation-table.ecsv`` under ``paths.data_targets``, collect every
    distinct atomic value appearing in ``colname`` (string cells and list elements),
    and return that set. Tables that cannot be read or lack ``colname`` are skipped.
    """
    unique = set()
    obstbl_paths = sorted(paths.data_targets.rglob('*observation-table.ecsv'))
    for path in obstbl_paths:
        try:
            tbl = ObsTable.read(path)
        except Exception:
            continue
        if colname not in tbl.colnames:
            continue
        col = tbl[colname]
        row_mask = np.array(getattr(col, 'mask', np.zeros(len(col), dtype=bool)), dtype=bool)
        for i in range(len(col)):
            if row_mask[i]:
                continue
            val = col[i]
            for item in ObsTable._iter_nonnull_cell_items(val):
                unique.add(item if isinstance(item, str) else str(item))
    return unique

reasons_menu = {
    'no data': 'no data taken',
    'shutter closed': 'shutter closed',
    'no gs lock': 'guide star tracking not locked',
    'acq issue + no flux': 'acquisition issues and negligible target flux',
    'acq issue + lo flux': 'acquisition issues and anomalously low target flux',
    'wave target': 'wave exposure'
}

flag_menu = {
    # observation
    'wave target': 'header targname = wave',

    # acquisition
    'bad acq' : 'acquisition untrustworthy',

    # data quality
    'zeros': 'fluxes all zero',
    'nans': 'fluxes all nan or non-finite',

    # flux
    'lo flux': 'flux anomalously low',
    'hi flux': 'flux anomalously high',
    'no flux': 'flux negligible',

    # wavelength
    'bad waves': 'wavelengths inaccurate'
}

notes_menu = {
    # gti issue
    'clock rollover' : 'exposure time falsely reported as zero '
                       'so GTIs were manually replaced based on first and last photon count',

    # acq
    'peakd zeros': 'COS PEAKD counts zero at all dwell points',
    'peakd lo cts': 'COS PEAKD counts < {} at all dwell points whereas values > {} are typical',
    'peakd big slew': 'COS PEAKD slewed {slew_diff:.2f} arcsec away from the count-weighed mean of dwell points '
                      'versus the {atol} threshold for this warning',
    'peakxd zeros': 'COS PEAKXD counts were zero',
    'peakxd big slew': 'COS PEAKXD slewed to a position {slew_diff:.2f} arsec from image centroid, '
                       'versus the {atol} threshold for this warning',
    'acq target flux': 'flux within central tile of {n}x{n} acquisition image tiles is {sigma:.2f} sigma from median',
    'cannot see target': '{user} could not identify target in acquisition image',

    # acq issues not issues
    'acq + plenty flux': 'flux near or above median of same-configuration spectra despite acquisition issues',

    # flux
    'line flux': '{line} flux {sigma:.1f} sigma from median over {wa:.2f}–{wb:.2f} AA band',

    # wavelength discrepancy
    'bad waves': '{user} reported substantial wavelength discrepancy',
}
# note that output from the stistools.tastis function is also used to populate notes