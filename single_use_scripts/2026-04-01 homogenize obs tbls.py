"""
Rewrite legacy strings in observation-table ECSV files to match *_menu canonical
wording in observation_table.py. Drops flags/notes/reason lines that concern
acquisition or lo/hi/no target flux (to be rebuilt later).

Set DRY_RUN = False to write files (overwrite=True).

(File dated 2024-04-01 per request; authored 2026-04-01.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy import table

# repo root (parent of single_use_scripts)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths
from stage1_processing import observation_table as obt

DRY_RUN = True

fm, nm, rm = obt.flag_menu, obt.notes_menu, obt.reasons_menu

# Legacy full strings from tables -> canonical menu *values*
FLAG_LEGACY_TO_CANON = {
    'header targname=wave': fm['wave target'],
    'All spectral values are NaN/non-finite.': fm['nans'],
    'Spectrum is all zeros.': fm['zeros'],
    'inaccurate wavelengths': fm['bad waves'],
}

NOTES_LEGACY_TO_CANON = {
    (
        'oecb11010_tag.fits had data but header set to zero exposure time. '
        'Manually replaced GTIs based on first and last photon count.'
    ): nm['clock rollover'],
    (
        'oecr02040_tag.fits had data but header set to zero exposure time. '
        'Manually replaced GTIs based on first and last photon count.'
    ): nm['clock rollover'],
}

REASON_LEGACY_TO_CANON = {
    'No data taken.': rm['no data'],
    'Shutter closed.': rm['shutter closed'],
}

# Canonical reason strings we never mask via heuristics
_REASON_PROTECTED = frozenset(
    {rm['no data'], rm['shutter closed'], rm['wave target'], 'wave exposure'}
)


def _norm_key(s: str) -> str:
    return s.strip()


def _flag_item_should_drop(s: str) -> bool:
    low = s.lower()
    if any(
        frag in low
        for frag in (
            'peakd',
            'peakxd',
            'stistools.tastis',
            'tastis logged acquisition',
            'acquisition untrustworthy',
            'identify target in acquisition',
            'missed acquisition',
        )
    ):
        return True
    if any(
        frag in low
        for frag in (
            'anomalously low target flux',
            'negligible target flux',
            'no target flux',
            'flux present',
        )
    ):
        return True
    if low in ('no flux', 'no flux.'):
        return True
    if 'acquisition issues and' in low:
        return True
    return False


def _note_item_should_drop(s: str) -> bool:
    low = s.lower()
    if 'despite acquisition issues' in low:
        return True
    if 'flux above median' in low and 'acquisition' in low:
        return True
    if 'flux near median' in low and 'acquisition' in low:
        return True
    return _flag_item_should_drop(s)


def _reason_should_mask(s: str) -> bool:
    t = _norm_key(s).rstrip('.')
    if t.lower() in {x.lower() for x in _REASON_PROTECTED}:
        return False
    low = s.lower()
    if 'acquisition' in low or 'missed acquisition' in low:
        return True
    if any(
        frag in low
        for frag in (
            'anomalously low target flux',
            'negligible target flux',
            'no target flux',
            'no flux.',
            'no flux',
        )
    ):
        return True
    if low == 'no flux':
        return True
    return False


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _replace_list_column(tbl: obt.ObsTable, colname: str, map_legacy: dict, should_drop) -> bool:
    """should_drop(s: str) -> bool. Returns True if any row changed."""
    col = tbl[colname]
    old_mask = np.array(getattr(col, 'mask', np.zeros(len(col), dtype=bool)), dtype=bool)
    new_data: list = []
    new_mask: list[bool] = []
    changed = False

    for i in range(len(col)):
        if old_mask[i]:
            new_data.append(None)
            new_mask.append(True)
            continue
        val = col[i]
        if obt.ObsTable._is_null_like(val):
            new_data.append(None)
            new_mask.append(True)
            continue

        raw_items = [x if isinstance(x, str) else str(x) for x in obt.ObsTable._iter_nonnull_cell_items(val)]
        built: list[str] = []
        row_changed = False
        for s in raw_items:
            if should_drop(s):
                row_changed = True
                continue
            s2 = map_legacy.get(_norm_key(s), s)
            if s2 != s:
                row_changed = True
            built.append(s2)

        before_dedupe_len = len(built)
        built = _dedupe_preserve_order(built)
        if len(built) != before_dedupe_len:
            row_changed = True

        if row_changed:
            changed = True

        if len(built) == 0:
            new_data.append(None)
            new_mask.append(True)
        else:
            new_data.append(built)
            new_mask.append(False)

    tbl.replace_column(
        colname,
        table.MaskedColumn(
            data=np.array(new_data, dtype=object),
            mask=np.array(new_mask, dtype=bool),
            name=colname,
            dtype=object,
        ),
    )
    return changed


def _replace_reason_column(tbl: obt.ObsTable) -> bool:
    col = tbl['reason unusable']
    old_mask = np.array(getattr(col, 'mask', np.zeros(len(col), dtype=bool)), dtype=bool)
    new_data: list = []
    new_mask: list[bool] = []
    changed = False

    for i in range(len(col)):
        if old_mask[i]:
            new_data.append(None)
            new_mask.append(True)
            continue
        s = col[i]
        if obt.ObsTable._is_null_like(s):
            new_data.append(None)
            new_mask.append(True)
            continue
        if not isinstance(s, str):
            s = str(s)
        key = _norm_key(s)
        out = REASON_LEGACY_TO_CANON.get(key, s)
        if _reason_should_mask(out):
            new_data.append(None)
            new_mask.append(True)
            if s.strip():
                changed = True
        else:
            new_data.append(out)
            new_mask.append(False)
            if out != s:
                changed = True

    tbl.replace_column(
        'reason unusable',
        table.MaskedColumn(
            data=np.array(new_data, dtype=object),
            mask=np.array(new_mask, dtype=bool),
            name='reason unusable',
            dtype=object,
        ),
    )
    return changed


def homogenize_table(tbl: obt.ObsTable) -> bool:
    """Apply mappings and drops. Returns True if anything changed."""
    c1 = _replace_list_column(tbl, 'flags', FLAG_LEGACY_TO_CANON, _flag_item_should_drop)
    c2 = _replace_list_column(tbl, 'notes', NOTES_LEGACY_TO_CANON, _note_item_should_drop)
    c3 = _replace_reason_column(tbl)
    any_changed = c1 or c2 or c3
    if any_changed:
        tbl.clean_duplicates_col_of_lists('flags')
        tbl.clean_duplicates_col_of_lists('notes')
    return any_changed


def main():
    paths_sorted = sorted(paths.data_targets.rglob('*observation-table.ecsv'))
    n_files = 0
    n_changed = 0
    for path in paths_sorted:
        try:
            tbl = obt.ObsTable.read(path)
        except Exception as exc:
            print(f'skip read error {path}: {exc}')
            continue
        n_files += 1
        if homogenize_table(tbl):
            n_changed += 1
            print(f'{"[dry-run] would write" if DRY_RUN else "write"} {path}')
            if not DRY_RUN:
                tbl.write(path, overwrite=True)
    print(f'done: {n_files} tables read, {n_changed} with changes, dry_run={DRY_RUN}')


if __name__ == '__main__':
    main()
