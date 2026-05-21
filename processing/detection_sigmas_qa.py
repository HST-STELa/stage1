"""QA helpers for per-planet detection-sigmas.ecsv tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy import units as u
from astropy.table import Table

from processing import transit_evaluation_utilities as tutils

ROW_COUNT_BUCKETS = ('model_no_cos', 'model_cos', 'flat')

COMMON_COLS = [
    'transit velocity ranges',
    'normalization velocity ranges',
    'normalized',
    'transit sigma',
    'time offset',
    'grating',
    'aperture',
    'lya reconstruction case',
]

MODEL_COLS_CURRENT = COMMON_COLS + [
    'eta',
    'Tion',
    'sw_ram_pressure_at_pl',
    'sw_velocity',
    'mass',
]

MODEL_COLS_LEGACY = COMMON_COLS + [
    'eta',
    'mdot_star',
    'Tion',
    'mass',
]

FLAT_COLS = COMMON_COLS + [
    'start',
    'stop',
    'rv_blue',
    'rv_red',
    'depth',
]

REQUIRED_META_KEYS = [
    'best time offset',
    'best safe time offset',
    'best base grating aperture',
    'base grating',
    'base aperture',
    'offsets considered',
    'max safe offset',
    'COS considered',
]

MASS_MIN_MEARTH = 0.1
MASS_MAX_MEARTH = (10 * u.Mjup).to_value(u.Mearth)

MIN_SAMPLE_CHECK = 5**4
MTIME_MISMATCH_DAYS = 30


@dataclass
class QaIssue:
    planet_key: str
    tst_type: str
    path: Path | None
    check: str
    detail: str


def planet_key(host, planet) -> str:
    return f'{host.hostname} {planet.stela_suffix}'


def expected_snr_paths(host, planet) -> dict[str, Path]:
    paths = {}
    for tst_type in ('model', 'flat'):
        filenamer = tutils.FileNamer(tst_type, planet, host)
        paths[tst_type] = filenamer.snr_tbl_full
    return paths


def row_count_bucket(tst_type: str, cos_considered: bool) -> str:
    if tst_type == 'flat':
        return 'flat'
    return 'model_cos' if cos_considered else 'model_no_cos'


def _col_values(tbl, name):
    col = tbl[name]
    if hasattr(col, 'quantity'):
        return col.quantity.value
    return np.asarray(col, dtype=float)


def _col_unit(tbl, name):
    col = tbl[name]
    if hasattr(col, 'quantity'):
        return col.unit
    info = tbl.info[name]
    if getattr(info, 'unit', None) is not None:
        return info.unit
    return None


def _values_outside_range(values, lo, hi, name):
    bad = ~np.isfinite(values) | (values < lo) | (values > hi)
    n_bad = int(np.sum(bad))
    if n_bad:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        return f'{n_bad} of {len(values)} {name} outside [{lo}, {hi}] (min={vmin}, max={vmax})'
    return None


def _issue(planet_key_str, tst_type, path, check, detail) -> QaIssue:
    return QaIssue(planet_key_str, tst_type, path, check, detail)


def _path_tail(path: Path | None, n: int = 4) -> str:
    if path is None:
        return ''
    return str(Path(*path.parts[-n:]))


def validate_detection_sigmas_table(
        path: Path,
        tst_type: str,
        planet_key_str: str,
        *,
        stale_cutoff: datetime | None,
        bucket_expected_rows: dict[str, int | None],
        model_colnames_ref: list[str] | None,
        flat_colnames_ref: list[str] | None,
        run_offset_stats_smoke_test: bool = False,
        sigma_threshold: float = 1,
        check_h5_companion: bool = False,
        host=None,
        planet=None,
) -> tuple[list[QaIssue], list[str] | None, list[str] | None]:
    issues: list[QaIssue] = []

    if not path.exists():
        issues.append(_issue(planet_key_str, tst_type, path, 'missing_file', 'file not found'))
        return issues, model_colnames_ref, flat_colnames_ref

    if stale_cutoff is not None:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if mtime < stale_cutoff:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'stale_mtime',
                f'mtime {mtime:%Y-%m-%d %H:%M} < cutoff {stale_cutoff:%Y-%m-%d}',
            ))

    try:
        tbl = Table.read(path)
    except Exception as exc:
        issues.append(_issue(planet_key_str, tst_type, path, 'read_error', str(exc)))
        return issues, model_colnames_ref, flat_colnames_ref

    colnames = list(tbl.colnames)
    has_outdated_schema = False

    if tst_type == 'model':
        if 'mdot_star' in colnames and 'sw_ram_pressure_at_pl' not in colnames:
            has_outdated_schema = True
            issues.append(_issue(
                planet_key_str, tst_type, path, 'outdated_schema',
                'has mdot_star but not sw_ram_pressure_at_pl / sw_velocity',
            ))
        if model_colnames_ref is None:
            model_colnames_ref = colnames
        elif colnames != model_colnames_ref:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'column_mismatch',
                f'columns differ from reference ({len(colnames)} vs {len(model_colnames_ref)})',
            ))
        if not has_outdated_schema:
            missing = set(MODEL_COLS_CURRENT) - set(colnames)
            if missing:
                issues.append(_issue(
                    planet_key_str, tst_type, path, 'missing_columns',
                    f'missing {sorted(missing)}',
                ))
    else:
        if flat_colnames_ref is None:
            flat_colnames_ref = colnames
        elif colnames != flat_colnames_ref:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'column_mismatch',
                f'columns differ from reference ({len(colnames)} vs {len(flat_colnames_ref)})',
            ))
        missing = set(FLAT_COLS) - set(colnames)
        if missing:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'missing_columns',
                f'missing {sorted(missing)}',
            ))

    for key in REQUIRED_META_KEYS:
        if key not in tbl.meta:
            issues.append(_issue(planet_key_str, tst_type, path, 'missing_meta', f'missing meta key {key!r}'))

    cos = tbl.meta.get('COS considered', None)
    if cos is not None:
        gratings = set(np.asarray(tbl['grating']).astype(str))
        if cos:
            if 'g130m' not in gratings:
                issues.append(_issue(
                    planet_key_str, tst_type, path, 'cos_rows', 'COS considered but no g130m rows',
                ))
            else:
                mask = (tbl['grating'] == 'g130m') & (tbl['aperture'] == 'psa')
                if not np.any(mask):
                    issues.append(_issue(
                        planet_key_str, tst_type, path, 'cos_rows', 'COS considered but no g130m+psa rows',
                    ))
        elif 'g130m' in gratings:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'cos_rows', 'COS not considered but g130m rows present',
            ))

    if 'transit sigma' in colnames:
        sig = np.asarray(tbl['transit sigma'], dtype=float)
        bad = ~np.isfinite(sig) | (sig < 0)
        if np.any(bad):
            issues.append(_issue(
                planet_key_str, tst_type, path, 'transit_sigma_invalid',
                f'{int(np.sum(bad))} values not finite and >= 0',
            ))

    if 'lya reconstruction case' in colnames:
        valid = set(tutils.LyaReconstruction.case_labels)
        cases = set(np.asarray(tbl['lya reconstruction case']).astype(str))
        bad = cases - valid
        if bad:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'lya_case_invalid',
                f'invalid cases: {sorted(bad)}',
            ))

    if tst_type == 'model' and not has_outdated_schema:
        if 'eta' in colnames:
            msg = _values_outside_range(_col_values(tbl, 'eta'), 0.01, 1.0, 'eta')
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'eta_range', msg))
        if 'Tion' in colnames:
            Tion_h = _col_values(tbl, 'Tion')
            if hasattr(tbl['Tion'], 'quantity'):
                Tion_h = tbl['Tion'].quantity.to_value('h')
            msg = _values_outside_range(Tion_h, 1e-2, 1e3, 'Tion [h]')
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'Tion_range', msg))
        if 'sw_velocity' in colnames:
            v = _col_values(tbl, 'sw_velocity')
            if hasattr(tbl['sw_velocity'], 'quantity'):
                v = tbl['sw_velocity'].quantity.to_value('km / s')
            msg = _values_outside_range(v, 10.0, 1e3, 'sw_velocity [km/s]')
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'sw_velocity_range', msg))
        if 'sw_ram_pressure_at_pl' in colnames:
            p = _col_values(tbl, 'sw_ram_pressure_at_pl')
            if hasattr(tbl['sw_ram_pressure_at_pl'], 'quantity'):
                p = tbl['sw_ram_pressure_at_pl'].quantity.to_value('dyn / cm2')
            msg = _values_outside_range(p, 1e-15, 1e5, 'sw_ram_pressure_at_pl')
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'sw_ram_pressure_range', msg))
        if 'mass' in colnames:
            mass_unit = _col_unit(tbl, 'mass')
            if mass_unit is not None and mass_unit != u.g:
                issues.append(_issue(
                    planet_key_str, tst_type, path, 'mass_unit',
                    f'mass unit is {mass_unit}, expected g',
                ))
            if hasattr(tbl['mass'], 'quantity'):
                mass_me = tbl['mass'].quantity.to_value('Mearth')
            else:
                mass_me = (_col_values(tbl, 'mass') * u.g).to_value(u.Mearth)
            # small relative tolerance: masses stored in g can sit just below 0.1 Mearth
            msg = _values_outside_range(
                mass_me,
                MASS_MIN_MEARTH * (1 - 1e-4),
                MASS_MAX_MEARTH * (1 + 1e-4),
                'mass [Mearth]',
            )
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'mass_range', msg))

    if tst_type == 'flat':
        if 'depth' in colnames:
            depth = np.asarray(tbl['depth'], dtype=float)
            msg = _values_outside_range(depth, 0.0, 1.0, 'depth')
            if msg:
                issues.append(_issue(planet_key_str, tst_type, path, 'depth_range', msg))
        if all(c in colnames for c in ('rv_blue', 'rv_red')):
            rv_blue = _col_values(tbl, 'rv_blue')
            rv_red = _col_values(tbl, 'rv_red')
            if hasattr(tbl['rv_blue'], 'quantity'):
                rv_blue = tbl['rv_blue'].quantity.to_value('km / s')
                rv_red = tbl['rv_red'].quantity.to_value('km / s')
            if np.any(rv_blue >= rv_red):
                issues.append(_issue(planet_key_str, tst_type, path, 'rv_range', 'rv_blue >= rv_red for some rows'))
        if all(c in colnames for c in ('start', 'stop')):
            start = _col_values(tbl, 'start')
            stop = _col_values(tbl, 'stop')
            if np.any(start >= stop):
                issues.append(_issue(planet_key_str, tst_type, path, 'time_range', 'start >= stop for some rows'))

    blocking = {i.check for i in issues} & {
        'read_error', 'missing_file', 'outdated_schema', 'missing_columns', 'missing_meta',
    }
    if cos is not None and not blocking:
        bucket = row_count_bucket(tst_type, bool(cos))
        nrows = len(tbl)
        expected = bucket_expected_rows.get(bucket)
        if expected is None:
            bucket_expected_rows[bucket] = nrows
        elif nrows != expected:
            issues.append(_issue(
                planet_key_str, tst_type, path, 'row_count_mismatch',
                f'{nrows} rows, expected {expected} for bucket {bucket}',
            ))

    if run_offset_stats_smoke_test and tst_type == 'model' and not blocking:
        try:
            snr_db = tutils.DetectabilityDatabase.from_file(path)
            snr_db.offset_stats(sigma_threshold, min_sample_check=MIN_SAMPLE_CHECK)
        except Exception as exc:
            issues.append(_issue(planet_key_str, tst_type, path, 'offset_stats_smoke', str(exc)))

    if check_h5_companion and tst_type == 'model' and host is not None and planet is not None:
        pattern = f'{host.dbname}-{planet.stela_suffix}.outflow-tail-model*.h5'
        if not any(host.folder.rglob(pattern)):
            issues.append(_issue(
                planet_key_str, tst_type, path, 'missing_h5_companion',
                f'no file matching {pattern}',
            ))

    return issues, model_colnames_ref, flat_colnames_ref


def validate_all_targets(
        targets: Iterable[str],
        host_catalog,
        planet_catalog,
        *,
        stale_cutoff: datetime | None = None,
        run_offset_stats_smoke_test: bool = False,
        sigma_threshold: float = 1,
        check_h5_companion: bool = False,
        mtime_mismatch_days: int = MTIME_MISMATCH_DAYS,
) -> tuple[list[QaIssue], set[str]]:
    all_issues: list[QaIssue] = []
    bucket_expected_rows: dict[str, int | None] = {b: None for b in ROW_COUNT_BUCKETS}
    model_colnames_ref: list[str] | None = None
    flat_colnames_ref: list[str] | None = None
    mtimes_by_planet: dict[str, dict[str, datetime]] = defaultdict(dict)

    for target in targets:
        host = tutils.Host(target, host_catalog, planet_catalog)
        for planet in host.planets:
            pkey = planet_key(host, planet)
            paths = expected_snr_paths(host, planet)

            for tst_type, path in paths.items():
                if path.exists():
                    mtimes_by_planet[pkey][tst_type] = datetime.fromtimestamp(path.stat().st_mtime)

            model_exists = paths['model'].exists()
            flat_exists = paths['flat'].exists()
            if model_exists != flat_exists:
                missing = 'model' if not model_exists else 'flat'
                all_issues.append(_issue(
                    pkey, missing, paths[missing], 'incomplete_pair',
                    f'{missing} detection-sigmas missing',
                ))

            for tst_type, path in paths.items():
                file_issues, model_colnames_ref, flat_colnames_ref = validate_detection_sigmas_table(
                    path,
                    tst_type,
                    pkey,
                    stale_cutoff=stale_cutoff,
                    bucket_expected_rows=bucket_expected_rows,
                    model_colnames_ref=model_colnames_ref,
                    flat_colnames_ref=flat_colnames_ref,
                    run_offset_stats_smoke_test=run_offset_stats_smoke_test,
                    sigma_threshold=sigma_threshold,
                    check_h5_companion=check_h5_companion,
                    host=host,
                    planet=planet,
                )
                all_issues.extend(file_issues)

            mt = mtimes_by_planet.get(pkey, {})
            if 'model' in mt and 'flat' in mt:
                delta = abs(mt['model'] - mt['flat'])
                if delta > timedelta(days=mtime_mismatch_days):
                    all_issues.append(_issue(
                        pkey, 'pair', None, 'mtime_mismatch',
                        f'model vs flat mtime differ by {delta.days} days',
                    ))

    excluded = {issue.planet_key for issue in all_issues}
    return all_issues, excluded


def planet_qa_reason(issues: list[QaIssue], planet_key_str: str) -> str:
    parts = []
    seen = set()
    for issue in issues:
        if issue.planet_key != planet_key_str:
            continue
        label = f'{issue.check}: {issue.detail}'
        if label not in seen:
            seen.add(label)
            parts.append(label)
    return '; '.join(parts)


def print_qa_report(issues: list[QaIssue]) -> None:
    if not issues:
        print('No detection-sigmas QA issues.')
        return
    by_planet: dict[str, list[QaIssue]] = defaultdict(list)
    for issue in issues:
        by_planet[issue.planet_key].append(issue)
    for pkey in sorted(by_planet):
        print(f'\n{pkey}')
        for issue in by_planet[pkey]:
            print(f'  [{issue.tst_type}] {issue.check}: {issue.detail}')
            print(f'    {_path_tail(issue.path)}')


def qa_summary_table(issues: list[QaIssue]) -> Table:
    counts = Counter(issue.check for issue in issues)
    return Table(names=['check', 'count'], data=[list(counts.keys()), list(counts.values())])
