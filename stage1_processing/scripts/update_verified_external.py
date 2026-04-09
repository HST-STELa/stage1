"""
Refresh ``target_selection_data/inputs/hand_checked/verified_external_observations.csv``
from HST observation tables for every target folder under the STELa data root.

Rows are keyed by ``hostname`` and ``tic_id`` from ``reference_files/locked_choices/stela_names.csv``.
Only observations with ``program != 17804`` (non-STELa) count toward pass/fail for Lyα and FUV.

This script is intentionally strict: missing files, bad metadata, unparseable CSV fields, and
unexpected table shapes are supposed to raise (or warn once) so problems surface immediately.

**Row order:** Lines keep the same order as the input CSV. Recomputed targets replace their
existing lines in place. Targets that were not in the CSV are appended at the end, sorted by
hostname (case-insensitive) then ``tic_id``, so ``git diff`` stays line-aligned for unchanged rows.

- **pass**: at least one qualifying row has ``usability status == 'all clear'`` and ``usable`` True,
  and the target does not meet the line-flux-table **lowsnr** downgrade (see below).
- **lowsnr**: would be **pass**, but ``{hostname}.line-flux-table.ecsv`` exists and both (a) and (b) hold:

  (a) Fewer than three non-Lyα lines have finite SNR > 3, *or* among those lines every SNR>3 line
      shares the same spectroscopic ionization (Roman numeral after the element name, e.g. C **II**,
      Si **III**; Lyα rows are ignored).

  (b) Total exposure time (seconds) summed over **all-clear** HST rows whose ``science config`` matches
      at least one instrument mode listed in the line-flux table meta ``source files`` *and* that
      mode belongs to the band (Lyα vs FUV) under consideration, is **< 1000 s**.

- **fail**: qualifying external rows exist but none are all clear.
- **none**: no qualifying rows in the obs table for that band.
- **planned** / **tentative** / **lowsnr**: preserved from the previous CSV when there are still
  no qualifying external rows for that band.

Targets with only program 17804 data (or no obs table) do not get a new row from the DB scan;
CSV lines for those TICs are copied through unchanged in their original positions.

Interactive use (Spyder/Jupyter-style): run the ``#%%`` cell at the bottom after setting ``dry_run``.
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits

import database_utilities as dbutils
import paths
from stage1_processing import observation_table as obt
from stage1_processing import preloads
from stage1_processing import target_lists


#%% settings

dry_run = True
targets = target_lists.everything_in_database()

STELA_PROGRAM_ID = 17804
LOW_SNR_EXPTIME_THRESHOLD_S = 1000.0

# If True, a missing verified CSV starts from an empty old-row list (with a warning).
# If False, ``FileNotFoundError`` when the CSV is absent.
allow_missing_verified_csv = False

LYA_CONFIG_MARKERS = ("stis-g140m", "stis-e140m")
FUV_CONFIG_MARKERS = (
    "stis-e140m",
    "stis-g140l",
    "cos-g140l",
    "cos-g130m",
    "cos-g160m",
)

PRESERVE_WHEN_NO_EXTERNAL_DATA = frozenset({"planned", "tentative", "lowsnr"})
ALLOWED_STATUS_VALUES = frozenset(
    {"pass", "fail", "none", "planned", "tentative", "lowsnr", "unchecked"}
)

_ION_ROMAN_TAIL = re.compile(r"\s+([IVXLC]+)\s*$", re.IGNORECASE)

VERIFIED_FIELDNAMES = ("hostname", "tic_id", "lya", "fuv")

#%% flags and notes to consider when deciding whether usable

external_data_usability = obt.UsabilityDefinition(
    ignore_notes = [
        'GTIs were manually replaced',
        'flux within central tile',
        'was able to identify target',
        'flux near or above median',
        'sigma from median over',
        'sigma from zero over',

        'Telemetry indicates that the intended exposures may not',
        'output lacks some information because',
        'Saturation of pixels in the second image',
        'Your ACQ appears to have succeeded',
        'typical of a successful',
    ],
    fail_notes = [
        'no acquisition found',
        'COS PEAKD counts zero at all dwell points',
        'at all dwell points whereas values',
        'COS PEAKD slewed',
        'COS PEAKXD counts were zero',
        'COS PEAKXD slewed to a position',
        'could not identify target',
        'substantial wavelength discrepancy',
        
        'problem with your acquisition',
        'The flux in the third image of the ACQ is lower',
        'problems in the ACQ/PEAK',
        'inadequate for an accurate',
        'Some pixels in the confirmation image were saturated',
        'The ACQ/PEAK flux test failed',
        'maximum flux in the sequence occurred at one end',
        ],
    ignore_flags = [
        'acquisition untrustworthy',
        'flux anomalously low',
        'flux anomalously high',
        'wavelengths inaccurate', # bad bc also means throughput probably compromised
        ],
    fail_flags = [
        'header targname = wave',
        'fluxes all zero',
        'fluxes all nan or non-finite',
    ],
)


#%% functions

def _norm(s) -> str:
    return str(s).strip().lower()


def _config_matches(cfg: str, markers: tuple[str, ...]) -> bool:
    c = _norm(cfg)
    return any(m in c for m in markers)


def _external_program_flag(prog, *, archive_id: str) -> bool:
    """True if program is a non-STELa HST program. Raises if program is missing or not integral."""
    if prog is None or prog is np.ma.masked:
        raise ValueError(f"HST row archive_id={archive_id!r} has missing or masked program")
    try:
        p = int(prog)
    except (TypeError, ValueError) as e:
        raise ValueError(f"HST row archive_id={archive_id!r} has non-integer program {prog!r}") from e
    return p != STELA_PROGRAM_ID


def _validate_hst_programs(obs_tbl: obt.ObsTable) -> None:
    obsv = obs_tbl["observatory"].filled("").astype(str)
    aid = obs_tbl["archive id"].filled("<missing>").astype(str)
    for i in range(len(obs_tbl)):
        if np.char.lower(str(obsv[i])) != "hst":
            continue
        _external_program_flag(obs_tbl["program"][i], archive_id=str(aid[i]))


def _row_all_clear(obs_tbl: obt.ObsTable, i: int) -> bool:
    us_col = obs_tbl["usability status"]
    u_col = obs_tbl["usable"]
    us_mask = np.asarray(getattr(us_col, "mask", np.zeros(len(obs_tbl), dtype=bool)), dtype=bool)
    u_mask = np.asarray(getattr(u_col, "mask", np.zeros(len(obs_tbl), dtype=bool)), dtype=bool)
    if us_mask[i]:
        return False
    status = us_col[i]
    if status is None or _norm(status) != "all clear":
        return False
    if u_mask[i]:
        return False
    return bool(u_col[i])


def _classify_band(obs_tbl: obt.ObsTable, row_mask: np.ndarray) -> str | None:
    idx = np.nonzero(row_mask)[0]
    if idx.size == 0:
        return None
    for i in idx:
        if _row_all_clear(obs_tbl, int(i)):
            return "pass"
    return "fail"


def _merge_column(obs_result: str | None, old_raw: str, *, col: str, tic: int) -> str:
    if obs_result is not None:
        return obs_result
    o = _norm(old_raw)
    if o in PRESERVE_WHEN_NO_EXTERNAL_DATA:
        return o
    if o not in ALLOWED_STATUS_VALUES:
        raise ValueError(f"Unexpected prior {col!r}={old_raw!r} for tic_id={tic}")
    return "none"


def _load_obs_tbl_if_present(hostfile: str) -> obt.ObsTable | None:
    p = obt.ObsTable.get_path(hostfile)
    if not p.is_file():
        return None
    return obt.ObsTable.read(p)


def _band_masks(obs_tbl: obt.ObsTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_hst_programs(obs_tbl)
    n = len(obs_tbl)
    obsv = obs_tbl["observatory"].filled("").astype(str)
    hst = np.char.lower(np.asarray(obsv)) == "hst"
    aid = obs_tbl["archive id"].filled("<missing>").astype(str)

    prog = obs_tbl["program"]
    pm = getattr(prog, "mask", None)
    prog_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not hst[i]:
            continue
        prog_mask[i] = _external_program_flag(prog[i], archive_id=str(aid[i]))

    cfg_col = obs_tbl["science config"].filled("").astype(str)
    lya = np.array([_config_matches(cfg_col[i], LYA_CONFIG_MARKERS) for i in range(n)], dtype=bool)
    fuv = np.array([_config_matches(cfg_col[i], FUV_CONFIG_MARKERS) for i in range(n)], dtype=bool)

    ext = hst & prog_mask
    return ext, ext & lya, ext & fuv


def _ion_roman_from_line_name(name: str) -> str | None:
    n = str(name).strip().strip('"').strip("'")
    if _norm(n) == "lya":
        return None
    m = _ION_ROMAN_TAIL.search(n)
    if not m:
        return None
    return m.group(1).upper()


def _line_flux_table_condition_a(lf: table.Table) -> bool:
    required = ("name", "snr")
    for c in required:
        if c not in lf.colnames:
            raise ValueError(f"line-flux-table missing required column {c!r}; got {lf.colnames}")
    n = len(lf)
    high_romans: list[str | None] = []
    n_high = 0
    snr_col = lf["snr"]
    for i in range(n):
        nm = str(lf["name"][i]).strip().strip('"').strip("'")
        if _norm(nm) == "lya":
            continue
        s = snr_col[i]
        if np.ma.is_masked(s):
            continue
        if not np.isfinite(s) or float(s) <= 3.0:
            continue
        n_high += 1
        high_romans.append(_ion_roman_from_line_name(nm))
    if n_high < 3:
        return True
    non_null = [r for r in high_romans if r is not None]
    if len(non_null) < n_high:
        return False
    return len(set(non_null)) <= 1


def _configs_from_line_flux_meta(meta, *, path: Path) -> list[str]:
    raw = meta.get("source files", None)
    if raw is None:
        raise ValueError(f"line-flux-table {path} meta missing key 'source files'")
    items = [raw] if isinstance(raw, str) else list(raw)
    if len(items) == 0:
        raise ValueError(f"line-flux-table {path} meta 'source files' is empty")
    configs: list[str] = []
    for item in items:
        s = str(item).strip().strip('"').strip("'")
        if not s:
            raise ValueError(f"line-flux-table {path} has empty entry in source files: {items!r}")
        d = dbutils.parse_filename(Path(s))
        cfg = d.get("config")
        if not cfg:
            raise ValueError(f"Could not get config from source file entry {s!r} in {path}")
        if cfg not in configs:
            configs.append(cfg)
    return configs


def _markers_for_band_from_configs(configs: list[str], band: str) -> tuple[str, ...]:
    if band not in ("lya", "fuv"):
        raise ValueError(f"band must be 'lya' or 'fuv', got {band!r}")
    band_markers = LYA_CONFIG_MARKERS if band == "lya" else FUV_CONFIG_MARKERS
    seen: list[str] = []
    for cfg in configs:
        cl = cfg.lower()
        for m in band_markers:
            if m in cl and m not in seen:
                seen.append(m)
    return tuple(seen)


def _row_exptime_seconds(obs_tbl: obt.ObsTable, i: int, data_dir: Path) -> float:
    ks = obs_tbl["key science files"][i]
    archive_id = str(obs_tbl["archive id"].filled("?")[i])
    if ks is None or obt.ObsTable._is_null_like(ks):
        raise ValueError(f"all-clear row archive_id={archive_id!r} has no key science files for exptime")
    names = list(ks) if not isinstance(ks, str) else [ks]
    tot = 0.0
    for name in names:
        if "x1d" in str(name).lower():
            continue
        fp, = dbutils.find_stela_files_from_hst_filenames([name], data_dir)
        with fits.open(fp) as h:
            eh = h[0].header.get("EXPTIME") or h[1].header.get("EXPTIME", 0)
            if eh is None:
                raise ValueError(f"EXPTIME missing in {fp.name} for archive_id={archive_id!r}")
            tot += float(eh)
    return tot


def _total_exptime_all_clear_for_markers(
    obs_tbl: obt.ObsTable,
    data_dir: Path,
    markers: tuple[str, ...],
) -> float:
    if not markers:
        return 0.0
    cfg_col = obs_tbl["science config"].filled("").astype(str)
    obsv = obs_tbl["observatory"].filled("").astype(str)
    total = 0.0
    for i in range(len(obs_tbl)):
        if np.char.lower(str(obsv[i])) != "hst":
            continue
        if not _row_all_clear(obs_tbl, i):
            continue
        c = cfg_col[i].lower()
        if not any(m in c for m in markers):
            continue
        total += _row_exptime_seconds(obs_tbl, i, data_dir)
    return total


def _should_downgrade_pass_to_lowsnr(
    line_tbl_path: Path,
    obs_tbl: obt.ObsTable | None,
    data_dir: Path,
    band: str,
) -> bool:
    if obs_tbl is None or len(obs_tbl) == 0 or not line_tbl_path.is_file():
        return False
    lf = table.Table.read(line_tbl_path, format="ascii.ecsv")
    if not _line_flux_table_condition_a(lf):
        return False
    configs = _configs_from_line_flux_meta(lf.meta, path=line_tbl_path)
    markers = _markers_for_band_from_configs(configs, band)
    if not markers:
        return False
    texp = _total_exptime_all_clear_for_markers(obs_tbl, data_dir, markers)
    return texp < LOW_SNR_EXPTIME_THRESHOLD_S


def _apply_pass_vs_lowsnr(
    band_result: str | None,
    line_tbl_path: Path,
    obs_tbl: obt.ObsTable | None,
    data_dir: Path,
    band: str,
) -> str | None:
    if band_result != "pass":
        return band_result
    if _should_downgrade_pass_to_lowsnr(line_tbl_path, obs_tbl, data_dir, band):
        return "lowsnr"
    return "pass"


def _read_verified_csv_row(r: dict[str, str], *, row_index: int) -> tuple[int, dict[str, str]]:
    missing = [k for k in VERIFIED_FIELDNAMES if k not in r]
    if missing:
        raise ValueError(f"CSV row {row_index}: missing columns {missing}")
    tic_s = str(r["tic_id"]).strip()
    try:
        tic = int(tic_s)
    except ValueError as e:
        raise ValueError(f"CSV row {row_index}: invalid tic_id {tic_s!r}") from e
    lya, fuv = _norm(r["lya"]), _norm(r["fuv"])
    if lya not in ALLOWED_STATUS_VALUES:
        raise ValueError(f"CSV row {row_index}: invalid lya value {r['lya']!r}")
    if fuv not in ALLOWED_STATUS_VALUES:
        raise ValueError(f"CSV row {row_index}: invalid fuv value {r['fuv']!r}")
    host = str(r["hostname"]).strip()
    if not host:
        raise ValueError(f"CSV row {row_index}: empty hostname for tic_id={tic}")
    return tic, {
        "hostname": host,
        "tic_id": str(tic),
        "lya": lya,
        "fuv": fuv,
    }


def _read_verified_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        if allow_missing_verified_csv:
            warnings.warn(f"Verified CSV not found at {path}; starting with no prior rows.", stacklevel=2)
            return []
        raise FileNotFoundError(f"Verified CSV not found: {path.resolve()}")
    with path.open(newline="", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))
    if rows_in and set(rows_in[0].keys()) != set(VERIFIED_FIELDNAMES):
        raise ValueError(
            f"CSV {path} must have exactly columns {VERIFIED_FIELDNAMES}; got {list(rows_in[0].keys())}"
        )
    parsed: list[dict[str, str]] = []
    for i, r in enumerate(rows_in):
        _, row = _read_verified_csv_row(r, row_index=i + 2)
        parsed.append(row)
    return parsed


def _write_verified_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(VERIFIED_FIELDNAMES))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in VERIFIED_FIELDNAMES})


#%% main

sn = preloads.stela_names
out_path = paths.checked / "verified_external_observations.csv"
old_rows = _read_verified_csv(out_path)
tics_in_old_csv = {int(r["tic_id"]) for r in old_rows}

old_by_tic: dict[int, dict[str, str]] = {}
for r in old_rows:
    tic = int(r["tic_id"])
    if tic not in old_by_tic:
        old_by_tic[tic] = dict(r)  # first row wins for duplicate TICs in the CSV

fresh_by_tic: dict[int, dict[str, str]] = {}
new_tic_append_order: list[int] = []

for hostfile in sorted(targets):
    try:
        sn_row = sn.loc["hostname_file", hostfile]
    except KeyError as e:
        raise KeyError(
            f"Folder {hostfile!r} has no match in stela_names.csv (hostname_file index). "
            "Add it or remove the stray data directory."
        ) from e

    tic = int(sn_row["tic_id"])
    hostname = str(sn_row["hostname"]).strip()
    if not hostname:
        raise ValueError(f"stela_names tic_id={tic} has empty hostname")

    old = old_by_tic.get(tic, {"lya": "none", "fuv": "none"})

    data_dir = paths.target_hst_data(hostfile)
    line_tbl_path = paths.target_data(hostfile) / f"{hostfile}.line-flux-table.ecsv"

    obs_tbl = _load_obs_tbl_if_present(hostfile)
    if obs_tbl is None or len(obs_tbl) == 0:
        lya_obs = None
        fuv_obs = None
        has_ext_lya = False
        has_ext_fuv = False
    else:
        ext, lya_m, fuv_m = _band_masks(obs_tbl)
        has_ext_lya = bool(np.any(lya_m))
        has_ext_fuv = bool(np.any(fuv_m))
        lya_raw = _classify_band(obs_tbl, lya_m)
        fuv_raw = _classify_band(obs_tbl, fuv_m)
        lya_obs = _apply_pass_vs_lowsnr(lya_raw, line_tbl_path, obs_tbl, data_dir, "lya")
        fuv_obs = _apply_pass_vs_lowsnr(fuv_raw, line_tbl_path, obs_tbl, data_dir, "fuv")

    lya_out = _merge_column(lya_obs, old.get("lya", "none"), col="lya", tic=tic)
    fuv_out = _merge_column(fuv_obs, old.get("fuv", "none"), col="fuv", tic=tic)

    preserve_lya = _norm(old.get("lya", "")) in PRESERVE_WHEN_NO_EXTERNAL_DATA
    preserve_fuv = _norm(old.get("fuv", "")) in PRESERVE_WHEN_NO_EXTERNAL_DATA
    include = has_ext_lya or has_ext_fuv or preserve_lya or preserve_fuv

    if not include:
        continue

    row_dict = {
        "hostname": hostname,
        "tic_id": str(tic),
        "lya": lya_out,
        "fuv": fuv_out,
    }
    fresh_by_tic[tic] = row_dict
    if tic not in tics_in_old_csv:
        new_tic_append_order.append(tic)

# Preserve original CSV line order; swap in recomputed rows where we have them.
new_rows: list[dict[str, str]] = []
for r in old_rows:
    tic = int(r["tic_id"])
    new_rows.append(fresh_by_tic[tic] if tic in fresh_by_tic else dict(r))

# New targets (in database + included, but absent from the prior CSV).
new_tic_append_order.sort(key=lambda t: (_norm(fresh_by_tic[t]["hostname"]), t))
for tic in new_tic_append_order:
    new_rows.append(fresh_by_tic[tic])

print(f"Wrote {len(new_rows)} rows (dry_run={dry_run}).")
if dry_run:
    for r in new_rows[:20]:
        print(f"  {r}")
    if len(new_rows) > 20:
        print(f"  ... and {len(new_rows) - 20} more")
else:
    _write_verified_csv(out_path, new_rows)
    print(f"Updated {out_path.resolve()}")
