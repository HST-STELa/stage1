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

- **pass**: at least one qualifying row passes ``ObsTable.custom_usability_mask`` with
  ``external_data_usability`` (see settings cell), and the target does not meet the per-config
  ``*.line_fluxes.ecsv`` **lowsnr** downgrade (see below).
- **lowsnr**: would be **pass**, but there is at least one ``*.line_fluxes.ecsv`` under the target
  HST data tree (see below) that is relevant to the band, and **every** such table (after picking one
  file per instrument config; see priority) satisfies **both** (a) and (b):

  (a) Fewer than three non-Lyα lines have finite SNR > 3, *or* among those lines every SNR>3 line
      shares the same spectroscopic ionization (Roman numeral after the element name, e.g. C **II**,
      Si **III**; Lyα rows are ignored).

  (b) Total exposure time (seconds) summed over HST rows that pass ``custom_usability_mask`` and whose
      ``science config`` matches at least one instrument mode associated with that line-flux table
      (from meta ``source files`` when present, otherwise from the filename) *and* that mode belongs
      to the band (Lyα vs FUV) under consideration, is **< 1000 s**.

  If **any** relevant config's chosen table does **not** satisfy (a) (good SNR) **or** does **not**
  satisfy (b) (exptime ≥ 1000 s), the band stays **pass** instead of **lowsnr**.

  Line-flux files are discovered under ``<target>/hst/ecsv/*.line_fluxes.ecsv`` when that directory
  exists, and under ``<target>/hst/*.line_fluxes.ecsv`` (deduplicated). For multiple files sharing the
  same instrument config (second ``.``-separated field of the basename, e.g. ``hst-cos-g140l``), the
  file kept is the one with the highest ``{n}exposure_coadd`` *n*; if none match, prefer ``x1dsum.``
  over ``x1d.`` in the basename.

- **fail**: qualifying external rows exist but none pass the custom usability mask.
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
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits

import database_utilities as dbutils
import paths
from processing import observation_table as obt
from processing import preloads
from processing import target_lists


#%% settings

dry_run = False
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

# ``action_on_unknown`` for ``custom_usability_mask``: flag/note substrings not in fail/ignore lists.
CUSTOM_USABILITY_ACTION_ON_UNKNOWN = "error"

#%% flags and notes to consider when deciding whether usable

external_data_usability = obt.UsabilityDefinition(
    ignore_notes = [
        'bot note',
        'bot warning acq image flux',  # leave out because should always have accompanying seen/unseen note in latest review
        'identified target in acquisition',

        'Telemetry indicates that the intended exposures may not',
        'output lacks some information because',
        'Saturation of pixels in the second image',
        'Your ACQ appears to have succeeded',
        'typical of a successful',
        'bot warning science flux',
    ],
    fail_notes = [
        'no acquisition found',
        'COS PEAKD counts zero at all dwell points',
        'bot warning COS PEAKD max',
        'bot warning COS PEAKD slew',
        'COS PEAKXD counts were zero',
        'bot warning COS PEAKXD slew',
        'deemed target absent',
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
        'flux negligible',
        ],
    fail_flags = [
        'header targname = wave',
        'fluxes all zero',
        'fluxes all nan or non-finite',
        'wavelengths inaccurate', # bad bc also means throughput probably compromised
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


def _archival_pass_mask(obs_tbl: obt.ObsTable) -> np.ndarray:
    """Boolean mask: True where ``custom_usability_mask`` passes (and ``usable`` if retained)."""
    mask, _uncat = obs_tbl.custom_usability_mask(
        external_data_usability,
        action_on_unknown=CUSTOM_USABILITY_ACTION_ON_UNKNOWN,
        retain_unusable_flags=True,
    )
    return np.asarray(mask, dtype=bool)


def _classify_band(row_mask: np.ndarray, archival_pass_mask: np.ndarray) -> str | None:
    """pass / fail / None (no rows in ``row_mask``)."""
    idx = np.nonzero(row_mask)[0]
    if idx.size == 0:
        return None
    if np.any(row_mask & archival_pass_mask):
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


def _config_from_line_fluxes_filename(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) < 3 or parts[-1] != "ecsv" or parts[-2] != "line_fluxes":
        raise ValueError(f"Expected *.line_fluxes.ecsv, got {path.name!r}")
    cfg = parts[1]
    if "hst-" not in cfg:
        raise ValueError(f"Could not read instrument config from line_fluxes basename {path.name!r}")
    return cfg


def _configs_for_line_fluxes_table(path: Path, lf: table.Table) -> list[str]:
    raw = lf.meta.get("source files", None)
    if raw is None:
        return [_config_from_line_fluxes_filename(path)]
    try:
        return _configs_from_line_flux_meta(lf.meta, path=path)
    except ValueError:
        return [_config_from_line_fluxes_filename(path)]


def _discover_line_fluxes_ecsv_paths(data_dir: Path) -> list[Path]:
    """``hst/ecsv/*.line_fluxes.ecsv`` (if present) plus ``hst/*.line_fluxes.ecsv``, deduped."""
    seen: set[Path] = set()
    out: list[Path] = []
    ecsv_sub = data_dir / "ecsv"
    if ecsv_sub.is_dir():
        for p in sorted(ecsv_sub.glob("*.line_fluxes.ecsv")):
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                out.append(p)
    for p in sorted(data_dir.glob("*.line_fluxes.ecsv")):
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            out.append(p)
    return out


_COADD_N_RE = re.compile(r"(\d+)exposure_coadd")


def _line_fluxes_priority_rank(path: Path) -> tuple[int, int, int]:
    """Higher tuple sorts later under ``max``: prefer larger coadd *n*, then ``x1dsum.``, then ``x1d.``."""
    name = path.name
    coadd_ns = [int(m) for m in _COADD_N_RE.findall(name)]
    coadd_n = max(coadd_ns) if coadd_ns else 0
    x1dsum = 1 if "x1dsum." in name else 0
    x1d = 1 if "x1d." in name else 0
    return (coadd_n, x1dsum, x1d)


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
        raise ValueError(
            f"archive_id={archive_id!r} passes custom usability but has no key science files for exptime"
        )
    names = list(ks) if not isinstance(ks, str) else [ks]
    tot = 0.0
    for name in names:
        fp, = dbutils.find_stela_files_from_hst_filenames([name], data_dir)
        with fits.open(fp) as h:
            eh = h[0].header.get("EXPTIME") or h[1].header.get("EXPTIME", 0)
            if eh is None:
                raise ValueError(f"EXPTIME missing in {fp.name} for archive_id={archive_id!r}")
            tot += float(eh)
    return tot


def _total_exptime_archival_pass_for_markers(
    obs_tbl: obt.ObsTable,
    data_dir: Path,
    markers: tuple[str, ...],
    archival_pass_mask: np.ndarray,
) -> float:
    if not markers:
        return 0.0
    cfg_col = obs_tbl["science config"].filled("").astype(str)
    obsv = obs_tbl["observatory"].filled("").astype(str)
    total = 0.0
    for i in range(len(obs_tbl)):
        if np.char.lower(str(obsv[i])) != "hst":
            continue
        if not archival_pass_mask[i]:
            continue
        c = cfg_col[i].lower()
        if not any(m in c for m in markers):
            continue
        total += _row_exptime_seconds(obs_tbl, i, data_dir)
    return total


def _should_downgrade_pass_to_lowsnr(
    data_dir: Path,
    obs_tbl: obt.ObsTable | None,
    band: str,
    archival_pass_mask: np.ndarray,
) -> bool:
    if obs_tbl is None or len(obs_tbl) == 0:
        return False
    lf_paths = _discover_line_fluxes_ecsv_paths(data_dir)
    if not lf_paths:
        return False
    by_cfg: dict[str, list[Path]] = defaultdict(list)
    for p in lf_paths:
        try:
            key = _config_from_line_fluxes_filename(p)
        except ValueError:
            continue
        by_cfg[key].append(p)

    saw_band_relevant = False
    for _cfg, group in by_cfg.items():
        chosen = max(group, key=lambda p: (_line_fluxes_priority_rank(p), p.name))
        lf = table.Table.read(chosen, format="ascii.ecsv")
        try:
            configs = _configs_for_line_fluxes_table(chosen, lf)
        except ValueError:
            continue
        markers = _markers_for_band_from_configs(configs, band)
        if not markers:
            continue
        saw_band_relevant = True
        bad_snr = _line_flux_table_condition_a(lf)
        texp = _total_exptime_archival_pass_for_markers(obs_tbl, data_dir, markers, archival_pass_mask)
        low_exp = texp < LOW_SNR_EXPTIME_THRESHOLD_S
        if (not bad_snr) or (not low_exp):
            return False
    return saw_band_relevant


def _apply_pass_vs_lowsnr(
    band_result: str | None,
    obs_tbl: obt.ObsTable | None,
    data_dir: Path,
    band: str,
    archival_pass_mask: np.ndarray,
) -> str | None:
    if band_result != "pass":
        return band_result
    if _should_downgrade_pass_to_lowsnr(data_dir, obs_tbl, band, archival_pass_mask):
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

    obs_tbl = obt.load_obs_tbl(hostfile)
    if obs_tbl is None or len(obs_tbl) == 0:
        lya_obs = None
        fuv_obs = None
        has_ext_lya = False
        has_ext_fuv = False
    else:
        _ext, lya_m, fuv_m = _band_masks(obs_tbl)
        archival_pass_mask = _archival_pass_mask(obs_tbl)
        has_ext_lya = bool(np.any(lya_m))
        has_ext_fuv = bool(np.any(fuv_m))
        lya_raw = _classify_band(lya_m, archival_pass_mask)
        fuv_raw = _classify_band(fuv_m, archival_pass_mask)
        lya_obs = _apply_pass_vs_lowsnr(lya_raw, obs_tbl, data_dir, "lya", archival_pass_mask)
        fuv_obs = _apply_pass_vs_lowsnr(fuv_raw, obs_tbl, data_dir, "fuv", archival_pass_mask)

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



#%% custom determinations

"""
next you went through the git diff of the verified obs file and restored the following lines to their original determinations
toi-421 kept lya as pass bc target is clear in the image and lya is bright enough that we already selected for stage 2
all others looked good
"""