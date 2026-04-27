"""
Evaluate archival transit coverage for caller-supplied hostnames using observation-table
``phase_*`` columns and ``ObsTable.custom_usability_mask``.

Intended to be imported and called from other scripts (not as a standalone entry point).

Like ``update_verified_external``, this walks ``target_lists.everything_in_database()`` and only
processes rows whose ``hostname`` is in the provided ``target_hostnames`` sequence.

``science_config_substrings`` (default ``None``): if set, only rows whose ``science config`` string
contains **any** of these substrings (case-insensitive, e.g. ``stis-g140m`` matches
``hst-stis-g140m``) contribute toward exposure totals, phase-error checks, and the linked
pre-transit + transit-near criteria below; if ``None``, all configs count.

``refresh_observed_transit_list`` accepts optional ``usability_definition`` and ``action_on_unknown``
(same meaning as ``ObsTable.custom_usability_mask``); defaults are ``DEFAULT_ARCHIVAL_USABILITY`` and
``DEFAULT_USABILITY_ACTION_ON_UNKNOWN``.

A target passes only if **some** ``phase_<planet>`` column and **some single** ``science config``
(among counted modes) jointly satisfy **all** of the following for that ``(planet, config)`` pair:

- **Pre-transit band** (default phase **-48 h to -3 h** inclusive): usable rows in that band and
  config have combined ``EXPTIME`` ≥ **1000 s** (from FITS ``key science files`` as elsewhere).
- **Transit-near band** (default **-4 h to +7 h** inclusive): usable rows in that band and **same**
  config have combined ``EXPTIME`` ≥ **3000 s** per **HST visit**. Visit is the two-character ``ss``
  field (indices 4–5) of the 9-character ``archive id`` (``ipppssoot``). The **> 3 h** ``start``
  span is computed only among rows sharing that visit (not across visits days or years apart).
- **Pre-transit pairing:** pre rows must have ``start`` no later than the **latest** ``start`` among
  the paired visit's transit-near rows. Latest pre and earliest transit ``start`` may overlap in time
  (negative ``tr_min - pre_max``) when phase bands overlap; that is allowed. When latest pre is
  **before** earliest transit, ``tr_min - pre_max`` must be at most
  ``max_pre_to_transit_start_gap_hours`` (default **48 h**). Pre-transit may come from **different**
  visits than the transit-near data.

- ``phase_<planet>_err`` is finite, **strictly positive**, and **not above 3 h** (when converted
  to hours) on every usable transit-near row (mode-filtered); otherwise the target is warned and
  does not pass.

Rows must pass ``custom_usability_mask`` (same rules as archival external verification).

If there is no observation-table file on disk for a target folder, or the table has no
``phase_*`` columns, that hostname is skipped and collected; after all targets are processed,
``warnings.warn`` lists those hostnames (separate messages for missing file, missing columns, or
bad ``phase_*_err`` (zero/masked or larger than 3 h).

Use ``parse_observed_transit_file`` if the caller keeps candidate hostnames in the usual
two-line-header text format. ``refresh_observed_transit_list`` only returns the qualified
hostnames and emits warnings; use ``write_observed_transit_list_file`` to write the list file.

``debug_archival_transit_for_hostname`` prints why a single target passes or fails. From the
stage1 repo root: ``python target_selection_tools/log_archival_transits.py "GJ 486"``.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy import time
from astropy import units as u
from astropy.io import fits

import database_utilities as dbutils
import paths
from processing import observation_table as obt
from processing import preloads
from processing import target_lists
from processing.observation_table import UsabilityDefinition

# Keep aligned with ``processing.scripts.update_verified_external.external_data_usability``.
DEFAULT_ARCHIVAL_USABILITY = UsabilityDefinition(
    ignore_notes=[
        "bot note",
        "bot warning acq image flux",
        "identified target in acquisition",
        "Telemetry indicates that the intended exposures may not",
        "output lacks some information because",
        "Saturation of pixels in the second image",
        "Your ACQ appears to have succeeded",
        "typical of a successful",
        "bot warning science flux",
    ],
    fail_notes=[
        "no acquisition found",
        "COS PEAKD counts zero at all dwell points",
        "bot warning COS PEAKD max",
        "bot warning COS PEAKD slew",
        "COS PEAKXD counts were zero",
        "bot warning COS PEAKXD slew",
        "deemed target absent",
        "substantial wavelength discrepancy",
        "problem with your acquisition",
        "The flux in the third image of the ACQ is lower",
        "problems in the ACQ/PEAK",
        "inadequate for an accurate",
        "Some pixels in the confirmation image were saturated",
        "The ACQ/PEAK flux test failed",
        "maximum flux in the sequence occurred at one end",
    ],
    ignore_flags=[
        "acquisition untrustworthy",
        "flux anomalously low",
        "flux anomalously high",
        "flux negligible",
    ],
    fail_flags=[
        "header targname = wave",
        "fluxes all zero",
        "fluxes all nan or non-finite",
        "wavelengths inaccurate",
    ],
)

# Passed to ``ObsTable.custom_usability_mask(..., action_on_unknown=...)`` when the caller omits it.
DEFAULT_USABILITY_ACTION_ON_UNKNOWN: str = "error"

_VALID_USABILITY_ACTION_ON_UNKNOWN = frozenset({"ignore", "fail", "warn", "error"})

DEFAULT_OUTPUT_PATH = paths.requested / "observed_transit.txt"

# Used when writing ``observed_transit``-style text if ``output_header_lines`` is omitted.
DEFAULT_OUTPUT_HEADER_LINES = (
    "list name: observed_transit",
    "list description: systems with planets where Lya transit has already been observed, automatically identified using the  ",
)

# Sum of EXPTIME (s) from key science FITS for in-window, usable rows (per science config).
_MIN_ARCHIVAL_TRANSIT_EXPTIME_S = 3000.0
# Require max(start) - min(start) > this (h); ``2 * 1.5`` h between first and last exposure.
_MIN_ARCHIVAL_TRANSIT_START_SPAN_H = 3.0

# Inclusive phase window in hours relative to mid-transit (see observation-table ``phase_*``).
DEFAULT_PHASE_HOURS_MIN = -4.0
DEFAULT_PHASE_HOURS_MAX = 7.0

# Reject targets if any in-window usable row has phase uncertainty above this (hours).
_MAX_PHASE_ERR_H = 3.0

# Pre-transit phase band (hours); sum EXPTIME in this band (counted modes only) must reach this.
_MIN_PRE_TRANSIT_EXPTIME_S = 1000.0
DEFAULT_PRE_TRANSIT_PHASE_HOURS_MIN = -48.0
DEFAULT_PRE_TRANSIT_PHASE_HOURS_MAX = -3.0

# When latest pre ``start`` is before earliest transit ``start``, their separation must be at most
# this (hours). Overlap (latest pre after earliest transit) is allowed; see module docstring.
DEFAULT_MAX_PRE_TO_TRANSIT_START_GAP_H = 48.0


def parse_observed_transit_file(path: Path) -> tuple[str, str, list[str]]:
    """Read a two-line-header target list (e.g. ``observed_transit.txt``); return headers and body."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} must have a two-line header and at least one hostname line")
    h0, h1 = lines[0], lines[1]
    if not h0.startswith("list name:"):
        raise ValueError(f"{path} line 1 must start with 'list name:', got {h0!r}")
    if not h1.startswith("list description:"):
        raise ValueError(f"{path} line 2 must start with 'list description:', got {h1!r}")
    body = [ln.strip() for ln in lines[2:]]
    if any(ln == "" for ln in body):
        raise ValueError(f"{path} has empty hostname lines in the body")
    if len(body) != len(set(body)):
        raise ValueError(f"{path} contains duplicate hostnames in the body")
    return h0, h1, body


def write_observed_transit_list_file(
    hostnames: Sequence[str],
    *,
    path: Path | str | None = None,
    output_header_lines: tuple[str, str] | None = None,
) -> Path:
    """
    Write an ``observed_transit``-style text file: two header lines, then one hostname per line.

    Parameters
    ----------
    hostnames
        Body lines (typically the return value of ``refresh_observed_transit_list``).
    path
        Output file path. If omitted, uses ``DEFAULT_OUTPUT_PATH``
        (``paths.requested / "observed_transit.txt"``).
    output_header_lines
        ``(list name: …, list description: …)``. If omitted, uses ``DEFAULT_OUTPUT_HEADER_LINES``.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    out_path = Path(DEFAULT_OUTPUT_PATH if path is None else path)
    header_name, header_desc = (
        output_header_lines
        if output_header_lines is not None
        else DEFAULT_OUTPUT_HEADER_LINES
    )
    if len(header_name.strip()) == 0 or len(header_desc.strip()) == 0:
        raise ValueError("output_header_lines must be two non-empty strings")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = [str(h).strip() for h in hostnames]
    if any(x == "" for x in body):
        raise ValueError("hostnames to write must not contain empty strings")
    lines_out = [header_name, header_desc, *body]
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return out_path


def _validate_target_hostnames(names: Sequence[str]) -> list[str]:
    out = [str(n).strip() for n in names]
    if any(x == "" for x in out):
        raise ValueError("target_hostnames contains an empty entry")
    if len(out) != len(set(out)):
        raise ValueError("target_hostnames contains duplicates")
    return out


def _phase_value_columns(obs_tbl: obt.ObsTable) -> list[str]:
    return [
        c
        for c in obs_tbl.colnames
        if c.startswith("phase_") and not c.endswith("_err")
    ]


def _phase_values_hours(col) -> np.ndarray:
    """Return float array of phase in hours, same length as ``col``."""
    q = getattr(col, "quantity", None)
    if q is not None:
        return np.asarray(q.to(u.h).value, dtype=float)
    unit = getattr(col, "unit", None)
    if unit is not None:
        return np.asarray(u.Quantity(col, unit).to(u.h).value, dtype=float)
    return np.asarray(col, dtype=float)


def _custom_pass_mask(
    obs_tbl: obt.ObsTable,
    *,
    usability_definition: UsabilityDefinition | None = None,
    action_on_unknown: str | None = None,
) -> np.ndarray:
    ud = DEFAULT_ARCHIVAL_USABILITY if usability_definition is None else usability_definition
    ao = DEFAULT_USABILITY_ACTION_ON_UNKNOWN if action_on_unknown is None else action_on_unknown
    if ao not in _VALID_USABILITY_ACTION_ON_UNKNOWN:
        raise ValueError(
            "action_on_unknown must be one of "
            f"{sorted(_VALID_USABILITY_ACTION_ON_UNKNOWN)}, got {ao!r}"
        )
    mask, _uncat = obs_tbl.custom_usability_mask(
        ud,
        action_on_unknown=ao,
        retain_unusable_flags=True,
    )
    return np.asarray(mask, dtype=bool)


def _normalize_science_config_markers(markers: Sequence[str] | None) -> tuple[str, ...] | None:
    """``None`` = all configs; else lowercased non-empty substrings to match against ``science config``."""
    if markers is None:
        return None
    out = tuple(str(m).strip().lower() for m in markers if str(m).strip())
    if not out:
        raise ValueError(
            "science_config_substrings must be None (all configs) or a non-empty sequence of substrings"
        )
    return out


def _mode_mask_for_obs_table(
    obs_tbl: obt.ObsTable, markers: tuple[str, ...] | None
) -> np.ndarray:
    """Per-row mask: True where ``science config`` matches ``markers`` (or all rows if ``markers`` is None)."""
    n = len(obs_tbl)
    if markers is None:
        return np.ones(n, dtype=bool)
    cfg = obs_tbl["science config"].filled("").astype(str)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        c = str(cfg[i]).lower()
        out[i] = any(m in c for m in markers)
    return out


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


def _visit_pair_from_archive_id(archive_id: str) -> str:
    """
    Two-character visit field from the HST 9-character ``archive id`` (``ipppssoot``): ``ss`` at
    indices 4–5 (0-based).
    """
    s = str(archive_id).strip().lower()
    if len(s) != 9:
        raise ValueError(
            f"archive id must be exactly 9 characters (ipppssoot) for visit grouping, got {len(s)}: {archive_id!r}"
        )
    return s[4:6]


def _group_indices_by_visit_ss(obs_tbl: obt.ObsTable, indices: np.ndarray) -> dict[str, list[int]]:
    """Map visit key (``ss``) -> row indices (subset of ``indices``)."""
    aid_col = obs_tbl["archive id"].filled("").astype(str)
    out: dict[str, list[int]] = {}
    for i in indices:
        ii = int(i)
        vk = _visit_pair_from_archive_id(str(aid_col[ii]))
        out.setdefault(vk, []).append(ii)
    return out


def _phase_in_window_mask(
    obs_tbl: obt.ObsTable,
    phase_col: str,
    *,
    phase_hours_min: float,
    phase_hours_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(row_ok_phase, in_window)`` boolean arrays (length n)."""
    col = obs_tbl[phase_col]
    vals = _phase_values_hours(col)
    m = getattr(col, "mask", None)
    if m is None:
        row_ok = np.ones(len(obs_tbl), dtype=bool)
    else:
        row_ok = ~np.asarray(m, dtype=bool)
    finite = np.isfinite(vals)
    in_window = (vals >= float(phase_hours_min)) & (vals <= float(phase_hours_max))
    return row_ok & finite, in_window


def _phase_err_cell_bad(err_col, i: int) -> bool:
    m = getattr(err_col, "mask", None)
    if m is not None:
        ma = np.asarray(m, dtype=bool)
        if ma.shape == ():
            if bool(ma):
                return True
        elif ma.reshape(-1)[i]:
            return True
    v = err_col[i]
    if v is None or obt.ObsTable._is_null_like(v):
        return True
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(fv) or fv == 0.0:
        return True
    return False


def _phase_err_uncertainty_hours(err_col, i: int) -> float:
    """Phase uncertainty at row ``i`` in hours (column unit if present, else minutes per pipeline)."""
    v = err_col[i]
    fv = float(v)
    unit = getattr(err_col, "unit", None)
    if unit is not None:
        return float(u.Quantity(fv, unit).to(u.h).value)
    return fv / 60.0


def _target_phase_err_failure_modes(
    obs_tbl: obt.ObsTable,
    pass_mask: np.ndarray,
    mode_mask: np.ndarray,
    *,
    phase_hours_min: float,
    phase_hours_max: float,
    max_phase_err_h: float = _MAX_PHASE_ERR_H,
) -> tuple[bool, bool]:
    """
    Return ``(zero_or_masked_or_invalid, uncertainty_over_limit)`` for usable in-window rows.

    Either flag causes the target to be rejected from the output list.
    """
    zero_or_masked = False
    too_large = False
    for phase_col in _phase_value_columns(obs_tbl):
        err_name = f"{phase_col}_err"
        if err_name not in obs_tbl.colnames:
            raise ValueError(
                f"Observation table has {phase_col!r} but no matching {err_name!r} column"
            )
        row_ok_phase, in_window = _phase_in_window_mask(
            obs_tbl,
            phase_col,
            phase_hours_min=phase_hours_min,
            phase_hours_max=phase_hours_max,
        )
        check = pass_mask & row_ok_phase & in_window & mode_mask
        ec = obs_tbl[err_name]
        for i in np.nonzero(check)[0]:
            ii = int(i)
            if _phase_err_cell_bad(ec, ii):
                zero_or_masked = True
            elif _phase_err_uncertainty_hours(ec, ii) > float(max_phase_err_h):
                too_large = True
    return zero_or_masked, too_large


def _target_has_archival_transit(
    obs_tbl: obt.ObsTable,
    *,
    data_dir: Path,
    pass_mask: np.ndarray,
    mode_mask: np.ndarray,
    phase_hours_min: float,
    phase_hours_max: float,
    pre_transit_phase_hours_min: float,
    pre_transit_phase_hours_max: float,
    max_pre_to_transit_start_gap_h: float,
) -> bool:
    """
    True if some ``(phase_<planet>, science config)`` has linked pre-transit + transit-near data
    (same config, pre rows not starting after the visit's last transit ``start``, optional gap cap
    when latest pre precedes earliest transit, exptime and span thresholds).
    """
    phase_cols = _phase_value_columns(obs_tbl)
    if not phase_cols:
        return False
    cfg_col = obs_tbl["science config"].filled("").astype(str)
    n = len(obs_tbl)
    cfg_ns = np.array([str(cfg_col[i]).strip() for i in range(n)], dtype=object)

    for phase_col in phase_cols:
        err_name = f"{phase_col}_err"
        if err_name not in obs_tbl.colnames:
            raise ValueError(
                f"Observation table has {phase_col!r} but no matching {err_name!r} column"
            )

        row_ok_pre, in_pre = _phase_in_window_mask(
            obs_tbl,
            phase_col,
            phase_hours_min=pre_transit_phase_hours_min,
            phase_hours_max=pre_transit_phase_hours_max,
        )
        base_pre = pass_mask & row_ok_pre & in_pre & mode_mask

        row_ok_tr, in_transit = _phase_in_window_mask(
            obs_tbl,
            phase_col,
            phase_hours_min=phase_hours_min,
            phase_hours_max=phase_hours_max,
        )
        base_tr = pass_mask & row_ok_tr & in_transit & mode_mask

        pre_cfgs = {c for c in np.unique(cfg_ns[base_pre]) if c}
        tr_cfgs = {c for c in np.unique(cfg_ns[base_tr]) if c}

        for cfg in pre_cfgs & tr_cfgs:
            pre_all_idx = np.nonzero(base_pre & (cfg_ns == cfg))[0]
            tr_all_idx = np.nonzero(base_tr & (cfg_ns == cfg))[0]
            if pre_all_idx.size == 0 or tr_all_idx.size == 0:
                continue

            tr_by_visit = _group_indices_by_visit_ss(obs_tbl, tr_all_idx)

            for _visit_key, tr_list in tr_by_visit.items():
                tr_idx = np.array(tr_list, dtype=int)
                if tr_idx.size == 0:
                    continue

                t_tr = 0.0
                for i in tr_idx:
                    t_tr += _row_exptime_seconds(obs_tbl, int(i), data_dir)
                if t_tr < _MIN_ARCHIVAL_TRANSIT_EXPTIME_S:
                    continue

                starts_tr = time.Time(obs_tbl["start"][tr_idx])
                span_h = (starts_tr.max() - starts_tr.min()).to(u.h).value
                if span_h <= _MIN_ARCHIVAL_TRANSIT_START_SPAN_H:
                    continue

                t_tr_min = starts_tr.min()
                t_tr_max = starts_tr.max()

                st_pre_cand = time.Time(obs_tbl["start"][pre_all_idx])
                time_ok = st_pre_cand <= t_tr_max
                pre_idx = pre_all_idx[np.asarray(time_ok, dtype=bool)]
                if pre_idx.size == 0:
                    continue

                t_pre = 0.0
                for i in pre_idx:
                    t_pre += _row_exptime_seconds(obs_tbl, int(i), data_dir)
                if t_pre < _MIN_PRE_TRANSIT_EXPTIME_S:
                    continue

                starts_pre = time.Time(obs_tbl["start"][pre_idx])
                t_pre_max = starts_pre.max()
                gap_h = (t_tr_min - t_pre_max).to(u.h).value
                if gap_h > float(max_pre_to_transit_start_gap_h):
                    continue

                return True
    return False


def refresh_observed_transit_list(
    *,
    target_hostnames: Sequence[str],
    science_config_substrings: Sequence[str] | None = None,
    phase_hours_min: float = DEFAULT_PHASE_HOURS_MIN,
    phase_hours_max: float = DEFAULT_PHASE_HOURS_MAX,
    pre_transit_phase_hours_min: float = DEFAULT_PRE_TRANSIT_PHASE_HOURS_MIN,
    pre_transit_phase_hours_max: float = DEFAULT_PRE_TRANSIT_PHASE_HOURS_MAX,
    max_pre_to_transit_start_gap_hours: float = DEFAULT_MAX_PRE_TO_TRANSIT_START_GAP_H,
    usability_definition: UsabilityDefinition | None = None,
    action_on_unknown: str | None = None,
) -> list[str]:
    """
    Re-evaluate the given hostnames; return those that qualify and emit any warning messages.

    Does not write to disk; use ``write_observed_transit_list_file`` to save results.

    To build ``target_hostnames`` from disk, use ``parse_observed_transit_file(path)`` and pass
    the returned body.

    Parameters
    ----------
    target_hostnames
        ``stela_names`` ``hostname`` values to check (must match exactly, no duplicates).
    science_config_substrings
        If ``None``, all ``science config`` rows are counted. Otherwise only rows whose config
        string contains at least one of these substrings (case-insensitive).
    phase_hours_min, phase_hours_max
        Inclusive transit-near phase window in hours relative to mid-transit (defaults -4 and +7).
    pre_transit_phase_hours_min, pre_transit_phase_hours_max
        Inclusive pre-transit phase window for the 1000 s exposure sum (defaults -48 and -3).
    max_pre_to_transit_start_gap_hours
        When latest pre ``start`` is before earliest transit ``start``, maximum allowed separation
        ``earliest transit start - latest pre start`` (hours); default 48. Overlap (negative
        separation) is allowed.
    usability_definition
        ``ObsTable.custom_usability_mask`` fail/ignore lists. If ``None``, uses
        ``DEFAULT_ARCHIVAL_USABILITY``.
    action_on_unknown
        How to treat flag/note substrings not in those lists: ``"ignore"``, ``"fail"``, ``"warn"``,
        or ``"error"`` (see ``ObsTable.custom_usability_mask``). If ``None``, uses
        ``DEFAULT_USABILITY_ACTION_ON_UNKNOWN``.

    Returns
    -------
    list of str
        Hostnames that qualify, in the same order as ``target_hostnames``.

    Raises
    ------
    KeyError
        If a hostname is missing from ``stela_names.csv``.
    """
    hostnames = _validate_target_hostnames(target_hostnames)

    if float(phase_hours_min) > float(phase_hours_max):
        raise ValueError(
            f"phase_hours_min ({phase_hours_min}) must be <= phase_hours_max ({phase_hours_max})"
        )

    if float(pre_transit_phase_hours_min) > float(pre_transit_phase_hours_max):
        raise ValueError(
            f"pre_transit_phase_hours_min ({pre_transit_phase_hours_min}) must be <= "
            f"pre_transit_phase_hours_max ({pre_transit_phase_hours_max})"
        )

    if float(max_pre_to_transit_start_gap_hours) < 0:
        raise ValueError("max_pre_to_transit_start_gap_hours must be >= 0")

    if action_on_unknown is not None and action_on_unknown not in _VALID_USABILITY_ACTION_ON_UNKNOWN:
        raise ValueError(
            "action_on_unknown must be one of "
            f"{sorted(_VALID_USABILITY_ACTION_ON_UNKNOWN)}, got {action_on_unknown!r}"
        )

    config_markers = _normalize_science_config_markers(science_config_substrings)

    sn = preloads.stela_names
    candidates = frozenset(hostnames)

    qualified_set: set[str] = set()
    missing_obs_tbl: list[str] = []
    missing_phase: list[str] = []
    bad_phase_err_zero_masked: list[str] = []
    bad_phase_err_too_large: list[str] = []
    seen_from_list: set[str] = set()

    for hostfile in target_lists.everything_in_database():
        row = sn.loc["hostname_file", hostfile]
        hostname = str(row["hostname"]).strip()
        if not hostname:
            raise ValueError(f"stela_names hostname_file={hostfile!r} has empty hostname")
        if hostname not in candidates:
            continue
        seen_from_list.add(hostname)

        obs_path = obt.ObsTable.get_path(hostfile)
        if not obs_path.is_file():
            missing_obs_tbl.append(hostname)
            continue

        obs_tbl = obt.ObsTable.read(obs_path)
        if len(obs_tbl) == 0:
            raise ValueError(f"Observation table for {hostname!r} is empty ({obs_path})")

        if not _phase_value_columns(obs_tbl):
            missing_phase.append(hostname)
            continue

        data_dir = paths.target_hst_data(hostfile)
        pass_mask = _custom_pass_mask(
            obs_tbl,
            usability_definition=usability_definition,
            action_on_unknown=action_on_unknown,
        )
        mode_mask = _mode_mask_for_obs_table(obs_tbl, config_markers)
        zm, lg = _target_phase_err_failure_modes(
            obs_tbl,
            pass_mask,
            mode_mask,
            phase_hours_min=phase_hours_min,
            phase_hours_max=phase_hours_max,
        )
        if zm:
            bad_phase_err_zero_masked.append(hostname)
        if lg:
            bad_phase_err_too_large.append(hostname)
        if zm or lg:
            continue

        if _target_has_archival_transit(
            obs_tbl,
            data_dir=data_dir,
            pass_mask=pass_mask,
            mode_mask=mode_mask,
            phase_hours_min=phase_hours_min,
            phase_hours_max=phase_hours_max,
            pre_transit_phase_hours_min=pre_transit_phase_hours_min,
            pre_transit_phase_hours_max=pre_transit_phase_hours_max,
            max_pre_to_transit_start_gap_h=max_pre_to_transit_start_gap_hours,
        ):
            qualified_set.add(hostname)

    missing_dirs = candidates - seen_from_list
    if missing_dirs:
        raise ValueError(
            "target_hostnames includes names with no target folder under the STELa data root "
            f"(see target_lists.everything_in_database): {sorted(missing_dirs)}"
        )

    qualified = [h for h in hostnames if h in qualified_set]

    if missing_obs_tbl:
        warnings.warn(
            "No observation-table file on disk for: " + ", ".join(missing_obs_tbl),
            stacklevel=2,
        )

    if missing_phase:
        warnings.warn(
            "Observation tables had no phase_* columns for: "
            + ", ".join(missing_phase),
            stacklevel=2,
        )

    if bad_phase_err_zero_masked:
        warnings.warn(
            "Phase uncertainty zero, masked, or invalid for at least one in-window usable exposure "
            "(see phase_*_err columns) for: "
            + ", ".join(bad_phase_err_zero_masked),
            stacklevel=2,
        )

    if bad_phase_err_too_large:
        warnings.warn(
            f"Phase uncertainty exceeds {_MAX_PHASE_ERR_H:g} h for at least one in-window usable "
            "exposure (see phase_*_err columns) for: "
            + ", ".join(bad_phase_err_too_large),
            stacklevel=2,
        )

    return qualified


def debug_archival_transit_for_hostname(
    hostname: str,
    *,
    science_config_substrings: Sequence[str] | None = None,
    phase_hours_min: float = DEFAULT_PHASE_HOURS_MIN,
    phase_hours_max: float = DEFAULT_PHASE_HOURS_MAX,
    pre_transit_phase_hours_min: float = DEFAULT_PRE_TRANSIT_PHASE_HOURS_MIN,
    pre_transit_phase_hours_max: float = DEFAULT_PRE_TRANSIT_PHASE_HOURS_MAX,
    max_pre_to_transit_start_gap_hours: float = DEFAULT_MAX_PRE_TO_TRANSIT_START_GAP_H,
    usability_definition: UsabilityDefinition | None = None,
    action_on_unknown: str | None = None,
) -> bool:
    """
    Print step-by-step diagnostics for one ``stela_names`` hostname (e.g. ``\"GJ 486\"``).

    Uses the same criteria as ``refresh_observed_transit_list`` (including ``usability_definition``
    and ``action_on_unknown``). Returns whether the target would qualify (``True``/``False``).
    Raises if the hostname is unknown, the observation table is missing, empty, or lacks
    ``phase_*`` columns.
    """
    hn = str(hostname).strip()
    if not hn:
        raise ValueError("hostname must be non-empty")

    print(f"=== debug archival transit: {hn!r} ===")

    sn = preloads.stela_names
    try:
        row = sn.loc["hostname", hn]
    except KeyError as e:
        raise KeyError(f"No stela_names.csv row for hostname={hn!r}") from e

    hostfile = str(row["hostname_file"]).strip()
    if not hostfile:
        raise ValueError(f"stela_names hostname={hn!r} has empty hostname_file")

    data_dir = paths.target_hst_data(hostfile)
    obs_path = obt.ObsTable.get_path(hostfile)
    print(f"hostname_file: {hostfile!r}")
    print(f"data_dir: {obs_path.parent}")
    print(f"observation table: {obs_path}")

    if not obs_path.is_file():
        raise FileNotFoundError(f"No observation table at {obs_path}")

    obs_tbl = obt.ObsTable.read(obs_path)
    if len(obs_tbl) == 0:
        raise ValueError(f"Observation table is empty: {obs_path}")

    phase_cols = _phase_value_columns(obs_tbl)
    if not phase_cols:
        raise ValueError(f"No phase_* columns in {obs_path}")

    print(f"rows: {len(obs_tbl)}  phase columns: {phase_cols}")

    if float(phase_hours_min) > float(phase_hours_max):
        raise ValueError("phase_hours_min must be <= phase_hours_max")
    if float(pre_transit_phase_hours_min) > float(pre_transit_phase_hours_max):
        raise ValueError("pre_transit phase min must be <= pre_transit phase max")
    if float(max_pre_to_transit_start_gap_hours) < 0:
        raise ValueError("max_pre_to_transit_start_gap_hours must be >= 0")
    if action_on_unknown is not None and action_on_unknown not in _VALID_USABILITY_ACTION_ON_UNKNOWN:
        raise ValueError(
            "action_on_unknown must be one of "
            f"{sorted(_VALID_USABILITY_ACTION_ON_UNKNOWN)}, got {action_on_unknown!r}"
        )

    config_markers = _normalize_science_config_markers(science_config_substrings)
    print(f"science_config_substrings -> {config_markers!r}")
    print(
        f"transit phase window (h): [{phase_hours_min}, {phase_hours_max}]  "
        f"pre-transit: [{pre_transit_phase_hours_min}, {pre_transit_phase_hours_max}]  "
        f"max pre->transit start gap (h): {max_pre_to_transit_start_gap_hours}"
    )
    _ao = (
        DEFAULT_USABILITY_ACTION_ON_UNKNOWN
        if action_on_unknown is None
        else action_on_unknown
    )
    print(
        "custom_usability_mask: "
        f"usability_definition={'DEFAULT_ARCHIVAL_USABILITY' if usability_definition is None else 'custom'}  "
        f"action_on_unknown={_ao!r}"
    )

    pass_mask = _custom_pass_mask(
        obs_tbl,
        usability_definition=usability_definition,
        action_on_unknown=action_on_unknown,
    )
    mode_mask = _mode_mask_for_obs_table(obs_tbl, config_markers)
    n_pass = int(np.count_nonzero(pass_mask))
    n_mode = int(np.count_nonzero(mode_mask))
    print(f"custom_usability pass rows: {n_pass}/{len(obs_tbl)}  mode_mask True rows: {n_mode}/{len(obs_tbl)}")

    zm, lg = _target_phase_err_failure_modes(
        obs_tbl,
        pass_mask,
        mode_mask,
        phase_hours_min=phase_hours_min,
        phase_hours_max=phase_hours_max,
    )
    print(f"phase_err (transit window): zero_or_masked={zm}  too_large={lg}")
    if zm or lg:
        for phase_col in phase_cols:
            err_name = f"{phase_col}_err"
            row_ok_tr, in_tr = _phase_in_window_mask(
                obs_tbl,
                phase_col,
                phase_hours_min=phase_hours_min,
                phase_hours_max=phase_hours_max,
            )
            check = pass_mask & row_ok_tr & in_tr & mode_mask
            ec = obs_tbl[err_name]
            for i in np.nonzero(check)[0]:
                ii = int(i)
                if _phase_err_cell_bad(ec, ii):
                    print(f"  row {ii} {phase_col}: err zero/masked/invalid")
                elif _phase_err_uncertainty_hours(ec, ii) > float(_MAX_PHASE_ERR_H):
                    print(
                        f"  row {ii} {phase_col}: err {_phase_err_uncertainty_hours(ec, ii):.4g} h "
                        f"(>{_MAX_PHASE_ERR_H:g} h)"
                    )

    if zm or lg:
        print("RESULT: would NOT qualify (phase uncertainty checks)")
        return False

    cfg_col = obs_tbl["science config"].filled("").astype(str)
    n = len(obs_tbl)
    cfg_ns = np.array([str(cfg_col[i]).strip() for i in range(n)], dtype=object)

    would_pass = False
    for phase_col in phase_cols:
        print(f"--- planet column {phase_col!r} ---")

        row_ok_pre, in_pre = _phase_in_window_mask(
            obs_tbl,
            phase_col,
            phase_hours_min=pre_transit_phase_hours_min,
            phase_hours_max=pre_transit_phase_hours_max,
        )
        base_pre = pass_mask & row_ok_pre & in_pre & mode_mask

        row_ok_tr, in_transit = _phase_in_window_mask(
            obs_tbl,
            phase_col,
            phase_hours_min=phase_hours_min,
            phase_hours_max=phase_hours_max,
        )
        base_tr = pass_mask & row_ok_tr & in_transit & mode_mask

        pre_cfgs = {c for c in np.unique(cfg_ns[base_pre]) if c}
        tr_cfgs = {c for c in np.unique(cfg_ns[base_tr]) if c}
        print(f"  configs with pre-band rows: {sorted(pre_cfgs)}")
        print(f"  configs with transit-band rows: {sorted(tr_cfgs)}")
        shared = pre_cfgs & tr_cfgs
        print(f"  intersection: {sorted(shared)}")
        if not shared:
            print("  (no shared science config between pre and transit bands)")
            continue

        for cfg in sorted(shared):
            print(f"  .. config {cfg!r}")
            pre_all_idx = np.nonzero(base_pre & (cfg_ns == cfg))[0]
            tr_all_idx = np.nonzero(base_tr & (cfg_ns == cfg))[0]
            print(
                f"     candidate pre rows (phase band): {pre_all_idx.size}  "
                f"transit-band rows: {tr_all_idx.size}"
            )
            if pre_all_idx.size == 0 or tr_all_idx.size == 0:
                continue

            tr_by_visit = _group_indices_by_visit_ss(obs_tbl, tr_all_idx)
            for visit_ss in sorted(tr_by_visit.keys()):
                tr_idx = np.array(tr_by_visit[visit_ss], dtype=int)
                print(f"     .... visit ss={visit_ss!r}  transit rows in visit: {tr_idx.size}")

                t_tr = 0.0
                tr_ex_fail: list[str] = []
                for i in tr_idx:
                    try:
                        t_tr += _row_exptime_seconds(obs_tbl, int(i), data_dir)
                    except Exception as ex:
                        tr_ex_fail.append(f"row {int(i)}: {ex}")
                print(
                    f"          transit EXPTIME sum this visit (s): {t_tr:.1f}  "
                    f"(need >= {_MIN_ARCHIVAL_TRANSIT_EXPTIME_S:g})"
                )
                for msg in tr_ex_fail[:3]:
                    print(f"          EXPTIME error: {msg}")
                if len(tr_ex_fail) > 3:
                    print(f"          ... {len(tr_ex_fail) - 3} more")
                if t_tr < _MIN_ARCHIVAL_TRANSIT_EXPTIME_S:
                    print("          FAIL: transit exptime below threshold (this visit)")
                    continue

                starts_tr = time.Time(obs_tbl["start"][tr_idx])
                span_h = (starts_tr.max() - starts_tr.min()).to(u.h).value
                print(
                    f"          transit start span within visit (h): {span_h:.4g}  "
                    f"(need > {_MIN_ARCHIVAL_TRANSIT_START_SPAN_H:g})"
                )
                if span_h <= _MIN_ARCHIVAL_TRANSIT_START_SPAN_H:
                    print("          FAIL: transit start span too small (this visit)")
                    continue

                t_tr_min = starts_tr.min()
                t_tr_max = starts_tr.max()
                print(f"          transit visit start min: {t_tr_min.isot}  max: {t_tr_max.isot}")

                st_pre_cand = time.Time(obs_tbl["start"][pre_all_idx])
                time_ok = np.asarray(st_pre_cand <= t_tr_max, dtype=bool)
                pre_idx = pre_all_idx[time_ok]
                n_dropped = int(pre_all_idx.size - pre_idx.size)
                if n_dropped:
                    print(
                        f"          pre rows after start <= last transit start: {pre_idx.size}  "
                        f"(dropped {n_dropped} with start > transit visit last start)"
                    )
                else:
                    print(
                        f"          pre rows after start <= last transit start: {pre_idx.size}"
                    )
                if pre_idx.size == 0:
                    print("          FAIL: no pre rows left after time filter")
                    continue

                t_pre = 0.0
                pre_ex_fail: list[str] = []
                for i in pre_idx:
                    try:
                        t_pre += _row_exptime_seconds(obs_tbl, int(i), data_dir)
                    except Exception as ex:
                        pre_ex_fail.append(f"row {int(i)}: {ex}")
                print(
                    f"          pre EXPTIME sum (s): {t_pre:.1f}  "
                    f"(need >= {_MIN_PRE_TRANSIT_EXPTIME_S:g})"
                )
                for msg in pre_ex_fail[:3]:
                    print(f"          EXPTIME error: {msg}")
                if len(pre_ex_fail) > 3:
                    print(f"          ... {len(pre_ex_fail) - 3} more")
                if t_pre < _MIN_PRE_TRANSIT_EXPTIME_S:
                    print("          FAIL: pre exptime below threshold")
                    continue

                starts_pre = time.Time(obs_tbl["start"][pre_idx])
                t_pre_max = starts_pre.max()
                gap_h = (t_tr_min - t_pre_max).to(u.h).value
                print(f"          latest pre start (after filter): {t_pre_max.isot}")
                print(f"          earliest transit start (visit): {t_tr_min.isot}")
                print(
                    f"          gap (tr_min - pre_max) (h): {gap_h:.4g}  "
                    f"(if >= 0, must be <= {max_pre_to_transit_start_gap_hours:g}; overlap/negative OK)"
                )
                if gap_h > float(max_pre_to_transit_start_gap_hours):
                    print("          FAIL: pre->transit start gap too large")
                    continue

                print("          PASS for this (planet, config, transit visit)")
                would_pass = True

    if would_pass:
        print("RESULT: would QUALIFY")
    else:
        print(
            "RESULT: would NOT qualify "
            "(no (planet, config, transit visit) satisfied all linked checks)"
        )
    return would_pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python target_selection_tools/log_archival_transits.py <hostname> [<hostname> ...]",
            file=sys.stderr,
        )
        sys.exit(2)
    for arg in sys.argv[1:]:
        debug_archival_transit_for_hostname(arg)
        print()
