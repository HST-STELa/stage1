from __future__ import annotations

import numpy as np


#%% utilities

def row_value_weighted_centroids(data: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    """
    For each row, the centroid along columns (0 .. n_col-1), weighted by the
    row's `data` values. Columns where `ignore_mask` is True are excluded
    (zero weight). Rows with no usable weight get NaN.

    Parameters
    ----------
    data
        2D array, shape (n_row, n_col).
    ignore_mask
        2D boolean array, same shape as `data`. True = ignore that pixel.

    Returns
    -------
    np.ndarray
        1D array of length n_row, centroid column index per row.
    """
    data = np.asarray(data, dtype=np.float64)
    ignore_mask = np.asarray(ignore_mask, dtype=bool)
    if data.shape != ignore_mask.shape or data.ndim != 2:
        raise ValueError("data and ignore_mask must be 2D and the same shape")
    n_col = data.shape[1]
    w = data.copy()
    w[ignore_mask] = 0.0
    x = np.arange(n_col, dtype=np.float64)
    sum_w = w.sum(axis=1)
    sum_xw = (w * x).sum(axis=1)
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    valid = sum_w != 0
    out[valid] = sum_xw[valid] / sum_w[valid]
    return out


# Median absolute deviation to the normal standard deviation for Gaussian data
_MAD_TO_SIGMA = 1.4826


def fit_line_reject_mad(
    x: np.ndarray,
    y: np.ndarray,
    n_sigma: float = 3.0,
    max_iter: int = 20,
) -> tuple[float, float, np.ndarray]:
    """
    Least-squares line y = intercept + slope * x, with iterative outlier removal.

    After each fit, the robust scale is ``sigma = _MAD_TO_SIGMA * median(|r|)`` where
    ``r`` is the residual to the line at inlier points. Points with
    ``|r| > n_sigma * sigma`` are removed and the fit is repeated.

    Parameters
    ----------
    x, y
        Same-length 1D arrays (raveled if higher-dimensional).
    n_sigma
        Rejection threshold in units of the MAD-based scale.
    max_iter
        Safety cap on iterations.

    Returns
    -------
    slope, intercept, inlier_mask
        Final straight-line parameters and a boolean mask (same length as x, y) for
        the points used in the last fit.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("need at least 2 finite (x, y) pairs")
    good = finite.copy()
    a = b = np.nan
    for _ in range(max_iter):
        if good.sum() < 2:
            break
        xs, ys = x[good], y[good]
        coef = np.polyfit(xs, ys, 1)
        a, b = float(coef[0]), float(coef[1])
        r = y - (a * x + b)
        abs_r = np.abs(r[good])
        med_abs = float(np.median(abs_r))
        if med_abs == 0.0 or not np.isfinite(med_abs):
            break
        sigma = _MAD_TO_SIGMA * med_abs
        limit = n_sigma * sigma
        new_good = good & (np.abs(r) <= limit)
        if new_good.sum() < 2:
            break
        if np.array_equal(new_good, good):
            break
        good = new_good
    return a, b, good


def sum_within_centroid_line_strip(
    image: np.ndarray,
    ignore_mask: np.ndarray,
    half_width_pixels: float,
) -> float:
    """
    Fit row centroids (column as a function of row), robust line ``col = a*row+b``,
    then sum ``image`` over columns with ``|j - (a*row+b)| <= half_width_pixels``.
    """
    data = np.asarray(image, dtype=np.float64)
    m = np.asarray(ignore_mask, dtype=bool)
    if data.shape != m.shape or data.ndim != 2:
        raise ValueError("image and ignore_mask must be 2D and match")
    nrow, ncol = data.shape
    rows = np.arange(nrow, dtype=np.float64)
    centroids = row_value_weighted_centroids(data, m)
    finite = np.isfinite(centroids)
    if finite.sum() < 2:
        return float("nan")
    a, b, inlier = fit_line_reject_mad(rows[finite], centroids[finite])
    if inlier.sum() < 2 or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    total = 0.0
    hw = float(half_width_pixels)
    for i in range(nrow):
        c0 = a * i + b
        j_lo = int(np.floor(c0 - hw))
        j_hi = int(np.ceil(c0 + hw))
        j0 = max(0, j_lo)
        j1 = min(ncol, j_hi + 1)
        if j0 >= j1:
            continue
        row = data[i, j0:j1].copy()
        row[m[i, j0:j1]] = 0.0
        total += float(np.sum(row))
    return total


def scale_background_template(
    image: np.ndarray,
    background: np.ndarray,
    line_strip_width_pixels: float,
    ignore_mask: np.ndarray | None = None,
    ignore_mask_bg: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Match a background template to an image by trace-normalized scaling.

    For each array, centroids are computed per row (ignoring masked pixels), a line
    is fit with :func:`fit_line_reject_mad`, and pixel values are summed in a
    column band of full width ``line_strip_width_pixels`` around that line. The
    background is scaled by ``sum_image / sum_background`` and returned.
    By default, non-finite pixels are ignored in centroids; pass explicit boolean
    masks (True = ignore) to match other conditions.

    Parameters
    ----------
    image, background
        2D arrays, same shape.
    line_strip_width_pixels
        Full width in columns: at each row, pixels *j* with
        ``|j - (a*row + b)| <= line_strip_width_pixels / 2`` are included.
    ignore_mask, ignore_mask_bg
        If None, all non-finite values are ignored. Otherwise 2D bool, same shape
        as the images; True = ignore.
    """
    im = np.asarray(image, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if im.shape != bg.shape or im.ndim != 2:
        raise ValueError("image and background must be 2D and the same shape")
    half = float(line_strip_width_pixels) * 0.5
    if line_strip_width_pixels < 0 or not np.isfinite(half):
        raise ValueError("line_strip_width_pixels must be finite and non-negative")
    if ignore_mask is None:
        m_im = ~np.isfinite(im)
    else:
        m_im = np.asarray(ignore_mask, dtype=bool)
        if m_im.shape != im.shape:
            raise ValueError("ignore_mask must match image shape")
    if ignore_mask_bg is None:
        m_bg = ~np.isfinite(bg)
    else:
        m_bg = np.asarray(ignore_mask_bg, dtype=bool)
        if m_bg.shape != im.shape:
            raise ValueError("ignore_mask_bg must match image shape")
    s_im = sum_within_centroid_line_strip(im, m_im, half)
    s_bg = sum_within_centroid_line_strip(bg, m_bg, half)
    if not (np.isfinite(s_im) and np.isfinite(s_bg)) or s_bg == 0.0:
        raise ValueError("invalid strip sums for scaling (nan or background sum 0)")
    factor = s_im / s_bg
    return factor * bg, factor

