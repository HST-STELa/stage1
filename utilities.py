import warnings
from math import nan
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from mpld3 import plugins
import mpld3
from astropy.units import Quantity


def add_unit_to_output(input, output):
    if hasattr(input, 'unit') and input.unit not in [None, ''] and not hasattr(output, 'unit'):
        output *= input.unit
    return output


def midpts(ary, axis=None):
    """Computes the midpoints between points in a vector.

    Output has length len(vec)-1.
    """
    if type(ary) not in [np.ndarray, Quantity]:
        ary = np.array(ary)
    if axis == None:
        return (ary[1:] + ary[:-1])/2.0
    else:
        hi = np.split(ary, [1], axis=axis)[1]
        lo = np.split(ary, [-1], axis=axis)[0]
        return (hi+lo)/2.0


def mids2bins(mids, bin_widths=None):
    """
    Reconstructs bin edges given only the midpoints by assuming the edges fall evenly between the midpoints.
    Note that this assumption can be invalid, particularly if the midpoints are not spaced evenly.
    If bin_widths are provided, no assumption is needed.

    Parameters
    ----------
    mids : 1-D array-like
        A 1-D array or list of the midpoints from which bin edges are to be
        inferred.
    bin_widths : 1-D array-like, optional
        width of the bin associated with each midpoint

    Result
    ------
    edges : np.array
        The inferred bin edges.

    """
    if bin_widths is None:
        edges = midpts(mids)
        d0 = edges[0] - mids[0]
        d1 = mids[-1] - edges[-1]
        edges = np.insert(edges, [0, len(edges)], [mids[0] - d0, mids[-1] + d1])
    else:
        edges = np.append(mids - bin_widths/2, mids[-1] + bin_widths[-1]/2)

    edges = add_unit_to_output(mids, edges)

    return edges


def cumulative_trapz(y, x, zero_start=False):
    result = np.cumsum(midpts(y)*np.diff(x))
    if zero_start:
        result = np.insert(result, 0, 0)
    return result


def interpolate_many(x_new, x, y, axis=None, infer=True, assume_sorted=False, left=nan, right=nan):
    """
    Vectorized 1D interpolation along a specified axis of an N-D array y.
    - x_new:   array with last dim K (can broadcast across leading dims of y)
    - x: array with last dim M (can broadcast across leading dims of y)
    - y: N-D array; one axis (given or inferred) has length M
    Returns: y interpolated at x_new along that axis.
    """
    x = np.asarray(x)
    x_new   = np.asarray(x_new)
    y       = np.asarray(y)

    # Infer axis if needed
    if axis is None:
        if infer:
            matches = [i for i, s in enumerate(y.shape) if s == x.shape[-1]]
            if not matches:
                raise ValueError("Could not infer axis: no dimension of y matches x_known's last dimension.")
            axis = matches[0]
        else:
            raise ValueError("Specify 'axis' or set infer=True.")

    # Move interpolation axis to the end
    yL = np.moveaxis(y, axis, -1)                  # shape: B... x M
    B = yL.shape[:-1]
    M = yL.shape[-1]

    # Broadcast x_known and x_new to match leading dims
    if x.shape[-1] != M:
        raise ValueError("x_known's last dimension must equal the length of y along the interpolation axis.")
    xk = np.broadcast_to(x, B + (M,))        # B... x M
    K  = x_new.shape[-1]
    xn = np.broadcast_to(x_new,   B + (K,))        # B... x K

    # Flatten leading dims for clean vectorized gather
    N  = int(np.prod(B)) or 1
    y2 = yL.reshape(N, M)
    xk2 = xk.reshape(N, M)
    xn2 = xn.reshape(N, K)

    # Optionally sort x_known (and y) along the M axis
    if not assume_sorted:
        order = np.argsort(xk2, axis=1)
        xk2 = np.take_along_axis(xk2, order, axis=1)
        y2  = np.take_along_axis(y2,  order, axis=1)

    # Find bin indices for each new-x (vectorized). O(N*M*K) but fully NumPy.
    # idx points to the left node of the interval [idx, idx+1].
    idx = (xn2[..., None] >= xk2[:, None, :]).sum(axis=-1) - 1  # shape (N, K)
    idx = np.clip(idx, 0, M - 2)

    # Gather neighbors
    x0 = np.take_along_axis(xk2, idx, axis=1)
    x1 = np.take_along_axis(xk2, idx + 1, axis=1)
    y0 = np.take_along_axis(y2,  idx, axis=1)
    y1 = np.take_along_axis(y2,  idx + 1, axis=1)

    # Linear interpolation
    t = (xn2 - x0) / (x1 - x0)
    out = y0 + t * (y1 - y0)

    # Handle left/right
    mask_left = xn2 < xk2[:, [0]]
    out = np.where(mask_left, left, out)
    mask_right = xn2 > xk2[:, [-1]]
    out = np.where(mask_right, right, out)

    # Reshape back and restore axis
    out = out.reshape(B + (K,))
    out = np.moveaxis(out, -1, axis)
    return out


def intergolate(x_bin_edges,xin,yin, left=None, right=None):
    """Compute average of xin,yin within supplied bins.

    This funtion is similar to interpolation, but averages the curve repesented
    by xin,yin over the supplied bins to produce the output, yout.

    This is particularly useful, for example, for a spectrum of narrow emission
    incident on a detector with broad pixels. Each pixel averages out or
    "dilutes" the lines that fall within its range. However, simply
    interpolating at the pixel midpoints is a mistake as these points will
    often land between lines and predict no flux in a pixel where narrow but
    strong lines will actually produce significant flux.

    left and right have the same definition as in np.interp
    """

    x = np.hstack((x_bin_edges, xin))
    x = np.sort(x)
    y = np.interp(x, xin, yin, left, right)
    I = cumulative_trapz(y, x, True)
    Iedges = np.interp(x_bin_edges, x, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)

    return y_bin_avg


def jitter(values, errs):
    return np.random.randn(*values.shape)*errs + values


def jitter_dex(values, errdex):
    return values*10**(np.random.randn(*values.shape) * errdex)


def pcolor_reg(x, y, z, **kw):
    """
    Similar to `pcolor`, but assume that the grid is uniform,
    and do plotting with the (much faster) `imshow` function.

    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y should be 1-dimensional")
    if z.ndim != 2 or z.shape != (y.size, x.size):
        raise ValueError("z.shape should be (y.size, x.size)")
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.allclose(dx, dx[0], 1e-2) or not np.allclose(dy, dy[0], 1e-2):
        raise ValueError("The grid must be uniform")

    if np.issubdtype(z.dtype, np.complexfloating):
        zp = np.zeros(z.shape, float)
        zp[...] = z[...]
        z = zp

    plt.imshow(z, origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='nearest',
               aspect='auto',
               **kw)
    plt.axis('tight')


def is_list_like(obj):
    return isinstance(obj, (list, tuple, np.ndarray))


def click_coords(fig=None, timeout=600.):
    if fig is None:
        fig = plt.gcf()

    xy = []
    def onclick(event):
        if not event.inaxes:
            fig.canvas.stop_event_loop()
        else:
            xy.append([event.xdata, event.ydata])

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.start_event_loop(timeout=timeout)
    fig.canvas.mpl_disconnect(cid)
    return np.array(xy)


def flux_integral(w, f, range=None, e=None, bin_widths=None):
    we = mids2bins(w, bin_widths=bin_widths)

    if range is not None:
        if range[0] < we[0] or range[-1] > we[-1]:
            raise ValueError('Integration range extends beyond spectrum.')

    if range is not None:
        a, b = we[:-1], we[1:]
        binmask = (we > range[0]) & (we < range[1])
        fmask = (b > range[0]) & (a < range[1])
        we = we[binmask]
        we = np.insert(we, (0, len(we)), range)
        f = f[fmask]
        if e is not None:
            e = e[fmask]

    dw = np.diff(we)
    flux = np.sum(dw*f)
    if e is None:
        return flux

    flux_var = np.sum((dw*e)**2)
    flux_err = np.sqrt(flux_var)
    return flux, flux_err


def sliding_center_slice(x, window_size):
    w = window_size
    n = len(x)
    out_len = n - w + 1
    start = w // 2
    end = start + out_len
    return slice(start, None if end == 0 else end)


def apply_across_window(x, ary_fn, window_size):
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(x, window_shape=window_size)
    center_slc = sliding_center_slice(x, window_size)
    result = np.ma.masked_all_like(x)
    result[center_slc] = ary_fn(windows, axis=-1)
    return result


def shift_floor_to_zero(x, window_size=50):
    """
    Shifts arrya so that the floor is zero based on a sliding window.
    The ends of the array will be masked where a full sliding window won't fit.

    Parameters
    ----------
    x
    window_size

    Returns
    -------

    """
    mins = apply_across_window(x, np.min, window_size)
    return x - mins


def click_n_plot(fig, plot_fn):
    def get_and_plot():
        xy = click_coords(fig)
        if len(xy):
            x, y = zip(*xy)
            plotted_artists = plot_fn(x)
            plt.draw()
            return x, y, plotted_artists
        else:
            return [], [], []

    print('Collecting points. Click off the plot when done.')
    x, y, artists = get_and_plot()
    while True:
        print('Click off the plot if satisfied. Click new points if not.')
        xnew, ynew, newartists = get_and_plot()
        if xnew:
            # Remove plotted artists before repeating
            for artist in artists:
                artist.remove()
            x, y, artists = xnew, ynew, newartists
        else:
            break
    return x, y


def query_next_step(batch_mode=True, care_level=0, threshold=0, msg='Continue?'):
    if batch_mode:
        if care_level >= threshold:
            answer = input(msg)
            if answer != '':
                raise StopIteration


# region plot utilities
def save_standard_mpld3(fig, path):
    dpi = fig.get_dpi()
    fig.set_dpi(150)
    plugins.connect(fig, plugins.MousePosition(fontsize=14))
    mpld3.save_html(fig, str(path))
    fig.set_dpi(dpi)


def step_edges(bin_edges, y, ax='current', **plt_kws):
    if ax == 'current':
        ax = plt.gca()
    edges2x = np.repeat(bin_edges, 2)
    y2 = np.repeat(y, 2)
    result = ax.plot(edges2x[1:-1], y2, **plt_kws)
    return result


def step_mids(bin_midpts, y, ax='current', bin_widths=None, **plt_kws):
    if ax == 'current':
        ax = plt.gca()
    edges = mids2bins(bin_midpts, bin_widths=bin_widths)
    return step_edges(edges, y, ax=ax, **plt_kws)


def save_pdf_png(fig, basepath, pngdpi=300):
    basepath = Path(basepath)
    fig.savefig(basepath.parent / (basepath.name + '.pdf'))
    fig.savefig(basepath.parent / (basepath.name + '.png'), dpi=pngdpi)

# endregion


def rebin(new_edges, old_edges, y):
    dx = np.diff(old_edges)
    areas = y*dx
    integral = np.insert(np.cumsum(areas), 0, 0)
    Iedges = np.interp(new_edges, old_edges, integral)
    return np.diff(Iedges)/np.diff(new_edges)


def is_in_range(x, lo, hi):
    return (x >= lo) & (x <= hi)


def quadsum(x, axis=None):
    return np.sqrt(np.sum(x**2, axis=axis))


def subinterval_cumsums(x):
    x = np.asarray(x)
    n = len(x)

    # Compute cumulative sum with one extra zero at the beginning
    cumsum = np.zeros(n + 1, dtype=x.dtype)
    cumsum[1:] = np.cumsum(x)

    # Now create a 2D array such that S[i, j] = np.sumsum[i:j] for i < j
    # Broadcast to build 2D difference matrix
    start = cumsum[:, None]  # shape (n, 1)
    end   = cumsum[None, :]   # shape (1, n)

    S = end - start  # shape (n, n)

    # Zero out the invalid parts (i >= j)
    i, j = np.indices((n+1, n+1))
    S[i >= j] = 0

    return S


def flux_average(exptimes, fluxes, errors, axis=None):
    F = exptimes * fluxes
    E = exptimes * errors
    T = np.sum(exptimes, axis=axis)
    avg = np.sum(F, axis=axis) / T
    avg_err = quadsum(E, axis=axis) / T
    return avg, avg_err


def chunk_edges(chunk_mask):
    """
    Find the start and end indices of contiguous True chunks in a boolean array.

    Parameters
    ----------
    chunk_mask : array-like of bool

    Returns
    -------
    chunks : list of tuple
        List of (start, end) index tuples for each contiguous True chunk.
        Each tuple represents a half-open interval [start, end).
    """
    chunk_mask = np.asarray(chunk_mask, dtype=bool)
    padded = np.pad(chunk_mask.astype(int), (1, 1), constant_values=0)
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    return list(zip(starts, ends))


def printprogress(items, attr=None, prefix=''):
    """
    Generator that yields items from a list while printing progress
    on the same line.
    """
    if prefix and not prefix.endswith(' '):
        prefix += ' '
    total = len(items)
    for i, item in enumerate(items, start=1):
        lbl = item if attr is None else getattr(item, attr)
        msg = f"{prefix}{i}/{total}: {lbl}"
        sys.stdout.write("\r" + msg + " " * 10 + "\n")  # pad to clear leftovers
        sys.stdout.flush()
        yield item
    print()  # move to a new line at the end


def qrange(a: Quantity, b: Quantity, step: Quantity):
    u = a.unit
    x = np.arange(a.to_value(u), b.to_value(u), step.to_value(u))
    return x * u


def contiguous_true_range(arr, start):
    n = len(arr)
    if not arr[start]:
        return None  # or raise an error

    # expand left
    left = start
    while left > 0 and arr[left - 1]:
        left -= 1

    # expand right
    right = start
    while right < n - 1 and arr[right + 1]:
        right += 1

    return left, right


def estimate_gain_sigma_eff(flux_data, sigma_data, flux_model, snr_min=2.0):
    """
    flux, sigma_data, model_flux: arrays (same shape)
    snr_min: use only pixels with flux/sigma_data >= snr_min and flux>0 to estimate gain
    Returns:
      sigma_eff: effective per-pixel uncertainties incorporating model-based Poisson floor
      g_used: per-pixel gain used (scalar if global)
    """
    flux_data = np.asarray(flux_data, float)
    sig  = np.asarray(sigma_data, float)
    var = sig ** 2
    mod    = np.asarray(flux_model, float)

    # robust gain estimate from good, positive-SNR pixels
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = flux_data / sig
        good = (flux_data > 0) & (sig > 0) & (snr >= snr_min)
        if not np.any(good):
            raise ValueError('No pixels have sufficient flux for a gain estimate.')
        gain = var / flux_data
        gain_pix = np.where(good, gain, np.nan)
    gain = np.nanmedian(gain_pix)

    # estimate Poisson variance from observed flux
    var_obs_pois = gain * np.clip(flux_data, 0, None)

    # estimate excess variance due to background flux
    var_bk = np.clip(var - var_obs_pois, 0, None)

    # Model-based Poisson variance in flux units
    var_mod_pois = gain * np.clip(mod, 0, None)

    # Combine with background variance
    var_eff = var_bk + var_mod_pois

    # avoid zeros
    var_eff = np.maximum(var_eff, np.finfo(float).tiny)

    return np.sqrt(var_eff), gain


def most_common_item(elements):
    (result, _), = Counter(elements).most_common(1)
    return result