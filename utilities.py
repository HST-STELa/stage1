import warnings

import numpy as np
from matplotlib import pyplot as plt


def midpts(ary, axis=None):
    """Computes the midpoints between points in a vector.

    Output has length len(vec)-1.
    """
    if type(ary) != np.ndarray: ary = np.array(ary)
    if axis == None:
        return (ary[1:] + ary[:-1])/2.0
    else:
        hi = np.split(ary, [1], axis=axis)[1]
        lo = np.split(ary, [-1], axis=axis)[0]
        return (hi+lo)/2.0


def grid2bins(mids):
    """
    Reconstructs bin edges given only the midpoints by assuming the edges fall evenly between the midpoints.
    Note that this assumption can be invalid, particularly if the midpoints are not spaced evenly.

    Parameters
    ----------
    mids : 1-D array-like
        A 1-D array or list of the midpoints from which bin edges are to be
        inferred.

    Result
    ------
    edges : np.array
        The inferred bin edges.

    """

    edges = midpts(mids)
    d0 = edges[0] - mids[0]
    d1 = mids[-1] - edges[-1]
    edges = np.insert(edges, [0, len(edges)], [mids[0] - d0, mids[-1] + d1])
    return edges


def cumulative_trapz(y, x, zero_start=False):
    result = np.cumsum(midpts(y)*np.diff(x))
    if zero_start:
        result = np.insert(result, 0, 0)
    return result


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


def flux_integral(w, f, e=None):
    we = grid2bins(w)
    dw = np.diff(we)
    flux = np.sum(dw*f)
    if e is None:
        return flux

    flux_var = np.sum((dw*e)**2)
    flux_err = np.sqrt(flux_var)
    return flux, flux_err