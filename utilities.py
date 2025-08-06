import warnings

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
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(x, window_shape=(window_size, 1))
    windows = windows[:, :, :, 0]
    min_error = np.min(windows, axis=-1)
    center_slc = sliding_center_slice(x, window_size)
    shifted_error = np.ma.masked_all_like(x)
    shifted_error[center_slc, ...] = x[center_slc, ...] - min_error
    return shifted_error


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


def query_next_step(batch_mode=True, care_level=0, threshold=0):
    if batch_mode:
        if care_level >= threshold:
            answer = input('Continue?')
            if answer != '':
                raise StopIteration


def save_standard_mpld3(fig, path):
    dpi = fig.get_dpi()
    fig.set_dpi(150)
    plugins.connect(fig, plugins.MousePosition(fontsize=14))
    mpld3.save_html(fig, str(path))
    fig.set_dpi(dpi)


def step_edges(bin_edges, y, ax=plt.gca(), **plt_kws):
    edges2x = np.repeat(bin_edges, 2)
    y2 = np.repeat(y, 2)
    result = ax.plot(edges2x[1:-1], y2, **plt_kws)
    return result


def step_mids(bin_midpts, y, ax=plt.gca(), bin_widths=None, **plt_kws):
    edges = mids2bins(bin_midpts, bin_widths=bin_widths)
    return step_edges(edges, y, ax=ax, **plt_kws)


def rebin(new_edges, old_edges, y):
    dx = np.diff(old_edges)
    areas = y*dx
    integral = np.insert(np.cumsum(areas), 0, 0)
    Iedges = np.interp(new_edges, old_edges, integral)
    return np.diff(Iedges)/np.diff(new_edges)