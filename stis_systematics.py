import numpy as np

from astropy.modeling.functional_models import Voigt1D

# airglow function runs about 5x faster if we instantiate this object now rather than within the function
voigt = Voigt1D()


"""
Ideas for airglow subtraction:

- could check and see if a GP can reproduce airglow variations in flux and width across masked regions of airglow line
- could check to see if a variations in width and flux seem consistent between exposures with the same settings
  (modulo an offset in flux)
- PCA with enough images could make be a one-size fits all 
- could simultaneously fit airglow with minimum parameter model
    + spline or GP could work, see convo with chatGPT
    + want to try using EMD on the residuals of airglow fits and then optimize so that the best airglow fit is one
      where the largest amplitude mode in the residuals has the fewest minima and maxima and no negatives, particularly
      within the expected width of the airglow.
    + seems like the key thing our brains are doing when we note residual airglow is that we see emission in the
      absorbed core that we know should be absent, so perhaps we can code this somehow
"""


# region basic utilities
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


def mids2edges(mids):
    """
    Reconstructs bin edges given only the midpoints.

    Parameters
    ----------
    mids : 1-D array-like
        A 1-D array or list of the midpoints from which bin edges are to be
        inferred.

    Result
    ------
    edges : np.array
        The inferred bin edges.

    Could be accelerated with a cython implementation.
    """

    edges = midpts(mids)
    d0 = edges[0] - mids[0]
    d1 = mids[-1] - edges[-1]
    return np.insert(edges, [0, len(edges)], [mids[0] - d0, mids[-1] + d1])


def cumulative_trapz(y, x, zero_start=False):
    result = np.cumsum(midpts(y)*np.diff(x))
    if zero_start:
        result = np.insert(result, 0, 0)
    return result


def bin_average(x_bin_edges, x, y, left=None, right=None):
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

    x = np.hstack((x_bin_edges, x))
    x = np.sort(x)
    y = np.interp(x, x, y, left, right)
    I = cumulative_trapz(y, x, True)
    Iedges = np.interp(x_bin_edges, x, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)

    return y_bin_avg


# endregion


class AirglowModel(object):
    def __int__(self, wavegrid, plate_scale, num_traces, tolerances):
        self.wavegrid = wavegrid
        self.plate_scale = plate_scale
        self.num_traces = num_traces
        self.tolerances = tolerances

    def set_params(self, params):
        pass

    def get_params_1d(self):
        pass

    def params_to_2d(self):
        pass

    def evaluate(self):
        pass

    def evaluate_prior(self):
        pass

    def __call__(self, wavegrid):
        pass


def airglow(wavegrid, midpoint, width, diffuse_flux, fwhm_G, fwhm_L, plate_scale):
    """

    Parameters
    ----------
    wavegrid : wavelength grid midpoints in AA
    midpoint : midpoint of the airglow line in AA
    width : width in arcsec
    diffuse_flux : flux of the airglow in erg s-1 cm-2 AA-1 arcsec-2
    fwhm_G : full width at half max of the gaussian (thermal/turbulent broadened/scattering?)
             component of the airglow emission in AA (i.e., 2 sqrt(2 log(2)) * sqrt(kB * T / m)
    fwhm_L : full width at half max of the lorentzian (naturally broadened)
             component of the airglow emission in AA (i.e., A * w0**2 / 2 pi c)

    Returns
    -------
    airglow flux in erg s-1 cm-2 AA-1 within the bins

    """
    # supersample the wavelength grid
    n = len(wavegrid)
    wavesup = np.linspace(wavegrid[0], wavegrid[-1], n*10)
    dwsup = wavesup[1] - wavesup[0]

    # boxcar width in AA based on plate scale
    width_AA = width * plate_scale

    # boxcar for the image of the aperture
    supbins = mids2edges(wavesup)
    ybox = boxcar_to_bins(supbins, midpoint, width_AA, diffuse_flux)

    # voigt profile for the natural and thermal broadening
    x_0 = 0
    amplitude_L = 2 / (np.pi * fwhm_L)
    voig_span = 3*(fwhm_G + fwhm_L)
    voigt_grid = np.arange(voig_span, voig_span + dwsup, dwsup)
    yvoigt = voigt.evaluate(voigt_grid, x_0=x_0, amplitude_L=amplitude_L, fwhm_L=fwhm_L, fwhm_G=fwhm_G)

    # convolve the two
    y = np.convolve(ybox, yvoigt, mode='same')

    # bin
    ybinned = bin_average(wavegrid, wavesup, y, left=0, right=0)

    return ybinned


def bk_params_to_2d(params):
    params = np.asarray(params)
    return params.reshape((-1,))


def generate_bk_prior_function(tolerances, n_lines):
    def logprior(params):
        params = np.asarray(params)
        params = params.reshape((-1,n_lines))
        means = np.mean(params)


def aperture_width_logprior(sample_width, expected_width=0.2, tolerance=0.02):
    """

    Parameters
    ----------
    sample_width : width being suggested by sampler, in arcsec
    expected_width : expected width, in arcsec
    tolerance : std deviation of gaussian prior

    Returns
    -------
    log likelihood from a gaussian prior on width

    Notes
    -----
    The expected width can possibly be determined from the aperture width multiplied by dispersion / plate scale.
    At Lya for STIS/G140M the dispersion / plate scale ratio is 1.838 AA/arcsec according to the IHB
    and for E140M it is 0.357 AA/arcsec
    Looking at the _flt files, I estimate values of 1.86 AA/arcsec (G140M) 0.386 AA/arcsec (E140M), but they were
    hard to estimate, so I think the handbook values are a good place to start.
    """
    return -(sample_width - expected_width) ** 2/2/tolerance**2


def aperture_midpt_logprior(sample_midpt, expected_midpt=1215.67, tolerance=0.1)
    """
    
    Parameters
    ----------
    sample_midpt : midpoint being suggested by sampler, in AA
    expected_midpt : where the emission should be centered, in AA
    tolerance : tolerance, in AA. For STIS G140M 20 km s-1 is likely reasonable 

    Returns
    -------
    log likelihood from a gaussian prior on midpt
    """
    return -(sample_midpt - expected_midpt) ** 2 / 2 / tolerance ** 2


def boxcar_to_bins(bin_edges, midpoint, width, height):
    """
    Distribute the area of a 1D boxcar function into bins.

    Parameters:
    ----------
    midpoint : float
        Center of the boxcar.
    width : float
        Full width of the boxcar.
    height : float
        Height of the boxcar.
    bin_edges : array-like
        Array of bin edges (length N+1 for N bins).

    Returns:
    -------
    bin_values : ndarray
        Array of values (length N), with the amount of boxcar area in each bin.
    """

    left = midpoint - width / 2
    right = midpoint + width / 2
    bin_edges = np.asarray(bin_edges)

    # Get overlap with each bin
    bin_lefts = bin_edges[:-1]
    bin_rights = bin_edges[1:]

    overlap_lefts = np.maximum(bin_lefts, left)
    overlap_rights = np.minimum(bin_rights, right)
    overlap_widths = np.maximum(0.0, overlap_rights - overlap_lefts)

    # Area in each bin = overlapping width * height
    bin_areas = overlap_widths * height

    return bin_areas


