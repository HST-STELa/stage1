import numpy as np
from scipy import optimize
from astropy.modeling.functional_models import Voigt1D
from astropy.io import fits
from matplotlib import pyplot as plt


# airglow function runs about 5x faster if the code instantiates this object now rather than within the function
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
    I = cumulative_trapz(y, x, True)
    Iedges = np.interp(x_bin_edges, x, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)
    return y_bin_avg


def boxcars_to_bins(midpoints, widths, heights, bin_edges):
    """
    Distribute the areas of multiple 1D boxcar functions into bins.

    Parameters
    ----------
    midpoints : array-like, shape (M,)
        Midpoints of the M boxcar functions.
    widths : array-like, shape (M,)
        Widths of the M boxcar functions.
    heights : array-like, shape (M,)
        Heights of the M boxcar functions.
    bin_edges : array-like, shape (N+1,)
        Bin edges defining N bins.

    Returns
    -------
    bin_areas : ndarray, shape (M, N)
        Amount of boxcar area from each boxcar in each bin.
    """
    midpoints = np.asarray(midpoints)[:, None]  # shape (M,1)
    widths = np.asarray(widths)[:, None]
    heights = np.asarray(heights)[:, None]
    bin_edges = np.asarray(bin_edges)  # shape (N+1,)

    lefts = midpoints - widths / 2  # (M,1)
    rights = midpoints + widths / 2  # (M,1)

    bin_lefts = bin_edges[:-1][None, :]  # shape (1, N)
    bin_rights = bin_edges[1:][None, :]  # shape (1, N)

    # Compute overlap region per boxcar and bin
    overlap_lefts = np.maximum(lefts, bin_lefts)  # shape (M,N)
    overlap_rights = np.minimum(rights, bin_rights)
    overlap_widths = np.maximum(0.0, overlap_rights - overlap_lefts)

    bin_areas = overlap_widths * heights  # shape (M, N)

    return bin_areas
# endregion


class AirglowModel(object):
    parameter_order = ['midpts', 'widths', 'fluxes', 'fwhm_Gs', 'fwhm_Ls']
    n_params_per_trace = len(parameter_order)
    _1d_organization_string = (f"[param1_trace1, param1_trace2, ..., paramn_trace1, paramn_trace2] for an example with two "
                               f"traces.\nThe order of the grouped parmeters is {str(parameter_order)}.")

    def __init__(self, wavegrids, dw_sample, plate_scale, tolerances, midpt_rng):
        """
        Initialize an AirglowModel to simultaneously model the airglow of multiple traces, intended for use with STIS
        modes that observe Lya.

        Parameters
        ----------
        wavegrids : list of wavelength grids to be used in model evaluations during MCMC or other optimization
        dw_sample : grid spacing for supersampling of the airglow profile
        plate_scale : factor enabling the scaling the slit width into a width in AA on the dispersion axis
            This is the dispersion (AA/pixel) divided by the conventional plate scale (arcsec / pixel) yielding
            units of AA/arcsec
        tolerances : list of sigmas for priors tying the parameters of the fits to each trace to each other.
            constrains how tightly the various parameters of the fit to each trace will be pinned to each
            other during fitting. smaller values mean the fits to each trace will not be allowed to differ as much.
        midpt_rng : where the airglow should be centered, used to define a uniform prior that can keep a sampler
            from getting stuck trying to fit the continuum

        Returns
        -------
        AirglowModel object

        Notes
        -----
        At Lya for STIS/G140M the dispersion / plate scale ratio is 1.838 AA/arcsec according to the IHB and for
        E140M it is 0.357 AA/arcsec
        These values roughly agree with a visual inspection of _flt files.
        """
        self.wavegrids = wavegrids
        self.dw_sample = dw_sample
        self.num_traces = len(wavegrids)
        self.plate_scale = plate_scale
        self.tolerances = np.asarray(tolerances)
        self.midpt_rng = midpt_rng

        # helpful for parsing 1d parameter set
        slices = [slice(3*i, 3*(i+1)) for i in range(self.n_params_per_trace)]
        self.param_slices = dict(zip(self.parameter_order, slices))

        # supersample the wavelength grids
        self.wavesup = self._wave_supersample(wavegrids, dw_sample)
        self.supbins = mids2edges(self.wavesup)

        # setting these to None here mainly to enable introspection
        self.midpts = None
        self.widths = None
        self.fluxes = None
        self.fwhm_Gs = None
        self.fwhm_Ls = None

    def _wave_supersample(self, wavegrids, dw_sample):
        # supersample the wavelength grid
        wmin = min(min(wavegrid) for wavegrid in wavegrids)
        wmax = max(max(wavegrid) for wavegrid in wavegrids)
        return np.arange(wmin, wmax, dw_sample)

    # region convenience functions for parsing and setting parameters
    def tile_params(self, params_single_set):
        """Copy a single set of parameters into a set for all of the traces."""
        ary = np.tile(params_single_set, (self.num_traces,1))
        return ary.T.ravel()

    def params_2d_to_1d(self, params_2d):
        if params_2d.shape[0] == self.num_traces:
            params_2d = params_2d.T
        return np.ravel(params_2d)

    def params_1d_to_2d(self, params):
        """Transform a 1d list of params into a 2d array where each row represents a parameter and each column a set
        of parameters for a given trace. Parameters should be organized as
        {}"""
        params = np.asarray(params)
        return params.reshape((self.n_params_per_trace, self.num_traces))
    params_1d_to_2d.__doc__.format(_1d_organization_string)

    def params_dict(self, params1d):
        params2d = self.params_1d_to_2d(params1d)
        return dict(*zip(self.parameter_order, params2d.T))

    def set_params(self, params1d):
        """
        Parameters
        ----------
        params1d : paremeters of the airglow model. This should be a list of parameters grouped by trace, e.g.,
            {}

        Returns
        -------
        None. Parameters of the object set in place.

        """
        params2d = self.params_1d_to_2d(params1d)
        for i, name in enumerate(self.parameter_order):
            setattr(self, name, params2d[i])
    set_params.__doc__.format(_1d_organization_string)

    @property
    def param_sets(self):
        return [getattr(self, name) for name in self.parameter_order]

    @property
    def params_1d(self):
        return np.hstack(self.param_sets)

    @property
    def params_2d(self):
        return np.array(self.param_sets)
    # endregion

    def evaluate(self, params=None, wavegrids=None, dw_sample=None):
        """
        Generate airglow profiles for each trace on the pre-defined grid with the given parameters. This will update
        the parameters of the AirglowModel object in-place. If None, the object's current parameters are used.

        Parameters
        ----------
        params : parameters of the airglow model. This should be a list of parameters grouped by trace, e.g.,
            {}

        Returns
        -------
        airglow flux in erg s-1 cm-2 AA-1 within the bins

        """
        if params is not None:
            self.set_params(params)
        if wavegrids is None:
            wavegrids = self.wavegrids
            wavesup = self.wavesup
            supbins = self.supbins
        else:
            wavesup = self._wave_supersample(wavegrids, dw_sample)
            supbins = mids2edges(wavesup)

        # boxcar widths in AA based on plate scale
        widths_AA = self.widths * self.plate_scale

        # boxcars for the image of the aperture
        yboxes = boxcars_to_bins(self.midpts, widths_AA, self.fluxes, supbins)

        # voigt profile for the natural and thermal broadening, convolve with boxcar
        x_0 = 0
        amplitude_Ls = 2 / (np.pi * self.fwhm_Ls)
        voigt_span = 3 * (max(self.fwhm_Gs) + max(self.fwhm_Ls))
        n = 2 * int(voigt_span // self.dw_sample) + 1 # ensures the grid is centered on 0
        voigt_grid = np.linspace(-voigt_span, voigt_span, num=n)
        nextra = len(voigt_grid) - len(self.wavesup)
        if nextra > 0: # in case the sampler tries out a voigt profile larger than the range being fit
            nclip = nextra // 2 + 1
            voigt_grid = voigt_grid[nclip:-nclip]
        sets = zip(yboxes, amplitude_Ls, self.fwhm_Ls, self.fwhm_Gs)
        ys = []
        for ybox, A, L, G in sets:
            yvoigt = voigt.evaluate(voigt_grid, x_0=x_0, amplitude_L=A, fwhm_L=L, fwhm_G=G)
            y = np.convolve(ybox, yvoigt, mode='same')
            ys.append(y)

        # bin
        ys_binned = [bin_average(w, wavesup, y, left=0, right=0) for w, y in zip(wavegrids, ys)]

        return ys_binned
    params_1d_to_2d.__doc__.format(_1d_organization_string)

    def loglike_tolerance_prior(self, params1d):
        """Return the log prior likelihood of the given set of parameters. The parameters should be organized as
        {}

        Models are penalized based on how much the parameters of the airglow fits to different traces vary, scaled
        by the tolerances.
        """
        params2d = self.params_1d_to_2d(params1d)
        means = np.mean(params2d, axis=1)
        terms = -(params2d - means[:,None])**2/2/self.tolerances[:,None]**2
        return np.sum(terms)
    loglike_tolerance_prior.__doc__.format(_1d_organization_string)

    def loglike_physical_prior(self, params1d):
        if np.any(params1d < 0):
            return -np.inf
        return 0

    def loglike_midpoint_prior(self, params1d):
        midpts = params1d[self.param_slices['midpts']]
        lo = midpts < self.midpt_rng[0]
        hi = midpts > self.midpt_rng[1]
        if np.any(lo | hi):
            return -np.inf
        return 0

    def __call__(self, wavegrids, dw_sample=None):
        """Generate airglow profiles across the supplied wavegrid."""
        return self.evaluate(params=None, wavegrids=wavegrids, dw_sample=dw_sample)


# code snippet to get files for testing on Parke's machine
import paths
base_name = 'toi-1696.hst-stis-g140m.2025-04-01T035830.ofhjal010_x1d{}.fits'
suffixes = ('bk1', 'bk2', 'trace')
three_trace_files = [paths.observations / 'hst-stis' /  base_name.format(suffix) for suffix in suffixes]

def test_airglow_models(three_trace_files):
    import emcee
    import corner

    fit_range = [1213, 1218]
    waves = []
    wavegrids = []
    fluxes = []
    errors = []
    for file in three_trace_files:
        data = fits.getdata(file, 1)
        w, f, e = [data[s][0] for s in ['wavelength', 'flux', 'error']]
        keep = (w > fit_range[0]) & (w < fit_range[1])
        fluxes.append(f[keep])
        errors.append(e[keep])
        waves.append(w[keep])
        wgrid = mids2edges(w[keep])
        wavegrids.append(wgrid)

    # increase the flux of one of the traces by 2
    # fluxes[1] *= 2

    sets = ((0.001, 'tight tolerance'),
            (10, 'loose tolerances'))
    for tol_rel, tol_label in sets:

        # set up a model with tight tolerances
        tolerances = np.array([0.1, 0.01, 5.5e-13, 0.2, 0.1]) * tol_rel
        model = AirglowModel(wavegrids, 0.01, 1.838,
                             tolerances=tolerances, midpt_rng=[1215.4, 1215.7])

        fluxstack = np.hstack(fluxes)
        errorstack = np.hstack(errors)
        def loglike(params):
            physical_prior = model.loglike_physical_prior(params)
            if physical_prior == -np.inf:
                return -np.inf
            ys = model.evaluate(params)
            ystack = np.hstack(ys)
            terms = -(fluxstack - ystack)**2/2/errorstack**2
            loglike_data = np.sum(terms)
            loglike_midpts = model.loglike_midpoint_prior(params)
            loglike_tolerance = model.loglike_tolerance_prior(params)
            return loglike_data + loglike_tolerance + loglike_midpts + physical_prior
            # return loglike_data + loglike_midpts

        # guess_single = [1215.54, 0.2, 5.5e-13, 0.2, 0.1]
        # guess_all_traces = model.tile_params(guess_single)

        np.random.seed(42)

        p0_2d = np.array(((1215.54, 0.2, 5.5e-13, 0.2, 0.1),
                          (1215.54, 0.2, 5.5e-13, 0.2, 0.1),
                          (1215.54, 0.2, 5.5e-13, 0.2, 0.1)))
        p0 = model.params_2d_to_1d(p0_2d)
        jitter_amplitude = np.tile((0.01, 0.01, 1e-14, 0.01, 0.01), (3,1))
        jitter_amplitude = model.params_2d_to_1d(jitter_amplitude)
        ndim = len(p0)
        nwalkers = ndim * 3
        jitter = np.random.randn(nwalkers, ndim) * jitter_amplitude
        p0 = p0[None,:] + jitter

        sampler = emcee.EnsembleSampler(nwalkers,  ndim, loglike)
        state = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        finalstate = sampler.run_mcmc(state, 1000, progress=True)

        # plot corner for fluxes
        for name, slc in model.param_slices.items():
            fig = plt.figure()
            labels = [f'flux {i+1}' for i in range(3)]
            # corner.corner(np.log10(sampler.flatchain[:,slc]), labels=labels, fig=fig)
            corner.corner(sampler.flatchain[:,slc], labels=labels, fig=fig)
            fig.suptitle(f'{tol_label} {name} posteriors')

        # plot median fits
        p_median = np.median(sampler.flatchain, axis=0)
        ys = model.evaluate(p_median)
        for i, (wave, flux, y) in enumerate(zip(waves, fluxes, ys)):
            plt.figure()
            plt.step(wave, flux, where='mid')
            plt.step(wave, y, where='mid')
            plt.title(f'{tol_label} | trace {i+1}')