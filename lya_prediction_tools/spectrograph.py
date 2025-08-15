import numpy as np
from astropy import units as u

import utilities as utils


class Spectrograph(object):
    def __init__(self, lsf_x, lsf_y, etc_table):
        self.etc = etc = etc_table
        self.wavegrid = w = etc['wavelength']
        self.wavebins = utils.mids2bins(w)
        self.binwidth = np.diff(self.wavebins)
        self.aperture = etc.meta['aperture'].upper()
        self.grating = etc.meta['grating'].upper()

        self.lsf_x = lsf_x
        self.lsf_y = lsf_y
        self.dpix = lsf_x[1] - lsf_x[0]
        self.n_lsf = len(lsf_x) // 2

    def flux_uncty(self, pixel_flux, exptime):
        etc = self.etc
        w = etc['wavelength'] * u.AA
        src = pixel_flux * etc['flux2cps'] * exptime
        bkgnd = etc['bkgnd_cps'] * exptime
        total = bkgnd + src
        err_counts = np.sqrt(total)
        err_flux = err_counts / exptime / etc['flux2cps']
        return err_flux

    def observe(self, mod_w, mod_f, exptime):
        mod_dw = np.diff(mod_w)
        inst_w = self.wavegrid
        inst_dw = self.binwidth
        if np.any(mod_dw > np.min(inst_dw)/3):
            raise ValueError('Spectrum not well resolved.')

        # get things on the right grids and convolve
        # note that this interpolation will not preserve flux perfectly, particularly for features that are very sharp
        inst_lsf_pixel_grid = np.arange(self.lsf_x[0], self.lsf_x[-1] + len(inst_w) - 1, self.dpix)
        inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
        mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
        mod_f_conv = np.convolve(mod_f_interp, self.lsf_y, mode='valid')
        mod_w_conv = inst_lsf_w_grid[self.n_lsf:-self.n_lsf]

        # integrate over the pixels
        inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
        inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)
        msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)

        # compute error
        msrd_f_uncty = self.flux_uncty(msrd_f, exptime)

        return msrd_f, msrd_f_uncty

    def recommended_wave_grid(self):
        npix = len(self.wavegrid)
        inst_w = self.wavegrid
        x_inst = np.arange(npix)

        pixel = self.lsf_x
        lsf_rng = pixel[-1] - pixel[0]
        oversample_factor = int(round(len(self.lsf) / lsf_rng))
        dx_sample = 1 / oversample_factor

        # sample the lsf and the model onto a consistent pixel axis
        x_sample = np.arange(-lsf_rng, npix + lsf_rng + dx_sample, dx_sample)
        wave_fit = np.polyfit(x_inst, inst_w, 3)  # fits G140M perfectly!
        w_sample = np.polyval(wave_fit, x_sample)

        w_test = np.interp(x_inst, x_sample, w_sample)
        assert np.max(np.abs(w_test - inst_w) / inst_w) < 1e-3

        return w_sample

    def fast_observe_function(self, mod_w):
        inst_w = self.wavegrid

        # get things on the right grids to convolve
        inst_lsf_pixel_grid = np.arange(2 * self.lsf_x[0], 2 * self.lsf_x[-1] + len(inst_w) - 1, self.dpix)
        inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
        mod_w_conv = inst_lsf_w_grid[self.n_lsf:-self.n_lsf]

        # construct instrument pixel grid
        inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
        inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)

        def fast_observe(mod_f, exptime):
            mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
            mod_f_conv = np.convolve(mod_f_interp, self.lsf_y, mode='valid')
            msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)
            msrd_f_err = self.flux_uncty(msrd_f, exptime)
            return msrd_f, msrd_f_err

        return fast_observe


def intergolate(x_bin_edges, xin, yin):
    I = cumtrapz(yin, xin)
    Iedges = np.interp(x_bin_edges, xin, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)
    return y_bin_avg


def interp_inst_waves(w, pixels):
    i = np.arange(len(w))
    p = np.polyfit(i, w, 3)
    return np.polyval(p, pixels)


def midpoints(edges):
    return (edges[..., :-1] + edges[..., 1:])/2.


def cumtrapz(y, x):
    areas = midpoints(y) * np.diff(x)
    result = np.cumsum(areas)
    result = np.insert(result, 0, 0)
    return result
