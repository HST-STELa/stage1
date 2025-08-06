import re

import numpy as np
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt # for debugging

import paths
import utilities as utils

from lya_prediction_tools import lya


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


# region ETC reference info for SNR calcs
_etc_output_files_g140m = [
    'g140m_counts_2021-09-28_exptime1700_flux1e-13.csv',
    'g140m_counts_2025-08-05_exptime900_flux1e-13.csv'
]
etc_runs = {'g140m':{}, 'g140l':{}, 'e140m':{}}
for _file in _etc_output_files_g140m:
    _etc = table.Table.read(paths.stis / _file)
    (_grating, _date, _expt, _flux), = re.findall(r'(.*?)_counts_(.*?)_exptime(.*?)_flux(.*?)\.', _file)
    _etc.meta['date of etc run'] = _date
    _etc.meta['exptime'] = float(_expt)
    _etc.meta['flux'] = float(_flux)
    _etc['flux2cps'] = _etc['target_counts'] / _etc.meta['exptime'] / _etc.meta['flux']
    _etc['bkgnd_cps'] = (_etc['total_counts'] - _etc['target_counts']) / _etc.meta['exptime']
    etc_runs[_grating][_date] = _etc
# endregion


def simple_sim_g140m_obs(f, expt, etc_run_date='2021-09-28'):
    etc = etc_runs['g140m'][etc_run_date]
    w = etc['wavelength'] * u.AA
    we = utils.mids2bins(w.value) * u.AA
    fpix = utils.intergolate(we, lya.wgrid_std, f)
    src = fpix * _etc['flux2cps'] * expt
    bkgnd = _etc['bkgnd_cps'] * expt
    total = bkgnd + src
    err_counts = np.sqrt(total)
    err_flux = err_counts / expt / _etc['flux2cps']
    return w, we, fpix, err_flux


etc_acq_times = table.Table.read(paths.stis / 'ACQ.csv')
etc_g140m_times = table.Table.read(paths.stis / 'G140M.csv')
etc_g140l_times = table.Table.read(paths.stis / 'G140L.csv')
etc_e140m_times = table.Table.read(paths.stis / 'E140M.csv')


class Spectrograph(object):
    def __init__(self, path_lsf, path_etc):
        # region load in STIS LSFs
        lsf_raw = np.loadtxt(path_lsf, skiprows=2)
        lsf_raw = table.Table(lsf_raw, names='pixel 52X0.1 52X0.2 52X0.5 52X2.0'.split())
        self.dpix = dpix = np.mean(np.diff(lsf_raw['pixel']))

        # lsf seems to be measured on a grid that varies slightly in spacing, so I will interpolate onto an even grid
        npix = lsf_raw['pixel'][-1] - lsf_raw['pixel'][0]
        npts = int(npix/dpix)
        if npts % 2 == 1:
            npts = npts - 1
        _x = np.arange(-npts/2*dpix, (npts/2+1)*dpix, dpix)
        empty = np.zeros((len(_x), len(lsf_raw.colnames)))
        lsf = table.Table(empty, names=lsf_raw.colnames)
        lsf['pixel'] = _x
        for key in lsf.colnames[1:]:
            y = np.interp(_x, lsf_raw['pixel'], lsf_raw[key])
            ynorm = np.trapz(y)
            lsf[key] = y/ynorm # normalize so convolution preserves flux
        lsf['52X0.05'] = lsf['52X0.1']
        self.lsf = lsf
        self.n_lsf = len(lsf) // 2
        # endregion

    def observe(self, inst_w, mod_w, mod_f, aperture, return_we=False):
        lsf = self.lsf
        n_lsf = self.n_lsf

        mod_dw = np.diff(mod_w)
        inst_dw = np.diff(inst_w)
        if np.any(mod_dw > np.min(inst_dw)/3):
            raise ValueError('Spectrum not well resolved.')

        # get things on the right grids and convolve
        # note that this interpolation will not preserve flux perfectly, particularly for features that are very sharp
        inst_lsf_pixel_grid = np.arange(lsf['pixel'][0], lsf['pixel'][-1] + len(inst_w) - 1, self.dpix)
        inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
        mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
        mod_f_conv = np.convolve(mod_f_interp, lsf[aperture], mode='valid')
        mod_w_conv = inst_lsf_w_grid[n_lsf:-n_lsf]

        # integrate over the pixels
        inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
        inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)
        msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)

        if return_we:
            return msrd_f, inst_w_edges
        return msrd_f

    def recommended_wave_grid(self, inst_w):
        npix = len(inst_w)
        x_inst = np.arange(npix)
        xedges = np.arange(-0.5, npix)

        pixel = self.lsf['pixel']
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

    def fast_observe_function(self, inst_w, mod_w, aperture):
        lsf = self.lsf
        n_lsf = self.n_lsf

        # get things on the right grids to convolve
        inst_lsf_pixel_grid = np.arange(2 * lsf['pixel'][0], 2 * lsf['pixel'][-1] + len(inst_w) - 1, self.dpix)
        inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
        mod_w_conv = inst_lsf_w_grid[n_lsf:-n_lsf]

        # construct instrument pixel grid
        inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
        inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)

        def fast_observe(mod_f):
            mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
            mod_f_conv = np.convolve(mod_f_interp, lsf[aperture], mode='valid')
            msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)
            return msrd_f

        return fast_observe


path_lsf_g140m = paths.stage1_code / 'lya_prediction_tools' / 'LSF_G140M_1200.txt'
g140m = Spectrograph(path_lsf_g140m)

path_lsf_g140l = paths.stage1_code / 'lya_prediction_tools' / 'LSF_G140L_1200.txt'
g140l = Spectrograph(path_lsf_g140l)

path_lsf_g140l = paths.stage1_code / 'lya_prediction_tools' / 'LSF_E140M_1200.txt'
e140m = Spectrograph(path_lsf_g140l)