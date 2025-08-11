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


def read_etc_output(etc_output_file):
    try:
        pattern = r'.*stis-(.*?)\..*(.*?)exptime(.*?)_flux(.*?)_aperture(.*?)\.csv'
        result, = re.findall(pattern, etc_output_file.name)
        grating, date, expt, flux, aperture = result
    except ValueError as e:
        if 'too many values' in str(e) or 'not enough values' in str(e):
            raise(ValueError('Nonstandard ETC output file name. Example standard name: '
                             'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv'))
        else:
            raise
    etc = table.Table.read(etc_output_file)
    etc.meta['date of etc run'] = date
    etc.meta['aperture'] = aperture
    etc.meta['grating'] = grating
    etc.meta['exptime'] = float(expt)
    etc.meta['flux'] = float(flux)
    etc['flux2cps'] = etc['target_counts'] / etc.meta['exptime'] / etc.meta['flux']
    etc['bkgnd_cps'] = (etc['total_counts'] - etc['target_counts']) / etc.meta['exptime']
    return etc


def read_lsf(path_lsf, aperture):
    lsf_raw = np.loadtxt(path_lsf, skiprows=2)
    with open(path_lsf) as f:
        lines = f.readlines()
    hdr_line = lines[1]
    hdr_line = hdr_line.replace('Rel pixel', 'pixel')
    names = hdr_line.split()
    lsf_raw = table.Table(lsf_raw, names=names)

    # seems to be measured on a grid that varies slightly in spacing, so I will interpolate onto an even grid
    dpix = np.mean(np.diff(lsf_raw['pixel']))
    npix = lsf_raw['pixel'][-1] - lsf_raw['pixel'][0]
    npts = int(npix / dpix)
    if npts % 2 == 1:
        npts = npts - 1
    x = np.arange(-npts / 2 * dpix, (npts / 2 + 1) * dpix, dpix)
    y = np.interp(x, lsf_raw['pixel'], lsf_raw[aperture])

    # normalize so convolution preserves flux
    ynormfac = np.trapz(y)
    ynorm = y / ynormfac

    return x, ynorm


_etc_files_for_smple_snr = [
    'etc.hst-stis-g140m.2021-09-28.unknown.exptime1700_flux1e-13_aperture52x0.2.csv',
    'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv'
]
_etc_dict_simple_snr = {}
for _file in _etc_files_for_smple_snr:
    _etc = read_etc_output(paths.stis / _file)
    _date = _etc.meta['date of etc run']
    _etc_dict_simple_snr[_date] = _etc


def simple_sim_g140m_obs(f, expt, etc_run_date='2021-09-28'):
    etc = _etc_dict_simple_snr[etc_run_date]
    w = etc['wavelength'] * u.AA
    we = utils.mids2bins(w.value) * u.AA
    fpix = utils.intergolate(we, lya.wgrid_std, f)
    src = fpix * etc['flux2cps'] * expt
    bkgnd = etc['bkgnd_cps'] * expt
    total = bkgnd + src
    err_counts = np.sqrt(total)
    err_flux = err_counts / expt / etc['flux2cps']
    return w, we, fpix, err_flux


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


default_etc_filenames = dict(
    g140m = {
        '52x0.5': 'etc.hst-stis-g140m.2025-08-05.2025103.exptime900_flux1e-13_aperture52x0.5.csv',
        '52x0.2': 'etc.hst-stis-g140m.2025-08-05.2025093.exptime900_flux1e-13_aperture52x0.2.csv',
        '52x0.1': 'etc.hst-stis-g140m.2025-08-05.2025102.exptime900_flux1e-13_aperture52x0.1.csv',
        '52x0.05': 'etc.hst-stis-g140m.2025-08-05.2025101.exptime900_flux1e-13_aperture52x0.05.csv'
    },
    e140m = {
        '0.2x0.2': 'etc.hst-stis-e140m.2025-08-05.2025092.exptime900_flux1e-13_aperture0.2x0.2.csv',
        '6x0.2': 'etc.hst-stis-e140m.2025-08-05.2025104.exptime900_flux1e-13_aperture6x0.2.csv',
        '52x0.05': 'etc.hst-stis-e140m.2025-08-05.2025105.exptime900_flux1e-13_aperture52x0.05.csv'
    },
    g140l = {
        '52x0.2': 'etc.hst-stis-g140l.2025-08-05.2025094.exptime900_flux1e-13_aperture52x0.2.csv'
    }
)

proxy_lsf_apertures = dict(
    g140m = {'52x0.05':'52x0.1'},
    e140m = {'52x0.05':'0.2x0.06'},
    g140l = {}
)

preloaded_spectrographs = {grating:{} for grating in ['g140m', 'e140m', 'g140l']}
for grating in preloaded_spectrographs.keys():
    for aperture in default_etc_filenames[grating].keys():
        # load in spectrograph info
        etc_file = paths.stis / default_etc_filenames[grating][aperture]
        lsf_file = paths.stis / f'LSF_{grating.upper()}_1200.txt'
        proxy_aperture = proxy_lsf_apertures[grating].get(aperture, aperture)
        etc = read_etc_output(etc_file)
        lsf_x, lsf_y = read_lsf(lsf_file, aperture=proxy_aperture)

        # slim down to just around the lya line for speed
        window = lya.v2w((-500, 500))
        in_window = utils.is_in_range(etc['wavelength'], *window)
        etc = etc[in_window]

        # initialize spectrograph object
        spec = Spectrograph(lsf_x, lsf_y, etc)

        preloaded_spectrographs[grating][aperture] = spec
g140m = preloaded_spectrographs['g140m']['52x0.2']
e140m = preloaded_spectrographs['e140m']['0.2x0.2']
g140l = preloaded_spectrographs['g140l']['52x0.2']

etc_acq_times = table.Table.read(paths.stis / 'ACQ_snr40_and_saturation_times.csv')
etc_g140m_times = table.Table.read(paths.stis / 'G140M_maxrates_and_buffer.csv')
etc_g140l_times = table.Table.read(paths.stis / 'G140L_maxrates_and_buffer.csv')
etc_e140m_times = table.Table.read(paths.stis / 'E140M_maxrates_and_buffer.csv')

breathing_rms = {
    '0.2x0.2': 0.05,
    '52x0.05': 0.118,
    '52x0.1': 0.088,
    '52x0.2': 0.045,
    '52x0.5': 0.027,
    '52x2': 0.013
}
