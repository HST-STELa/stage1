import numpy as np
from astropy import table
from astropy import units as u

import paths
import utilities as utils
from hst_utilities import read_etc_output

from lya_prediction_tools import lya
from lya_prediction_tools.spectrograph import Spectrograph as GenericSpectrograph


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


def load_etc_and_lsf(grating, aperture):
    # load in spectrograph info
    etc_file = paths.stis / default_etc_filenames[grating][aperture]
    lsf_file = paths.stis / f'lsf.hst-stis-{grating}-1200.txt'
    proxy_aperture = proxy_lsf_apertures[grating].get(aperture, aperture)
    etc = read_etc_output(etc_file)
    lsf_x, lsf_y = read_lsf(lsf_file, aperture=proxy_aperture)
    return etc, lsf_x, lsf_y


class Spectrograph(GenericSpectrograph):
    @classmethod
    def from_grating_aperture(cls, grating, aperture, wave_window=None):
        etc, lsf_x, lsf_y = load_etc_and_lsf(grating, aperture)
        etc = cls._window_etc(etc, wave_window)
        return cls(lsf_x, lsf_y, etc)

    @classmethod
    def _window_etc(cls, etc, wave_window):
        if wave_window is None:
            return etc

        in_window = utils.is_in_range(etc['wavelength'], *wave_window)
        etc = etc[in_window]
        return etc

    @classmethod
    def from_x1d(cls, x1d_hdu, order=1, wave_window=None, grating='infer', aperture='infer'):
        hdr = x1d_hdu[0].header
        if grating == 'infer':
            grating = hdr['opt_elem'].lower()
        if aperture == 'infer':
            aperture = hdr['aperture'].lower()
        wdata = x1d_hdu[1].data['wavelength'][order - 1]

        # interpolate etc data onto the wavelength grid of the x1d
        etc, lsf_x, lsf_y = load_etc_and_lsf(grating, aperture)
        ycols = set(etc.colnames) - {'wavelength'}
        wetc = etc['wavelength']
        etc_x1d = table.Table([wdata], names=['wavelength'], meta=etc.meta)
        for ycol in ycols:
            ynew = np.interp(wdata, wetc, etc[ycol])
            etc_x1d[ycol] = ynew

        etc_x1d = cls._window_etc(etc_x1d, wave_window)

        return cls(lsf_x, lsf_y, etc_x1d)


etc_acq_times = table.Table.read(paths.stis / 'ACQ_snr40_and_saturation_times.csv')
etc_g140m_times = table.Table.read(paths.stis / 'G140M_maxrates_and_buffer.csv')
etc_g140l_times = table.Table.read(paths.stis / 'G140L_maxrates_and_buffer.csv')
etc_e140m_times = table.Table.read(paths.stis / 'E140M_maxrates_and_buffer.csv')

breathing_rms = {
    '0.2x0.2': 0.05,
    '6x0.2': 0.05, # using 0.2x0.2 as proxy
    '52x0.05': 0.118,
    '52x0.1': 0.088,
    '52x0.2': 0.045,
    '52x0.5': 0.027,
    '52x2': 0.013
}

peakup_overhead = {
    '52x0.05' : 300,
    '52x0.1' : 220,
}

peakup_num_exposures = {
    '52x0.05' : 16,
    '52x0.1' : 12
}

def shorten_exposures_by_peakups(aperture, acq_exptime, exptimes, visit_start_indices):
    if aperture in peakup_overhead:
        overhead = peakup_overhead[aperture] + peakup_num_exposures[aperture] * acq_exptime
        exptimes_mod = exptimes.copy()
        exptimes_mod[visit_start_indices] -= overhead * u.s
    else:
        exptimes_mod = exptimes.copy()
    return exptimes_mod




