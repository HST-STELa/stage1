from math import pi

import numpy as np
from astropy.timeseries.periodograms.lombscargle.implementations.mle import design_matrix
from astropy import units as u

import utilities as utils
from utilities import quadsum


def expand_jitter(t, jitter_amplitude):
    if np.isscalar(jitter_amplitude):
        jitter_amplitude = np.tile(jitter_amplitude, t.shape)
    return jitter_amplitude


def rotation(t, rotation_amplitude, rotation_period, phase_offset):
    phase_offset = np.atleast_1d(phase_offset)
    phases = t[None, :] / rotation_period + phase_offset[:, None]
    argument = (2*pi*phases).to_value('')
    rot_vec = rotation_amplitude * np.sin(argument)
    return np.squeeze(rot_vec)


def noisy_rotation(t, jitter_amplitude, rotation_amplitude, rotation_period, num_draws=1, return_truth=False):
    m = num_draws
    n = len(t)
    jitter_amplitude = expand_jitter(t, jitter_amplitude)
    jitter = np.random.randn(m, n) * jitter_amplitude[None,:]
    random_phase_offset = np.random.random(m)
    rot_vec = rotation(t, rotation_amplitude, rotation_period, random_phase_offset)
    noisy_rot_vec = jitter + rot_vec
    noisy_rot_vec = np.squeeze(noisy_rot_vec)
    if return_truth:
        return noisy_rot_vec, rot_vec
    else:
        return rot_vec


def added_transit_uncertainty(
        t,
        exptimes,
        normalization_snrs,
        jitter_amplitude,
        rotation_period,
        rotation_amplitude,
        in_transit_range,
        baseline_range,
        num_draws
):
    """
    Inject and then try to recover rotation period signals in order to come up with an estimate for the uncertainties
    resulting from unknown rotational phase and amplitude.

    Parameters
    ----------
    t
    exptimes
    normalization_snrs : snr of the flux measured in the normalization range for each exposure
    jitter_amplitude : relative amplitude of stellar/instrumental jitter
    rotation_period : stellar rotation period
    rotation_amplitude : relative amplitude of rotational modulation
    in_transit_range : time range that will be used for in-transit flux average
    baseline_range : time range that will be used for out-of-transit baseline
    num_draws

    Returns
    -------
    """
    y_noisy, y_true = noisy_rotation(
        t,
        jitter_amplitude,
        rotation_amplitude,
        rotation_period,
        num_draws,
        return_truth=True)
    dy = quadsum((np.array(1/normalization_snrs, jitter_amplitude)))
    frequency = 1/rotation_period

    # fit the data with rotation and jitter added
    X = design_matrix(t, frequency, dy=dy, bias=True, nterms=1)
    y_scaled = y_noisy / dy[None,:]
    theta_mle = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_scaled.T))

    # compute residuals relative to true model
    X_fit = design_matrix(t, frequency, bias=True, nterms=1)
    y_mle = np.dot(X_fit, theta_mle).T
    residuals = y_mle - y_true

    # compute the difference between baseline and transit for each draw
    bx = utils.is_in_range(t, *baseline_range)
    tx = utils.is_in_range(t, *in_transit_range)
    ybases, _ = utils.flux_average(exptimes[None,bx], residuals[:,bx], 0, axis=1)
    ytransits, _ = utils.flux_average(exptimes[None,tx], residuals[:,tx], 0, axis=1)
    diffs = ytransits - ybases

    # return the scatter on those differences
    return np.std(diffs)
