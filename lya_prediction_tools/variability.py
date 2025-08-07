from math import pi

import numpy as np
from astropy.timeseries.periodograms.lombscargle.implementations.mle import design_matrix
from astropy import units as u

import utilities as utils


def expand_jitter(t, jitter_amplitude):
    if np.isscalar(jitter_amplitude):
        jitter_amplitude = np.tile(jitter_amplitude, t.shape)
    return jitter_amplitude


def timeseries(t, jitter_amplitude, rotation_amplitude, rotation_period, num_draws=1):
    m = num_draws
    n = len(t)
    jitter_amplitude = expand_jitter(t, jitter_amplitude)
    jitter = np.random.randn(m, n) * jitter_amplitude[None,:]
    random_phase_offset = np.random.random(m)
    phases = t[None,:]/rotation_period + random_phase_offset[:,None]
    argument = (2*pi*phases).to_value('')
    rot_vec = rotation_amplitude * np.sin(argument)
    y = jitter + rot_vec
    return np.squeeze(y)


def rotation_errorbars(
        t,
        exptimes,
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
    times
    fluxes
    jitter
    rotation_period
    in_transit_range
    baseline_range
    num_draws

    Returns
    -------
    """
    ydraws = timeseries(t, jitter_amplitude, rotation_amplitude, rotation_period, num_draws)
    dy = expand_jitter(t, jitter_amplitude)
    frequency = 1/rotation_period

    # fit the data with rotation and jitter added
    X = design_matrix(t, frequency, dy=dy, bias=True, nterms=1)
    y_scaled = ydraws / dy[None,:]
    theta_MLE = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_scaled.T))

    # compute residuals
    X_fit = design_matrix(t, frequency, bias=True, nterms=1)
    models = np.dot(X_fit, theta_MLE).T
    residuals = ydraws - models
    # residuals = ydraws

    # compute the difference between baseline and transit for each draw
    basemask = utils.is_in_range(t, *baseline_range)
    transitmask = utils.is_in_range(t, *in_transit_range)
    Y = residuals * exptimes[None,:]
    Tbase = np.sum(exptimes[basemask])
    Ttransit = np.sum(exptimes[transitmask])
    ybases = np.sum(Y[:, basemask], axis=1) / Tbase
    ytransits = np.sum(Y[:, transitmask], axis=1) / Ttransit
    diffs = ytransits - ybases

    # estimate expected variance in differences based on noise alone
    E = dy * exptimes
    base_uncty = utils.quadsum(E[basemask]) / Tbase
    transit_uncty =  utils.quadsum(E[transitmask]) / Ttransit
    expected_scatter = utils.quadsum(u.Quantity((base_uncty, transit_uncty)))

    # estimate actual variance and compare
    actual_scatter = np.std(diffs)
    if actual_variance < expected_variance:
        return 0
    excess_noise = np.sqrt(actual_variance - expected_variance)
    return excess_noise
