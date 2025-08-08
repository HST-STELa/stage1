from math import pi

import numpy as np

import empirical
import utilities as utils



def rossby_number(mass, rotation_period):
    tau = empirical.turnover_time(mass.to_value('Msun'))
    return rotation_period / tau


def saturation_decay(x, x_break, y_saturation, x_ref, y_ref):
    x = np.atleast_1d(x)
    slope = (y_ref - y_saturation) / (x_ref - x_break)
    result = np.zeros_like(x)
    sat = x <= x_break
    result[sat] = y_saturation
    y_decay = slope * (x - x_break) + y_saturation
    result[~sat] = y_decay[~sat]
    return np.squeeze(result)


def saturation_decay_loglog(x, x_break, y_saturation, x_ref, y_ref):
    args = list(map(np.log10, (x, x_break, y_saturation, x_ref, y_ref)))
    ly = saturation_decay(*args)
    return 10**ly


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


def noisy_rotation(t, jitter_amplitude, rotation_amplitude, rotation_period, num_draws=1):
    m = num_draws
    n = len(t)
    jitter_amplitude = expand_jitter(t, jitter_amplitude)
    jitter = np.random.randn(m, n) * jitter_amplitude[None,:]
    random_phase_offset = np.random.random(m)
    rot_vec = rotation(t, rotation_amplitude, rotation_period, random_phase_offset)
    noisy_rot_vec = jitter + rot_vec
    noisy_rot_vec = np.squeeze(noisy_rot_vec)
    return noisy_rot_vec, rot_vec


def added_transit_uncertainty(
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
    y_noisy, _ = noisy_rotation(
        t,
        jitter_amplitude,
        rotation_amplitude,
        rotation_period,
        num_draws)

    # compute the difference between baseline and transit for each draw
    bx = utils.is_in_range(t, *baseline_range)
    if sum(bx) == 0:
        raise ValueError('Baseline range contains no points.')
    tx = utils.is_in_range(t, *in_transit_range)
    if sum(tx) == 0:
        raise ValueError('In-transit range contains no points.')
    ybases, _ = utils.flux_average(exptimes[None,bx], y_noisy[:,bx], 0, axis=1)
    ytransits, _ = utils.flux_average(exptimes[None,tx], y_noisy[:,tx], 0, axis=1)
    diffs = ytransits - ybases

    # return the scatter on those differences
    return np.std(diffs)
