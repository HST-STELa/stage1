from math import pi

import numpy as np

def timeseries(tgrid, jitter_amplitude, rotation_amplitude, rotation_period):
    n = len(tgrid)
    jitter_vec = np.random.randn(n) * jitter_amplitude
    random_phase_offset = np.random.random(1)
    phases = tgrid/rotation_period + random_phase_offset
    rot_vec = rotation_amplitude * np.sin(2*pi*phases)
    variability = jitter_vec + rot_vec
    return variability