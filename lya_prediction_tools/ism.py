import numpy as np
from astropy import units as u
from astropy import coordinates as coord

from reference_tables import ism_columns

n_H = 10 ** ism_columns['N_H'] * u.cm ** -2 / (ism_columns['d'] * u.pc)
n_H = n_H.to('cm-3')
ism_columns['n_H'] = n_H


def ism_velocity(ra, dec):
    # see http://lism.wesleyan.edu/ColoradoLIC.html and
    # lallement & bertin 1992
    v_sun = 26*u.km/u.s
    v_sun_direction = coord.SkyCoord(74.5*u.deg, -6*u.deg)
    v_sun_vector = v_sun * v_sun_direction.represent_as('cartesian').xyz

    target_directions = coord.SkyCoord(ra, dec)
    target_vectors = target_directions.represent_as('cartesian').xyz

    v_sun_component = np.dot(v_sun_vector.to_value(), target_vectors) * v_sun_vector.unit
    return v_sun_component


def ism_n_H_percentile(percentile):
    n_H_msmts = ism_columns['n_H'].quantity.to_value('cm-3')
    n_H = np.percentile(n_H_msmts, percentile) * u.cm**-3 # unit gets lost in percentile
    return n_H