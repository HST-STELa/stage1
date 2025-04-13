import numpy as np
from astropy import units as u


# location of radius gap
def ho_gap_lowlim(P, Mstar):
    A, B, C = -0.09, 0.21, 0.35
    width = 0.1
    logP = np.log10(P)
    logM = np.log10(Mstar)
    logR = A*logP + B*logM + C - width/2
    return 10**logR


# rossby number
def turnover_time(Msun):
    # eqn 6 from Wright+ 2018
    log_tau = 2.33 - 1.50 + 0.31*Msun**2
    return 10**log_tau * u.d

