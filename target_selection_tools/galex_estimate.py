import numpy as np

from target_selection_tools import reference_tables as ref

# using a catalog of galex mags vs B-V color to do estimate a mag for a standard field star and then adjusting
# based on rotation rate or assuming saturated for stars without a known rotation rate or age
usable = ~ref.catsup['B-V'].mask & ~ref.catsup['FUVmag'].mask
catsup = ref.catsup[usable]
catsup['FUV-V'] = catsup['FUVmag'] - catsup['Vmag']
_outlier = (catsup['FUV-V'] > 10) & (catsup['B-V'] < 0.25)
catsup = catsup[~_outlier]
BV, FV = catsup['B-V'], catsup['FUV-V']
BVbreak = 0.8

# first fit to determine the "median"
_lo = BV < BVbreak
_pmlo = np.polyfit(BV[_lo], FV[_lo], 2)
_pmhi = np.polyfit(BV[~_lo], FV[~_lo], 1)


def FUV_V_median_function(BVvec):
    result = np.zeros_like(BVvec)
    lo = BVvec < BVbreak
    result[lo] = np.polyval(_pmlo, BVvec[lo])
    result[~lo] = np.polyval(_pmhi, BVvec[~lo])
    return result


# then fit to the upper half
_upper = FV > FUV_V_median_function(BV)
_plo = np.polyfit(BV[_upper & _lo], FV[_upper & _lo], 2)
_phi = np.polyfit(BV[_upper & ~_lo], FV[_upper & ~_lo], 1)


def FUV_V_upper_function(BVvec):
    result = np.zeros_like(BVvec)
    lo = BVvec < BVbreak
    result[lo] = np.polyval(_plo, BVvec[lo])
    result[~lo] = np.polyval(_phi, BVvec[~lo])
    return result


def estimate_FUV_V_color_conservative(BVvec):
    return FUV_V_upper_function(BVvec)

