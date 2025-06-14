from math import nan
import re

from astropy import table
import numpy as np
from scipy import interpolate

import catalog_utilities as catutils
from lya_prediction_tools import etc
from target_selection_tools import query
from target_selection_tools import reference_tables as ref
from target_selection_tools import empirical as emp
from target_selection_tools import galex_estimate


def get_simbad_info(simbad_ids):
    fields = 'coo_err_maja plx pmra pmdec flux(V) flux_error(V) flux(G) flux_error(G) flux(B) flux_error(B) flux(U) flux_error(U) sptype sp_qual'.split()
    simbad = query.get_simbad_from_names(simbad_ids, fields)
    return simbad


def fill_optical_magnitudes(catalog, simbad):
    # use latest V mag from simbad wherever possible
    transfer = np.isfinite(simbad['FLUX_V'].filled(nan))
    catalog['sy_vmag'][transfer] = simbad['FLUX_V'][transfer]
    catalog['sy_vmagerr1'][transfer] = catalog['sy_vmagerr2'][transfer] = simbad['FLUX_ERROR_V'][transfer]
    print(f'{sum(transfer)} SIMBAD V mags available and adopted.')

    # copy over G mags from simbad
    transfer = np.isfinite(simbad['FLUX_G'].filled(nan))
    catalog['sy_gmag'][transfer] = simbad['FLUX_G'][transfer]
    print(f'{sum(transfer)} SIMBAD G mags available and adopted.')

    # copy over B mags from simbad
    transfer = np.isfinite(simbad['FLUX_B'].filled(nan))
    catalog['sy_bmag'][transfer] = simbad['FLUX_B'][transfer]
    print(f'{sum(transfer)} SIMBAD B mags available and adopted.')

    # copy over U mags from simbad
    transfer = np.isfinite(simbad['FLUX_U'].filled(nan))
    catalog['sy_umag'][transfer] = simbad['FLUX_U'][transfer]
    print(f'{sum(transfer)} SIMBAD U mags available and adopted.')

    # use G mags to estimate V mag based on mamajek table where there isn't a V
    G_V = catutils.safe_interp_table(catalog['st_teff'].filled(0), 'Teff', 'G-V', ref.mamajek)
    V_from_G = catalog['sy_gmag'] - G_V
    missing = catalog['sy_vmag'].mask
    transfer = ~V_from_G.mask & missing
    assert not np.any(transfer & catalog['st_teff'].mask)  # if this isn't true, need to deal with masked values
    print(f'{sum(transfer)} of {sum(missing)} missing V mags filled by inferring from G mag.')
    catalog['sy_vmag'][transfer] = V_from_G[transfer]

    # estimate remaining empty V mags from the mamajek table based on Teff and distance
    Mv = catutils.safe_interp_table(catalog['st_teff'], 'Teff', 'Mv', ref.mamajek)
    d = catalog['sy_dist'].quantity.to_value('pc')
    V = Mv + 5 * np.log10(d) - 5
    transfer = catalog['sy_vmag'].mask
    print(f'Remaining {sum(transfer)} missing V mags filled by inferring from Teff.')
    catalog['sy_vmag'][transfer] = V[transfer]


def fill_spectral_types(catalog, simbad, return_flags=False):
    # flag = 0 existing SpT, 1 filled from SIMBAD, 2 filled based on Teff
    flags = np.zeros(len(catalog), int)

    # use SIMBAD where it is good
    missing = catalog['st_spectype'].filled('') == ''
    simbad_good = (simbad['SP_QUAL'] <= 'C') & (simbad['SP_TYPE'] != '')  # copy only if quality is better than C
    transfer = missing & simbad_good
    catalog['st_spectype'][transfer] = simbad['SP_TYPE'][transfer]
    flags[transfer] = 1
    print(f'{sum(transfer)} of {sum(missing)} missing SpTs filled from SIMBAD.')

    # infer from Teff for the rest
    interper = interpolate.interp1d(ref.mamajek['Teff'], np.arange(len(ref.mamajek)), 'nearest')
    missing = catalog['st_spectype'].filled('') == ''
    itype = interper(catalog['st_teff'].filled(0)[missing])
    mamSpT = ref.mamajek['SpT'][itype.astype(int)]
    catalog['st_spectype'][missing] = mamSpT
    flags[missing] = 2
    print(f'Remaining {sum(missing)} missing SpT values filled based on Teff.')

    if return_flags:
        return flags


def conservatively_bright_FUV(catalog):
    FUV_adopted = np.array([nan]*len(catalog))

    # estimate all values based on B-V color from Teff
    BV = catutils.safe_interp_table(catalog['st_teff'], 'Teff', 'B-V', ref.mamajek)
    FV = galex_estimate.estimate_FUV_V_color_conservative(BV)
    FUVmag_field = FV + catalog['sy_vmag']

    # adjust for Rossby number
    tau = emp.turnover_time(catalog['st_mass'])
    Ro = catalog['st_rotp']/tau
    saturated = Ro.filled(0) < 0.1  # assume saturated as a worst case
    old = catalog['st_age'].filled(0) > 1  # but not for stars we know are old
    saturated[old] = False
    Ro_sat = 0.1
    Ro_nom = 0.7
    index = 1.8
    fluxfac = (Ro.filled(Ro_nom) / Ro_nom) ** -index  # note filling with Ro_nom here
    fluxfac[saturated] = (Ro_sat / Ro_nom) ** -index  # then replace for stars we know or assume are saturated
    mag_offset = -2.5 * np.log10(fluxfac)
    FUV_from_BV = FUVmag_field + mag_offset

    # adopt measured values where available
    measured = catalog['sy_fuvmaglim'].filled(1) == 0
    FUV_adopted[measured] = catalog['sy_fuvmag'][measured]

    # use limits if more stringent than B-V based estimate
    lims = catalog['sy_fuvmaglim'].filled(0) == -1
    limit_is_dimmer_than_BV = catalog['sy_fuvmag'].filled(0) > FUV_from_BV.filled(999)
    stringent_lim = lims & limit_is_dimmer_than_BV
    FUV_adopted[stringent_lim] = catalog['sy_fuvmag'][stringent_lim]

    # use B-V for any remaining values
    remainder = np.isnan(FUV_adopted)
    FUV_adopted[remainder] = FUV_from_BV[remainder]

    return FUV_adopted


def categorize_activity(catalog_with_mags):
    catalog = catalog_with_mags
    assert np.all(np.isfinite(catalog['sy_vmag'].filled(nan)))
    B_V = catutils.safe_interp_table(catalog['st_teff'], 'Teff', 'B-V', ref.mamajek)
    n = len(catalog)
    masked = lambda x: np.ma.is_masked(x)
    empty_obj_ary = lambda: np.array(['']*n, dtype=object)
    Prot = catalog['st_rotp'].filled(0).quantity

    uv_factor = 3
    usable_uv_max_Teff = 5500
    Ro_limit = 0.5
    age_limit = 1

    active = np.ones(n, bool)
    final = np.zeros(n, bool)
    justification = empty_obj_ary()
    info = {
        'mass':empty_obj_ary(),
        'Teff':empty_obj_ary(),
        'fuv':empty_obj_ary(),
        'Ro':empty_obj_ary(),
        'age':empty_obj_ary()
    }

    for i, row in enumerate(catalog):
        tau = emp.turnover_time(row['st_mass'])
        info['mass'][i] = f'stellar mass {row['st_mass']:.2f}'
        info['Teff'][i] = f'stellar Teff {row['st_teff']:.2f}'

        # FUV and NUV
        mag = row[f'sy_fuvmag']
        lim = row[f'sy_fuvmaglim']
        if masked(mag):
            info['fuv'][i] = f'no GALEX fuv observation'
        else:
            assert lim in [0, -1]
            uv_v = galex_estimate.FUV_V_median_function(B_V[i])
            min_uv = uv_v + row['sy_vmag'] - 2.5*np.log10(uv_factor)
            comparator = '>' if lim == -1 else '='
            info['fuv'][i] = f'GALEX fuv mag {comparator} {mag:.2f}'
            if (row['st_teff'] < usable_uv_max_Teff) and not final[i]:
                if (mag > min_uv):
                    active[i] = False
                    final[i] = True
                    justification[i] = f'deemed INACTIVE on the basis fuv no more than {uv_factor}x brighter than median for field stars of similar Teff < {usable_uv_max_Teff}'
                if (lim == 0) and (mag <= min_uv):
                    active[i] = True
                    final[i] = True
                    justification[i] = f'deemed ACTIVE on the basis of fuv more than {uv_factor}x brighter than median for field stars of similar Teff < {usable_uv_max_Teff}'

        # Rossby number
        Ro = Prot[i] / tau
        if Ro == 0:
            info['Ro'][i] = 'Rossby number unknown due to no cataloged rotation period'
        else:
            info['Ro'][i] = f'Rossby number estimate of {Ro:.2f} based on measured {Prot[i]:.1f} rotation period'
            if not final[i]:
                final[i] = True
                if Ro > Ro_limit:
                    active[i] = False
                    justification[i] = f'deemed INACTIVE on the basis of Rossby number > {Ro_limit}'
                else:
                    active[i] = True
                    justification[i] = f'deemed ACTIVE on the basis of Rossby number <= {Ro_limit}'

        # age
        age = row['st_age']
        if masked(age):
            info['age'][i] = 'no cataloged age'
        else:
            info['age'][i] = f'cataloged age of {age:.1g} Gyr'
            if not final[i]:
                final[i] = True
                if age > age_limit:
                    active[i] = False
                    justification[i] = f'deemed INACTIVE on the basis of age > {age_limit}'
                else:
                    active[i] = True
                    justification[i] = f'deemed ACTIVE on the basis of age <= {age_limit}'

        # no info
        if not final[i]:
            justification[i] = f'deemed ACTIVE due to the absence of information indicating otherwise'

    return active, justification, info


def cat2apt_names(catalog_names):
    names = np.char.upper(catalog_names)
    names = np.char.replace(names, ' ', '')
    return names


def make_apt_target_table(catalog_with_mags, simbad):
    catalog = catalog_with_mags
    n = len(catalog)

    names = catalog['hostname'].astype(str).copy()
    names = cat2apt_names(names)
    aptcat = table.Table([names], names=['Target Name'], masked=True)

    # APT sky coordinates
    aptcat['RA'] = simbad['RA']
    aptcat['RA Uncertainty'] = simbad['COO_ERR_MAJA']/1000 # this will be an upper limit, needs to be in arcsec
    aptcat['DEC'] = simbad['DEC']
    aptcat['DEC Uncertainty'] = simbad['COO_ERR_MAJA']/1000 # needs to be in arcsec
    aptcat['Reference Frame'] = 'ICRS'
    aptcat['Epoch'] = 2000.0

    # APT proper motion and parallax
    aptcat['RA PM'] = simbad['PMRA']
    aptcat['RA PM units'] = str(simbad['PMRA'].unit).upper().replace(' ', '')
    aptcat['DEC PM'] = simbad['PMDEC']
    aptcat['DEC PM units'] = str(simbad['PMDEC'].unit).upper().replace(' ', '')
    aptcat['Annual Parallax'] = simbad['PLX_VALUE']/1000 # needs to be in arcsec

    # APT magnitudes
    aptcat['V-Magnitude'] = catalog['sy_vmag']
    catalog['sy_vmagerr1'].mask = catalog['sy_vmagerr1'].mask | ~np.isfinite(catalog['sy_vmagerr1']) # somehow some nans got in
    aptcat['Mag Uncertainty'] = catalog['sy_vmagerr1']
    other_fluxes_col = []
    ismasked = np.ma.is_masked
    for i in range(n):
        other_fluxes = []
        G = simbad['FLUX_G'][i]
        if not ismasked(G):
            other_fluxes.append(f'G={G:.2f}')
        nuv, nuvlim = catalog['sy_nuvmag'][i], catalog['sy_nuvmaglim'][i]
        if not ismasked(nuv) and nuvlim == 0:
            other_fluxes.append(f'NUV={nuv:.2f}')
        fuv, fuvlim = catalog['sy_fuvmag'][i], catalog['sy_fuvmaglim'][i]
        if not ismasked(fuv) and fuvlim == 0:
            other_fluxes.append(f'FUV={fuv:.2f}')
        other_fluxes_col.append(', '.join(other_fluxes))
    aptcat['Other Fluxes'] = other_fluxes_col

    # APT description
    aptcat['Category'] = 'STAR'
    descriptions = []
    for i in range(n):
        SpT = catalog['st_spectype'][i]
        SpT = SpT.replace(' ', '')
        assert 'I' not in SpT
        if '/' in SpT:
            SpT = SpT.split('/')[0]
        letter = SpT[0]
        if letter == 'F':
            try:
                num = int(SpT[1])
            except (ValueError, IndexError):
                num = 1
            if num < 3:
                SpT = 'F0-F2'
            else:
                SpT = 'F3-F9'
        else:
            SpT = f'{letter} V-IV'
        desc = f'[{SpT}, Extra-solar Planetary System]'
        descriptions.append(desc)
    aptcat['Description'] = descriptions
    aptcat['Radial Velocity'] = catalog['st_radv']

    # APT comments
    fuv_conservative = conservatively_bright_FUV(catalog)
    _, just, info = categorize_activity(catalog)
    comments = []
    for i in range(n):
        comment = (f'Predicted Lya flux before ISM absorption {catalog['Flya_earth_no_ISM'][i]:.1e};'
                   f'FUV used for buffer time estimate {fuv_conservative[i]:.2f};'
                   f'{just[i]};')
        comment += ';'.join([info[key][i] for key in ['mass', 'Teff', 'fuv', 'Ro', 'age']])
        comments.append(comment)
    aptcat['Comments'] = comments

    return aptcat


def write_target_table(aptcat, filepath, overwrite=False):
    aptcat.write(filepath, format='ascii.commented_header', overwrite=overwrite, delimiter=',')


def acquisition_setup(catalog):
    etc_acq = etc.etc_acq_times
    n = len(catalog)
    colnames = 'acq_filter acq_Texp_snr40 acq_Texp acq_Tsat'.split()
    dtypes = 'object float float float'.split()
    acq_setups =  catutils.empty_table(n, colnames, dtypes)

    ietcs = list(range(len(etc_acq)))
    findline = interpolate.interp1d(etc_acq['Teff'], ietcs, 'nearest')
    T_hard_min = 0.1
    LP = 'F28X50LP'
    ND = 'F25ND3'
    for i in range(n):
        Teff = catalog['st_teff'][i]
        ietc = int(findline(Teff))
        Vtarg = catalog['sy_vmag'][i]

        Texp_40 = {}
        Texp = {}
        Tsat = {}
        min_margin = {}
        for filter in [LP, ND]:
            Vetc = etc_acq[f'V_{filter}'][ietc]
            Texp_etc = etc_acq[f'Texp_{filter}'][ietc]
            Tsat_etc = etc_acq[f'Tsat_{filter}'][ietc]
            flux_ratio = 10 ** ((Vetc - Vtarg) / 2.5)
            Texp_40[filter] = Texp_etc / (flux_ratio) ** 2  # ^2 to get the same SNR as ETC
            Tsat[filter] = Tsat_etc / flux_ratio

            # if the saturation time is less than the min exposure, it's not going to work
            if Tsat[filter] < T_hard_min:
                Texp[filter] = np.nan
                min_margin[filter] = 0
                continue

            # try to split the difference between saturation and SNR 40 with a geometric mean
            Texp_desired = np.sqrt(Texp_40[filter] * Tsat[filter])
            if Texp_desired < T_hard_min:
                Texp_desired = T_hard_min
            Texp[filter] = Texp_desired
            bright_margin = Tsat[filter] / Texp_desired
            faint_margin = Texp_desired / Texp_40[filter]
            min_margin[filter] = min(faint_margin, bright_margin)

        # pick the filter that gives better margins
        # if both margins are > 10, pick the shorter exposure
        if (min_margin[LP] > 10) and (min_margin[ND] > 10):
            filter = LP
        elif min_margin[LP] > min_margin[ND]:
            filter = LP
        else:
            filter = ND
        assert min_margin[filter] > 3
        acq_setups['acq_filter'][i] = filter
        acq_setups['acq_Texp_snr40'][i] = Texp_40[filter]
        acq_setups['acq_Texp'][i] = Texp[filter]
        acq_setups['acq_Tsat'][i] = Tsat[filter]

    return acq_setups


def find_nearest_etc_rows(Teff, grating):
    Tetc = getattr(etc, f'etc_{grating}_times')
    ietcs = list(range(len(Tetc)))
    findline = interpolate.interp1d(Tetc['Teff'], ietcs, 'nearest', bounds_error=False, fill_value='extrapolate')
    i_lines = findline(Teff).astype(int)
    return Tetc[i_lines]


def buffer_times(catalog, grating='g140m'):
    assert not np.any(catalog['st_teff'].mask)
    etc = find_nearest_etc_rows(catalog['st_teff'].filled(nan), grating)
    FUVmags = conservatively_bright_FUV(catalog)
    flux_ratio = 10 ** ((etc['FUVmag'] - FUVmags) / 2.5)
    Tbuffer = etc['buffer_time'] / flux_ratio
    Tbuffer *= 4 / 5
    return Tbuffer


def does_mdwarf_isr_require_e140m(name, type='lya'):
    result = False
    if name in ref.mdwarf_isr['Target']:
        isr_config = ref.mdwarf_isr.loc[name][f'SCIENCE SETUP {type.upper()}']
        if 'E140M' in isr_config.upper():
            result = True
    return result


def count_rate_estimates(catalog, grating='g140m'):
    assert not np.any(catalog['st_teff'].mask)
    etc = find_nearest_etc_rows(catalog['st_teff'].filled(nan), grating)
    FUVmags = conservatively_bright_FUV(catalog)
    flux_ratio = 10 ** ((etc['FUVmag'] - FUVmags) / 2.5)
    cps_local = etc['local_max'] * flux_ratio
    cps_global = etc['global_max'] * flux_ratio
    return cps_local, cps_global


class VisitLabel:
    _character_shift = 13

    def __init__(self, label="A0"):
        """
        Initialize the label generator with a given label.
        :param current_label: Starting label (e.g., 'C3'). Defaults to 'A0'.
        """
        self.label = label

    def _increment_label(self, label):
        """
        Increment the label according to the specified format.
        """
        first_part, second_part = label[:-1], label[-1]

        # Increment the numeric part
        if second_part.isdigit():
            next_num = int(second_part) + 1

            if next_num < 10:
                return f"{first_part}{next_num}"
            else:
                # Reset to '0' and increment letter
                next_letter = chr(ord(first_part) + 1)
                if next_letter <= 'M':
                    return f"{next_letter}0"
                else:
                    return f"AA"

        # Handle second character sequences (e.g., AA, AB, ...)
        else:
            if second_part == 'Z':
                next_letter1 = chr(ord(first_part[0]) + 1)
                next_letter2 = 'A'
                return f"{next_letter1}{next_letter2}"
            else:
                next_letter2 = chr(ord(second_part) + 1)
                return f"{first_part[0]}{next_letter2}"

    def next_label(self):
        """
        Generate and return the next label.
        """
        self.label = self._increment_label(self.label)
        return VisitLabel(self.label)

    @classmethod
    def _shift(cls, letter, shift):
        i = ord(letter) + shift
        return chr(i)

    @classmethod
    def base_from_pair(cls, paired_label):
        base0 = cls._shift(paired_label[0], -cls._character_shift)
        return VisitLabel(base0 + paired_label[-1])

    @classmethod
    def paired_label(cls, base_label):
        paired0 = cls._shift(base_label[0], cls._character_shift)
        return VisitLabel(paired0 + base_label[-1])

    def next_pair(self):
        base = self.next_label()
        pair = self.paired_label(base)
        return base, pair

    def __eq__(self, other):
        return self.label == other.label

    def __gt__(self, other):
        s, o = self.label, other.label
        if s[1] > '9' and o[1] > '9': # both end in letters
            return s > o
        elif s[1] <= '9' and o[1] <= '9': # both end in numbers
            return s > o
        else: # one ends in a letter and the other in a number
            return s[1] > o[1]

    def __str__(self):
        return self.label

    def __repr__(self):
        return f'VisitLabel({str(self)})'

    def __getitem__(self, item):
        return self.label[item]


def parse_acqs_from_formatted_listing(path):

    # Read file lines
    with open(path, 'r') as file:
        lines = file.readlines()

    # Find lines containing 'ACQ'
    acq_lines = [line for line in lines if 'ACQ' in line]

    # Initialize lists to store extracted data
    targets, apers, exptimes = [], [], []

    # Define regex pattern for extraction
    pattern = re.compile(r'\d+\s+(\S+)\s+STIS/CCD ACQ\s+(\S+).*?(\d+\.\d+|\d+)\s*S')

    # Extract relevant information from each line
    for line in acq_lines:
        match = pattern.search(line)
        if match:
            target, aper, exptime = match.groups()
            targets.append(target)
            apers.append(aper)
            exptimes.append(float(exptime))

    # Create Astropy table
    acq_table = table.Table([targets, apers, exptimes], names=('target', 'aper', 'exptime'))

    return acq_table


