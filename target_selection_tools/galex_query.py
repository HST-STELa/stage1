from __future__ import division, print_function, absolute_import
import numpy as np
import requests
from astropy import table, time

# the SQL query for this code was modified from one in the gPhoton package
# (https://github.com/cmillion/gPhoton).
#
# description of MCAT columns available at
# http://www.galex.caltech.edu/wiki/Public:Documentation/Appendix_A.1

# constants
_j2000_jd = 2451545.0
_usable_fov = 1.2
_aper_7_radius = 23 / 2.  # pix
_aper_7_area = np.pi * (_aper_7_radius * 1.5) ** 2  # arcsec2

# details for accessing the MCAT from MAST
baseURL = ('https://mastcomp.stsci.edu/portal/Mashup/MashupQuery.asmx/Galex'
           'PhotonListQueryTest?query=')
MCATDB = 'GR6Plus7.dbo'
default_columns = "ra, dec, NUV_FLUX_APER_7, NUV_FLUXERR_APER_7, " \
                  "FUV_FLUX_APER_7, FUV_FLUXERR_APER_7, " \
                  "nuv_skybg, fuv_skybg, vpe.fexptime, " \
                  "vpe.nexptime, nuv_artifact, fuv_artifact, " \
                  "vpe.fexpstar,  vpe.fexpend, vpe.nexpstar, vpe.nexpend, " \
                  "fov_radius".split(', ')


def _combine_columns(addnl_columns):
    cols = default_columns
    if addnl_columns is not None:
        cols += addnl_columns
    return cols


def _assemble_http_query(ra, dec, radius, addnl_columns=None):
    cols = _combine_columns(addnl_columns)
    cols_str = ', '.join(cols)
    query = "{baseURL}select " \
            "{cols} " \
            "from {MCATDB}.visitphotoobjall as vpo " \
            "" \
            "inner join {MCATDB}.visitphotoextract as vpe on " \
            "vpo.photoextractid=vpe.photoextractid " \
            "" \
            "inner join {MCATDB}.fGetNearbyVisitObjEq({ra},{dec},{radius}) " \
            "as nb on vpo.objid=nb.objid " \
            "" \
            "&format=extjs" \
            "".format(baseURL=baseURL, MCATDB=MCATDB, ra=ra, dec=dec,
                      radius=radius*60., cols=cols_str)
    return query


def _strip_prefix(name):
    pieces = name.split('.')
    return pieces[-1]


def _mcattime2jd(time_mcat):
    time_unix = time.Time(time_mcat, format='unix')
    return time_unix.jd


def _skydist(ra0, dec0, ra1, dec1):
    """
    Compute arc-distance between two points on sky assuming small values.
    """
    dra = ra1 - ra0
    if hasattr(dra, '__iter__'):
        gtr = dra > 180
        dra[gtr] = 360 - dra[gtr]
    else:
        if dra > 180:
            dra = 360 - dra

    return np.sqrt(dra ** 2 + (dec1 - dec0) ** 2)


def _other_band(band):
    if band.upper() == "FUV":
        return "NUV"
    if band.upper() == "NUV":
        return "FUV"


def compute_expected_positions(ra, dec, pm_ra, pm_dec, jd):
    """
    Propagate proper motions to get the coordinates at a certain  time.

    Parameters
    ----------
    ra : array-like
        right ascenscion(s) J2000
    dec : array-like
        declination(s) J2000
    pm_ra : array-like
        right ascenscion proper motion in mas/yr
    pm_dec : array-like
        declination proper motion in mas/yr
    jd : float
        julian date at which RA and Dec are required

    Returns
    -------
    ra1, dec1 : arrays
        RA and Dec at jd.

    """
    pm = np.array((pm_ra, pm_dec))/3600/1000./365 # deg/d
    def translate(x, dxdt):
        return x + dxdt*(jd-_j2000_jd)
    return tuple(map(translate, (ra, dec), pm))


def fetch_mcat_data(ra, dec, search_radius=2 / 60., addnl_columns=None,
                    timeout=60., connection_retries=10):
    """
    Quiery the MCAT for sources within the search radius and put everything
    it returns into  an astropy table.

    Parameters
    ----------
    ra : float
        right ascenscion for center of search
    dec : float
        declination for center of search
    search_radius : float
        radius of search in degrees. For reference, 1-sigma astrometric
        uncertainty is 0.4 arcseconds.
    addnl_columns : list
        columns to request from server in addition to
        galex_motion.default_columns
    timeout : float
        How long to wait for the MAST  server to respond in seconds.
    connection_retries : int
        Number of times to retry if the server does not respond.

    Returns
    -------
    tbl : astropy table
        a table with the requested columns for all sources within the search
        radius
    """

    # query the server
    query = _assemble_http_query(ra, dec, search_radius,
                                 addnl_columns=addnl_columns)
    count = 1
    r = None
    while count <= connection_retries:
        try:
            r = requests.get(query, timeout=timeout)
            break
        except (requests.ConnectionError, requests.exceptions.ConnectTimeout):
            count += 1
            continue

    # parse output into a table
    try:
        rows = r.json()['data']['Tables'][0]['Rows']
    except:
        if r is None:
            raise ValueError(f'Request not filled due to connection errors on all {connection_retries} attempts.')
        else:
            raise ValueError(f'Request filled but no data returned. Instead got:\n{r.content}')
    cols = _combine_columns(addnl_columns)
    names = list(map(_strip_prefix, cols))
    if len(rows) == 0:
        tbl = table.Table(names=names, masked=True)
    else:
        tbl = table.Table(rows=rows, names=names, masked=True)
    tbl.meta['search_position'] = ra, dec
    tbl.meta['search_radius'] = search_radius
    return tbl


def add_position_offset_column(ra, dec, pm_ra, pm_dec, tbl):
    """
    Add a column to tbl in place that gives the offset of each detected source
    from some moving target.

    Parameters
    ----------
    ra : float
        Right ascenscion of reference point.
    dec : float
        Declination of reference point.
    pm_ra : float
        Proper motion in right ascencion.
    pm_dec : float
        Proper motion in declination.
    tbl : astropy table
        Table from an MCAT query.

    Returns
    -------
    None
    """

    # I suppose computing the exposure midpoint would be best, but the target
    # motion should always be negligible relative to exposure time so it
    # isn't necessary

    # find the time of each exposure for the FUV and NUV channels for each row
    fuv_and_nuv_times = np.array((tbl['nexpstar'], tbl['fexpstar']))

    # take the max of the two (will ensure -999 null value is not included)
    times = np.max(fuv_and_nuv_times, axis=0)

    # convert mcat times to jd
    jd = _mcattime2jd(times)

    # find the target ra, dec at the time of exposure for each row
    ra_expected, dec_expected = compute_expected_positions(ra, dec, pm_ra,
                                                           pm_dec, jd)

    # compute offsets from the proper motion shifted target and add to table
    tbl['offset'] = _skydist(ra_expected, dec_expected, tbl['ra'], tbl['dec'])


def get_nearest_source_fluxes(tbl, band, match_radius=2/3600.,
                              upper_limits=True):
    """
    Within a table of results from an  MCAT query with an offset column,
    find matches to the target source.

    Parameters
    ----------
    tbl : astropy table
        Results from an MCAT query with offset column added via
        `add_position_offset_column`.
    band : 'fuv' or 'nuv'
    match_radius : float
        Radius within which to consider a GALEX source a match to the target
        in degrees. For reference, the 1-sigma astrometric uncertainty is 0.4
        arcseconds for GALEX.
    upper_limits : bool
        Estimate upper limits for exposures where there is no match for the
        source.

    Returns
    -------
    fluxes : array
        Flux  of matching sources for each  GALEX visit, -999 indicates no
        match. units of counts s-1
    errors : array
        1-sigma error on source flux. If flux is -999 but error is positive,
        then the error gives the 2-sigma upper limit on the source flux.
    exptimes : array
        Exposure time of each GALEX visit.
    offsets : array
        Offset of the match from the target coordinates.
    starts : array
        Start of each GALEX visit in UNIX time, I believe (have not been able
        to verify this).
    """
    if len(tbl) == 0:
        return [], [], [], [], []

    # for each unique time (i.e. each unique exposure), find nearest source
    # note that there  can be multiple exposures  for a single tile, so don't
    # use tile id for this.
    letter = band[0]
    fluxcol = band.upper() + '_FLUX_APER_7'
    othercol = _other_band(band) + '_FLUX_APER_7'
    errcol = band.upper() + '_FLUXERR_APER_7'
    unique_expend = np.unique(tbl[letter + 'expend'])
    unique_expend = unique_expend[unique_expend > 0]
    fluxes, errs, expts, offsets, starts = [], [], [], [], []
    for expend in unique_expend:
        exp_tbl = tbl[tbl[letter + 'expend'] == expend]
        isort = np.argsort(exp_tbl['offset'])
        i_closest = isort[0]
        flux = exp_tbl[fluxcol][i_closest]
        null_flux = flux <= -99.
        if null_flux and len(isort) > 1:
            i_next = isort[1]
            # check to see if GALEX accidentally made one source two separate
            # sources in FUV and NUV
            flux_next = exp_tbl[fluxcol][i_next]
            flux_other = exp_tbl[othercol][i_closest]
            if flux_next > 0 and flux_other > 0:
                i_match = i_next
            else:
                i_match = i_closest
        else:
            i_match = i_closest

        flux = exp_tbl[fluxcol][i_match]
        err = exp_tbl[errcol][i_match]
        null_flux = flux <= -99.
        too_far = exp_tbl['offset'][i_match] > match_radius
        if too_far or null_flux:
            # first make sure tile center is close enough that source would have
            # been within the match radius if it was present. there is not a column
            # in the mcat database for boresight position, but if the center is
            # close enough then the distance between the points and the search
            # coordinates plus the points and the boresight will all be greater
            # than the detector radius
            ra_search, dec_search = tbl.meta['search_position']
            dist = _skydist(exp_tbl['ra'], exp_tbl['dec'],
                            ra_search, dec_search)
            if np.all(dist + exp_tbl['fov_radius'] > _usable_fov/2.):
                continue

            if upper_limits:
                exptcol = exp_tbl[letter + 'exptime'].filled(-1)
                startcol = exp_tbl[letter + 'expstar'].filled(-1)
                bg_cpscol = exp_tbl[band + '_skybg'].filled(-1)
                usable = (exptcol > 0) & (startcol > 0) & (bg_cpscol > 0)
                expt = np.median(exptcol[usable])
                start = np.median(startcol[usable])
                bg_cps = np.mean(bg_cpscol[usable]) * _aper_7_area
                bg_cpserr = np.sqrt(bg_cps/expt)
                lim = 2*bg_cpserr
                fluxes.append(-999.)
                errs.append(lim)
                expts.append(expt)
                starts.append(start)
            else:
                fluxes.append(-999.)
                errs.append(-999.)
                expts.append(-999.)
                starts.append(-999.)
            offsets.append(-999.)
        else:
            fluxes.append(flux)
            errs.append(err)
            offsets.append(exp_tbl['offset'][i_closest])
            expts.append(exp_tbl[letter + 'exptime'][i_closest])
            starts.append(exp_tbl[letter + 'expstar'][i_closest])

    return fluxes, errs, expts, offsets, starts


def extract_source(ra, dec, pm_ra, pm_dec, match_radius=4./3600.,
                   search_radius=25./60, query_timeout=60.,
                   upper_limits=True):
    """
    Find measurements in the MCAT matching a source.

    Parameters
    ----------
    ra : float
        Right ascencion of target.
    dec : float
        Declination of target.
    pm_ra : float
        Right ascencion proper motion of target.
    pm_dec : float
        Declination proper motion of target.
    match_radius : float
        Radius within which to consider a GALEX source a match to the target
        in degrees. For reference, the 1-sigma astrometric uncertainty is 0.4
        arcseconds for GALEX.
    search_radius : float
        Radius in  which  to query the MCAT in degrees. If upper limits are
        desired, this should  be large enough for the MCAT to return results
        whenever exposures were taken near enough that the target could have
        been in the aperture.
    query_timeout : float
        Seconds to wait for server to respond before giving up.
    upper_limits : bool
        Estimate upper limits for exposures where there is no match for the
        source.

    Returns
    -------
    (nuv_fluxes, fuv_fluxes) :
        nuv_fluxes and fuv_fluxes are each length 4 tuples of arrays giving
        the fluxes, errors, exposure times, and sky distance of the source to
        the target for each source matched to the target. Upper limits are
        give  a flux of -999. and the 2-sigma limit is given as the error.
        units of counts s-1
    """

    # compute coordinates at start and end of galex lifetime (2004-2011)
    def correct_coords(year):
        jd = (year - 2000)*365.25 + _j2000_jd
        return compute_expected_positions(ra, dec, pm_ra, pm_dec, jd)
    ra04, dec04 = correct_coords(2004)
    ra08, dec08 = correct_coords(2008)
    ra12, dec12 = correct_coords(2012)

    # enlarge radius to catch source over that range of positions
    path_length = np.sqrt((dec12 - dec04)**2 + (ra12 - ra04)**2)
    capture_radius = path_length/2 + search_radius

    # pull MCAT data from larger area to catch any tiles that might have had
    # the source
    tbl = fetch_mcat_data(ra08, dec08, capture_radius, timeout=query_timeout)

    # add column of offsets from expected source positions
    add_position_offset_column(ra, dec, pm_ra, pm_dec, tbl)

    # match source and get fluxes for each band
    def get_fluxes(band):
        return get_nearest_source_fluxes(tbl, band, match_radius=match_radius,
                                         upper_limits=upper_limits)

    return get_fluxes('nuv'), get_fluxes('fuv')


def coadd_fluxes(fluxes, errs, expts, sigma_clip=None):
    """
    Coadd a set of fluxes. If only upper limits (flux = -999, error > 0),
    return the most restrictive upper limit. Fluxes are weighted by exposure
    time.

    Parameters
    ----------
    fluxes : array
        units of counts s-1
    errs : array
    expts : array
        Exposure times.
    sigma_clip : float
        Exclude fluxes > this many sigma relative to their measurement error
        from median flux.
    Returns
    -------
    coadd_flux, coadd_err : floats
        Coadded flux and error or upper limit (flux = -999, error = upper
        limit).  units of counts s-1

    """
    # if not fluxes, return null
    if len(fluxes) == 0:
        return -999., -999.

    fluxes, errs, expts = map(np.asarray, (fluxes, errs, expts))

    # if all measurements are upper limits just pick most restrictive limit
    if np.all(fluxes <= -99.):
        return -999., np.min(errs)

    # compute coadd
    else:
        def apply_filter(keep):
            return [a[keep] for a in (fluxes, errs, expts)]

        # filter out negative fluxes
        keep = fluxes > 0
        fluxes, errs, expts = apply_filter(keep)

        # filter out outliers
        if sigma_clip:
            median = np.median(fluxes)
            keep = (fluxes - median)/errs < sigma_clip
            fluxes, errs, expts = apply_filter(keep)

        if len(fluxes) == 0:
            return -999., -999.

        # compute average flux weighted by exposure time and error
        coadd_flux = np.sum(fluxes*expts)/np.sum(expts)
        coadd_err = np.sqrt(np.sum((errs*expts)**2))/np.sum(expts)

        return coadd_flux, coadd_err


def extract_and_coadd(ra, dec, pm_ra, pm_dec, match_radius=4./3600.,
                      search_radius=25./60, sigma_clip=None, query_timeout=60.,
                      upper_limits=True, return_exps=False):
    """
    The top-level function of this module, extract_and_coadd finds sources in
    GALEX archive matching the target while accounting for its proper motion
    between observing visits, the coadds the fluxes from each visit.

    Parameters
    ----------
    ra : float
        Right ascencion of target in decimal degrees.
    dec : float
        Declination of target in decimal degrees.
    pm_ra : float
        Right ascencion proper motion of target in mas/yr.
    pm_dec : float
        Declination proper motion of target in mas/yr.
    match_radius : float
        Radius within which to consider a GALEX source a match to the target
        in degrees. For reference, the 1-sigma astrometric uncertainty is 0.4
        arcseconds for GALEX.
    search_radius : float
        Radius in  which  to query the MCAT in degrees. If upper limits are
        desired, this should  be large enough for the MCAT to return results
        whenever exposures were taken near enough that the target could have
        been in the aperture.
    sigma_clip : float
        Exclude fluxes > this many sigma from median flux relative to their
        measurement error. Careful with this. Stars show real variability
        that is often well beyond measurement errors, so it is probably
        unwise to sigma clip in most cases.
    query_timeout : float
        Seconds to wait for server to respond before giving up.
    upper_limits : bool
        Estimate upper limits for exposures where there is no match for the
        source.
    return_exps : bool
        If True, return all the data provided by extract_source.

    Returns
    -------
    nuv_coadd : tuple
        Coadded flux and error in counts s-1 and, optionally, exposure info returned by
        extract_source. Upper limits show up as -999 for the flux with a positive error.
    fuv_coadd : tuple
        As above, for fuv.

    """
    data = extract_source(ra, dec, pm_ra, pm_dec, match_radius,
                          search_radius, query_timeout, upper_limits)
    nuv_data, fuv_data = data

    if len(nuv_data[0]) == 0:
        nuv = -999., -999.
    else:
        nuv = list(coadd_fluxes(*nuv_data[:3], sigma_clip=sigma_clip))
    if len(fuv_data[0]) == 0:
        fuv = -999., -999.
    else:
        fuv = list(coadd_fluxes(*fuv_data[:3], sigma_clip=sigma_clip))
    if return_exps:
        nuv.append(nuv_data)
        fuv.append(fuv_data)
    nuv, fuv = map(tuple, (nuv, fuv))
    return (nuv, fuv)


def nonlinearity_correction(cps, err, band):
    """
    Correct GALEX fluxes for the detector nonlinearity at high fluxes.

    A good rule of thumb is not to trust correction when error following
    correction is >3x the error prior to correction. Instead, in these cases
    the uncorrected flux should be taken as a lower limit.

    Parameters
    ----------
    cps : array
        Count rate in counts s-1.
    err : array
        Error on count rate.
    band : 'fuv' or 'nuv'

    Returns
    -------
    corrected_cps : array
        Nonlinearity-corrected count rate.
    corrected_err : array
        Error on the above.
    """

    scalar_input = False
    if np.isscalar(cps):
        scalar_input = True
        cps, err = [np.array([a]) for a in (cps, err)]

    # I fit these from the plot for the APER_7 curve in Morrissey+ 2007
    # because APER_7 is what the rest of this code uses
    # they are quadratic terms for the function
    # MR = a * PR**2 + b * PR + c
    # where MR stands for measured rate, PR stands for predicted rate
    if band.lower() == "fuv":
        a, b, c0 = -0.23428377,  1.69610651, -0.51978853
    elif band.lower() == "nuv":
        a, b, c0 = -0.15198511,  1.60029826, -0.621034
    else:
        raise ValueError("Band must be NUV or FUV.")

    # find where the corrected values are closest to the measured values
    logPR0 = np.sqrt(c0/a)
    logMR0 = a*logPR0**2 + b*logPR0 + c0

    # use quadratic eqn to get PR from MR
    logMR = np.log10(cps)
    c = c0 - logMR
    logPR = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    PR = 10**logPR
    dlogPR_dc = 1./np.sqrt(b**2 - 4*a*c)
    dc_dPR = 1/np.log(10)/cps

    # compute error (does not include  error on quadratic fit)
    PRerr = 10**logPR*np.log(10)*dlogPR_dc*dc_dPR*err

    # use corrected rate for all values in the range where the curve fits
    # begin to show an increasing deviation of MR from PR
    correct = logMR > logMR0
    PR[~correct] = cps[~correct]
    PRerr[~correct] = err[~correct]

    # make output scalar again, if input was scalar
    if scalar_input:
        return tuple(map(np.squeeze, (PR, PRerr)))
    return PR, PRerr


def galex_cps2mag(cps, band):
    scale = 18.82 if band == 'fuv' else 20.08
    with np.errstate(invalid='ignore'):
        mag = -2.5 * np.log10(cps) + scale
    return mag


def cps2magerr(cps, cpserr):
    # mag = -2.5*log10(cps) + mag0
    # dmag/dcps = -2.5 * 1/(cps * ln(10))
    # magerr = 2.5/ln(10) * cpserr/cps
    return 2.5/np.log(10) * cpserr/cps
