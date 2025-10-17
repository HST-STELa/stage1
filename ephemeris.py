import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy.table import QTable


earth_center = coord.EarthLocation.from_geocentric(0, 0, 0, unit='m')

def bary_offset(t, star_coord):
    # add this to a geocentric time and you get the barycentric time
    return t.light_travel_time(star_coord, location=earth_center)


class Ephemeris():
    # if you want barcyentric output, give barycentric input
    def __init__(self, tmid, tmid_err, P, P_err, duration):
        self.tmid = tmid
        self.tmid_err = tmid_err
        self.P = P
        self.P_err = P_err
        self.duration = self.T = duration

    @classmethod
    def from_table_row(cls, ephemeris_table_row):
        row = ephemeris_table_row
        qtbl = QTable(row.table)
        qrow = qtbl[row.index]
        eph = Ephemeris(
            tmid=time.Time(qrow['pl_tranmid'], format='jd'),
            tmid_err=qrow['pl_tranmiderr'],
            P=qrow['pl_orbper'],
            P_err=qrow['pl_orbpererr'],
            duration=np.nan,
        )
        return eph

    def transit_offset(self, times):
        transit_time, transit_time_error = self.nearest_transit(times)
        offsets = times - transit_time
        return offsets, transit_time_error
    offset = transit_offset

    def in_transit(self, times):
        off, _ = self.transit_offset(times)
        return np.abs(off.sec) < self.duration.to_value('s')/2

    def phase(self, times):
        toff, err = self.offset(times)
        phase = toff/self.P
        return phase, phase*np.sqrt((err/toff)**2 + (self.P_err/self.P)**2)

    def nearest_transit(self, times):
        Nperiods = (times - self.tmid)/self.P
        Nperiods = np.round(Nperiods.to_value(''))

        transit_time = self.tmid + self.P*Nperiods
        transit_time_error = np.sqrt(self.tmid_err.to_value('s')**2 + (Nperiods*self.P_err.to_value('s'))**2)
        transit_time_error =  time.TimeDelta(transit_time_error, format='sec')

        return transit_time, transit_time_error

    def transit_times(self, time_window):
        Nperiods = (time_window - self.tmid)/self.P
        Nperiods = Nperiods.to_value('')
        Nperiods = np.ceil(Nperiods)
        Nperiods = np.arange(*Nperiods)
        transit_times = self.tmid + self.P*Nperiods
        transit_time_errors = np.sqrt(self.tmid_err**2 + (Nperiods*self.P_err)**2)

        return transit_times, transit_time_errors.to('d')