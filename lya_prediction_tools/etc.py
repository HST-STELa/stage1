from astropy import table

import paths

_countrate_filename = 'g140m_countrates_1700_1e-13.csv'
g140m_etc_countrates = table.Table.read(paths.etc / _countrate_filename)
_pieces = _countrate_filename.split('_')
g140m_flux_ref = float(_pieces[-1][:-4])
g140m_expt_ref = float(_pieces[-2])

etc_acq_times = table.Table.read(paths.etc / 'ACQ.csv')
etc_g140m_times = table.Table.read(paths.etc / 'G140M.csv')
etc_g140l_times = table.Table.read(paths.etc / 'G140L.csv')
etc_e140m_times = table.Table.read(paths.etc / 'E140M.csv')