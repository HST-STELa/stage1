from astropy import table
import pandas as pd
import numpy as np

import paths

mamajek = table.Table.read(paths.reference_tables / 'mamajek_SpT_table.ecsv')
catsup = table.Table.read(paths.reference_tables / 'catsup.vot')
ism_columns = table.Table.read(paths.reference_tables / 'redfield_N_H_columns.ecsv')
lp22 = table.Table.read(paths.reference_tables / 'lp22 m-r water worlds.csv', names=['m', 'r'], data_start=0)

_etc_filename = 'g140m_countrates_1700_1e-13.csv'
g140m_etc_countrates = table.Table.read(paths.etc / _etc_filename)
_pieces = _etc_filename.split('_')
g140m_flux_ref = float(_pieces[-1][:-4])
g140m_expt_ref = float(_pieces[-2])

etc_acq_times = table.Table.read(paths.etc / 'ACQ.csv')
etc_g140m_times = table.Table.read(paths.etc / 'G140M.csv')
etc_g140l_times = table.Table.read(paths.etc / 'G140L.csv')
etc_e140m_times = table.Table.read(paths.etc / 'E140M.csv')


# M-dwarf isr clearance table
# this table will need to be updated whenever the pipeline hits targets it wants to include in observations
# that aren't already in the table
mdwarf_isr = pd.read_excel(paths.checked / 'mdwarf_isr_continuously_updated.xlsx', header=1)
i_footer, = np.nonzero(mdwarf_isr['Target'].values == 'EXAMPLES BELOW')
mdwarf_isr = mdwarf_isr[1:i_footer[0]]
mdwarf_isr = table.Table.from_pandas(mdwarf_isr)
mdwarf_isr.add_index('Target')

