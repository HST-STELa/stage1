from astropy import table
import pandas as pd
import numpy as np

import paths

mamajek = table.Table.read(paths.reference_tables / 'mamajek_SpT_table.ecsv')
catsup = table.Table.read(paths.reference_tables / 'catsup.vot')
lp22 = table.Table.read(paths.reference_tables / 'lp22 m-r water worlds.csv', names=['m', 'r'], data_start=0)

# M-dwarf isr clearance table
# this table will need to be updated whenever the pipeline hits targets it wants to include in observations
# that aren't already in the table
mdwarf_isr = pd.read_excel(paths.mdwarf_google_sheet_xlsx_export, header=1)
i_footer, = np.nonzero(mdwarf_isr['Target'].values == 'EXAMPLES BELOW')
mdwarf_isr = mdwarf_isr[1:i_footer[0]]
mdwarf_isr = table.Table.from_pandas(mdwarf_isr)
mdwarf_isr.add_index('Target')

stela_names = table.Table.read(paths.stela_name_tbl)