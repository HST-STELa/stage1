from astropy import table

#%% set up table

MC = table.MaskedColumn
linecat = table.Table()
linecat['key'] = MC(dtype='object', mask=[])
linecat['name'] = MC(dtype='object', mask=[])
linecat['wave'] = MC(dtype='float', mask=[])
linecat['Tform'] = MC(dtype='float', mask=[])

#%% add a few example lines

row1 = dict(key='c4', name='C IV', wave=1548.2043, Tform=4.95)
row2 = dict(key='c4', name='C IV', wave=1550.784, Tform=4.95)
linecat.add_row(row1)
linecat.add_row(row2)

#%% save
"""enter the rest directly into the saved file"""

linecat.write('reference_files/fuv_line_list.ecsv')