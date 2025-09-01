from datetime import datetime
import warnings

from astropy import table
import numpy as np

import paths
import catalog_utilities as catutils
import database_utilities as dbutils

from stage1_processing import visit_status_xml_parser as xparse


#%% target properties

planets = catutils.load_and_mask_ecsv(paths.selection_intermediates / 'chkpt8__target-build.ecsv')
hosts = catutils.planets2hosts(planets)
# convert to QTables
with catutils.catch_QTable_unit_warnings():
    planets = table.QTable(planets)
    hosts = table.QTable(hosts)
planets.add_index('tic_id')
hosts.add_index('tic_id')


#%% observation progress table
_path_progress = dbutils.pathname_max(paths.status_input, 'Observation Progress*.xlsx')
progress_table = catutils.read_excel(_path_progress)
progress_table.add_index('Target')

progress_table = progress_table
for col in progress_table.colnames:
    valid = progress_table[col] != ''
    if not np.any(valid):
        continue
    test_value = progress_table[col][valid][0]
    if isinstance(test_value, datetime):
        progress_table[col][valid] = [x.strftime("%b %d, %Y") for x in progress_table[col][valid]]


#%% stela name map
stela_names = table.Table.read(paths.locked / 'stela_names.csv')
stela_names.add_index('tic_id')
stela_names.add_index('hostname')
stela_names.add_index('hostname_file')
stela_names.add_index('hostname_hst')


#%% visit status

_latest_status_path = dbutils.pathname_max(paths.status_input, 'HST-17804-visit-status*.xml')
visit_status = xparse.load_visit_status_xml_as_table(_latest_status_path)