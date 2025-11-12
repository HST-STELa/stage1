from pathlib import Path
import env

data = Path(env.stela_db_root_path)
scratch = Path(env.stela_scratch_path)
stage1_code = Path(env.stage1_code_path)


catalogs = data / 'catalogs'
packages = data / 'packages'
inbox = packages / 'inbox'
data_targets =  data / 'targets'
def target_data(target):
    return data_targets / target
def target_hst_data(target):
    return data_targets / target / 'hst'
def target_epehemerides(target):
    return data_targets / target / 'ephemerides'

selection_inputs = Path('target_selection_data/inputs')
selection_intermediates = Path('target_selection_data/intermediates')
selection_outputs = Path('target_selection_data/outputs')

reference_files = Path('reference_files')
reference_tables = reference_files / 'reference_tables'
reference_spectra = reference_files / 'reference_uv_spectra' / 'lya_added_intrinsic'
stis = reference_files / 'stis'
cos = reference_files / 'cos'
locked = reference_files / 'locked_choices'
uv_lines = reference_files / 'uv_lines'

stela_name_tbl = locked / 'stela_names.csv'

ipac = selection_inputs / 'exoplanet_archive'
checked = selection_inputs / 'hand_checked'
other = selection_inputs / 'other'
requested = selection_inputs / 'requested_targets'
hst_observations = selection_inputs / 'hst_observations' # should probably rename this "hst_database" or something

reports = selection_outputs / 'target_request_reports'
difftbls = selection_outputs / 'diff_tables'

stage1_processing = Path('stage1_processing')
status_input = stage1_processing / 'status_input'
status_output = stage1_processing / 'status_output'
data_reduction_tools = Path('data_reduction_tools')