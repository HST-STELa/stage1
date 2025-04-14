from pathlib import Path

observations = '/Users/parke/Google Drive/Research/STELa/data/uv_observations'

selection_inputs = Path('target_selection_data/inputs')
selection_intermediates = Path('target_selection_data/intermediates')
selection_outputs = Path('target_selection_data/outputs')

reference_tables = selection_inputs / 'reference_tables'

ipac = selection_inputs / 'exoplanet_archive'
checked = selection_inputs / 'hand_checked'
reference_spectra = selection_inputs / 'reference_uv_spectra' / 'lya_added_intrinsic'
etc = selection_inputs / 'etc_grids'
other = selection_inputs / 'other'
requested = selection_inputs / 'requested_targets'
hst_observations = selection_inputs / 'hst_observations'
locked = selection_inputs / 'locked_choices'

reports = selection_outputs / 'target_request_reports'
difftbls = selection_outputs / 'diff_tables'