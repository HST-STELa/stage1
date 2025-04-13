from pathlib import Path

inputs = Path('target_selection_data / inputs')
intermediates = Path('target_selection_data / intermediates')
outputs = Path('target_selection_data / outputs')

ipac = inputs / 'exoplanet_archive'
checked = inputs / 'hand_checked'
reference_tables = inputs / 'reference_tables'
reference_spectra = inputs / 'reference_uv_spectra' / 'lya_added_intrinsic'
etc = inputs / 'etc_grids'
other = inputs / 'other'
requested = inputs / 'requested_targets'
hst_observations = inputs / 'hst_observations'
locked = inputs / 'locked_choices'

reports = outputs / 'target_request_reports'
difftbls = outputs / 'diff_tables'