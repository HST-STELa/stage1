from datetime import datetime

from stage1_processing import preloads


def eval_no(n):
    prog = preloads.progress_table
    mask = prog["Stage 2 Eval\nBatch"] == n
    tics =  prog['TIC ID'][mask]
    selected = preloads.stela_names.loc['tic_id', tics]
    return selected['hostname_file'].tolist()


def observed_since(isot_date_string):
    date = datetime.fromisoformat(isot_date_string)
    status = preloads.visit_status
    fillvalue = datetime.fromisoformat('0001-01-01')
    mask = (status['obsdate'].filled(fillvalue) >= date) & (status['status'] == 'Executed')
    hst_names = status['target'][mask]
    selected = preloads.stela_names.loc['hostname_hst', hst_names]
    return selected['hostname_file'].tolist()
