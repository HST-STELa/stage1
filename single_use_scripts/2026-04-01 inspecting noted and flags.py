
from processing import observation_table as obt

#%%

reasons = obt.inspect_values_of_all_tables('reason unusable')
flags = obt.inspect_values_of_all_tables('flags')
notes = obt.inspect_values_of_all_tables('notes')