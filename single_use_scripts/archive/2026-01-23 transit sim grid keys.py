import h5py
import paths

#%%

files = list(paths.data_targets.rglob('*.h5'))
files = sorted(files)

for fpath in files:
    with h5py.File(fpath) as f:
        key, = [k for k in f.keys() if 'mdot_star' in k]
        planet = fpath.name.split('.')[0]
        print(f'{planet}: {key}')

len(files)

# outcome: there were only three files that still had "mdot_star_scaling" and they were the ones that had run late