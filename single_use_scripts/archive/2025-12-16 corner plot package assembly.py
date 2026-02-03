import shutil

import paths

package_folder = paths.packages / '2025-12-10.detection_sigma_corner_plots'

#%%

files = list(paths.data_targets.rglob('*corner.png'))

for file in files:
    shutil.copy(file, package_folder / file.name)

#%%

files = list(paths.data_targets.rglob('*[ge]140m*plot.png'))

for file in files:
    shutil.copy(file, package_folder / file.name)


#%%

names = """AU Mic	b
AU Mic	c
GJ 143	b
GJ 3470	b
GJ 357	b
GJ 436	b
HAT-P-11	b
HD 136352	b
HD 136352	c
HD 207897	b
HD 219134	b
HD 219134	c
HD 42813	b
HD 42813	c
HD 42813	d
HD 63433	b
HD 63433	c
HD 63433	d
HD 63935	b
HD 63935	c
HD 73583	b
HD 95338	b
HD 97658	b
Kepler-444	c
Kepler-444	d
Kepler-444	e
Kepler-444	f
L 98-59	c
L 98-59	d
LHS 1140	c
LTT 1445 A	b
LTT 1445 A	c
TOI-1231	b
TOI-1434	1
TOI-1467	b
TOI-1695	b
TOI-1759	b
TOI-1774	1
TOI-178	c
TOI-178	d
TOI-178	g
TOI-2015	b
TOI-2076	c
TOI-2079	2
TOI-2134	b
TOI-2194	b
TOI-2285	b
TOI-2443	b
TOI-2459	b
TOI-421	b
TOI-421	c
TOI-5789	1
TOI-6850	1
Wolf 503	b"""

names = names.split('\n')
names = [n.replace('\t', '-') for n in names]
names = [n.lower() for n in names]

best_target_folder = package_folder / 'det vol above 1pct'
for name in names:
    files1 = list(package_folder.glob(f"{name.replace(' ', '-')}*"))
    files2 = list(package_folder.glob(f"{name.replace(' ', '')}*"))
    files3 = list(package_folder.glob(f"{name.replace(' ', '-')[:-2]}*plot.png"))
    files4 = list(package_folder.glob(f"{name.replace(' ', '')[:-2]}*plot.png"))
    for f in files1 + files2 + files3 + files4:
        shutil.copy(f, best_target_folder / f.name)