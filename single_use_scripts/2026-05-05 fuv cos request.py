import numpy as np
from astropy import table

import paths

import database_utilities as dbutils

#%% settings

e140m_line_detection_limit = 7e-15


#%% targets to check

"""
check any in plan with e140m for their obs, plus any outstanding issues josh identified
I just filtered for e140m in the APT, then looked through the associated targets to see if e140m was for lya or fuv
remember if already observed with e140m, new mode would require new obs, of which we only have 2 to allocate
"""

apt_targets = """
toi-1695
toi-620
toi-1730
toi-1231
toi-198
hd5278
toi-1224
k2-72
toi-1467
toi-2094
lp714-47
toi-2015
toi-1452
gj3090
toi-133
toi-2079
k2-25
toi-1224
hd5278
toi-2015
toi-4529
toi-4556
"""
apt_targets = apt_targets.strip().split()
apt_targets = sorted(list(set(apt_targets)))

"""
looking through apt (cmd+F for each target) checking for (2026-05-05)
  targs with only one e140m (lya + fuv) where I want to add an orbit
  with two e140m = accidental duplicate, but now will swap e140m (I hope) for cos
  e140m already completed, want to add an orbit

then checked if those without fuv were a pass (all were lemons)

g = g140m, l=g140l, e = e140m, x = external, - = not planned, √ = completed or scheduled

target      lya fuv notes
--------------------------
gj3090      g√  e
hd5278      e√  e   ! duplicate, swap fuv for new mode
k2-25       x   e   
k2-72       e√  -   X lemon, remove
lp714-47    g√  e
toi-1224    e√  e   ! duplicate, swap fuv for new mode
toi-1231    x   e
toi-133     g√  e
toi-1452    e√  -   X lemon, remove
toi-1467    g√  e
toi-1695    g√  e   note first fuv failed, waiting on repeat
toi-1730    g√  e
toi-198     g   e
toi-2015    e√  e   !
toi-2079    e√  l   X fuv uses g140l as desired, remove
toi-2094    e   -   X lemon, remove
toi-4529    g   e
toi-4556    e   e   ! duplicate
toi-620     g√  e   
"""

remove = (
    'k2-72',
    'toi-1452',
    'toi-2079',
    'toi-2094',
)

josh_outstanding = """
kepler-444
lp714-47
hd5278
"""
josh_outstanding = josh_outstanding.strip().split()

"""
not included but flagged by josh:
toi-1696 lemon
toi-2128 companion clears
wasp-8 companion clears
toi-870 clears with ism absorption
"""

targets = (set(apt_targets) | set(josh_outstanding)) - set(remove)
targets = sorted(list(targets))

#%% load line lists and mark targets where we can keep e140m

assert np.all(np.isin(targets, dbutils.stela_name_tbl['hostname_file']))

mainlines = "Si III, N V, C II, Si IV".split(', ')

notbl = []
use_e140m = []
for target in targets:
    linepth = paths.target_data(target) / f'{target}.line-flux-table.ecsv'
    if not linepth.exists():
        notbl.append(target)
        continue
    linetbl = table.Table.read(linepth)

    linetbl.add_index('name')
    linefluxes = [np.sum(linetbl.loc[name]['flux']) for name in mainlines]
    if np.all(np.array(linefluxes) > e140m_line_detection_limit):
        use_e140m.append(target)

try_cos = set(targets) - set(use_e140m)
try_cos = sorted(list(try_cos))

for n in try_cos:
    print(n)

#%% check cos

"""
run through m dwarf procedure to try out COS G140L for targets for which you cannot use e140m. 
mark final chosen mode with this table.
c4 and si4 fluxes above 1e-12 will not clear cos g140l

at this point I realized that we could get many of these to clear stis/g140l by using the 0.05" slit and
using an nd filter could still be better than g130m in cases that won't clear cos/g140l, so many of the targets
I moved to stis
"""

"""
target	    mode
gj3090	            g130m
hd5278	            g130m
k2-25	            g130m
kepler-444	        g130m
lp714-47	        g140l
toi-1224	        g130m
toi-1231	        g140l
toi-133	            g130m
toi-1467	        g140l
toi-1695	stis	g140l	52x0.05
toi-1730	stis	g140l	52x0.05
toi-198	            g140l
toi-2015	        g130m
toi-4529	
toi-4556	
toi-620	
"""

cos_targets =


##%