import warnings

import h5py
import numpy as np
from astropy.table import Table
from astropy import units as u

import paths

from lya_prediction_tools import ism
from lya_prediction_tools import lya

from stage1_processing import target_lists
from stage1_processing import preloads
from stage1_processing import processing_utilities as pu

#%%

host_catalog = preloads.hosts.copy()
planet_catalog = preloads.planets.copy()
stela_name_tbl = preloads.stela_names.copy()

temp_simfile, = list(paths.data_targets.rglob(f'hd149026*outflow-tail-model*transit-b.h5'))
with h5py.File(temp_simfile) as f:
    default_sim_wavgrid = f['wavgrid'][:] * 1e8

#%%
targets = target_lists.eval_no(1)

#%%
for target in targets:
    targfolder = paths.target_data(target)

    tic_id, hostname = stela_name_tbl.loc['hostname_file', target][['tic_id', 'hostname']]
    host = host_catalog.loc[tic_id]
    i_planet = planet_catalog.loc_indices[tic_id]
    i_planet = np.atleast_1d(i_planet)
    planets = planet_catalog[i_planet]

    rv_star_kms = host['st_radv']
    lya_reconstruction_file, = targfolder.rglob('*lya-recon*')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='OverflowError converting to FloatType in column')
        lya_recon = Table.read(lya_reconstruction_file)
    flya = lya_recon['lya_model unconvolved_median']
    wlya = lya_recon['wave_lya']
    vlya_sys = lya.w2v(wlya) - rv_star_kms

    # find the blue wing peak
    rv_ism_kms = ism.ism_velocity(host['ra']*u.deg, host['dec']*u.deg).to_value('km s-1')
    v0_ism_sys = rv_ism_kms - rv_star_kms
    blue_lya = vlya_sys < v0_ism_sys
    imx = np.argmax(flya[blue_lya])
    vpk = vlya_sys[imx]

    for planet in planets:
        letter = planet['pl_letter']

        transit_simulation_file, = targfolder.rglob(f'*outflow-tail-model*transit-{letter}.h5')

        # load in the transit models
        with h5py.File(transit_simulation_file) as f:
            transit_timegrid = f['tgrid'][:]
            transit_wavegrid_sys = f['wavgrid'][:] * 1e8
            if np.all(transit_wavegrid_sys == 0):
                transit_wavegrid_sys = default_sim_wavgrid
            transmission_array = f['intensity'][:]
            eta_sim = f['eta'][:]
            wind_scaling_sim = f['mdot_star_scaling'][:]
            phion_scaling_sim = f['phion_scaling'][:]
            params = dict(f['system_parameters'].attrs)



        labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1]'.split(',')
        phion = params['phion_rate'] * phion_scaling_sim['phion_scaling']
        Tion = 1 / phion * u.s
        lTion = np.log10(Tion.to_value('h'))
        Mdot = params['mdot_star'] * wind_scaling_sim['wind scaling']
        lMdot = np.log10(Mdot)
        eta = eta_sim
        leta = np.log10(eta)
        param_vecs = [leta, lTion, lMdot]
        cfig, _ = pu.make_hist_corner(param_vecs, snr_vec, labels=labels)

        title = f'{hostname} {letter}'
        cfig.suptitle(title)
        name = transit_simulation_file.name.replace('.h5', f'.plot-tailiness.ecsv')
        cfig.savefig(targ_transit_folder / cname_pdf)
        cfig.savefig(targ_transit_folder / cname_png, dpi=300)

        plt.close('all')
