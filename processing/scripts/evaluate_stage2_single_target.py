# make a copy of this script in the script_runs folder with the date (and a label, if needed)
# then run that sript. This avoids constant merge conflicts in the Git repo for things like settings
# changes or one-off mods to the script.

# changes that will be resused (bugfixes, feature additions, etc.) should be made to the base script
# then commited and pushed so we all benefit from them

import argparse
from typing import Literal
from pathlib import Path
import os

# Keep direct/scripted runs from spawning extra BLAS/OpenMP threads.
# The Slurm wrapper sets these too; these defaults help when running manually.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from astropy import table
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import paths
import catalog_utilities as catutils
import utilities as utils

from processing import target_lists
from processing import processing_utilities as pu
from processing import transit_evaluation_utilities as tutils

from lya_prediction_tools import stis

#%% global settings

targets = sum([target_lists.eval_no(i) for i in range(1,4)], [])
targets = set(targets) - {'v1298tau'}

tst_types = ('model', 'flat')
sigma_threshold = 1
staging_area = paths.packages / '2026-03-10.stage2.eval3.staging_area'


#%% misc

mpl.use('Agg')
lyarecon_flag_tables = list(paths.inbox.rglob('*lya*recon*/README*'))
min_samples = 5**4 # used as a check later to ensure all grid pts of Ethan's sims were sampled


#%% set assumed observation timing

obstimes = [-22.5, -21., -3., -1.5,  0.,  1.5,  3.] * u.h
exptimes = [2000, 2700, 2000, 2700, 2700, 2700, 2700] * u.s
offsets = range(0, 17)*u.h
max_safe_offset = 3*u.h # offsets we will actually consider at this stage
baseline_exposures = slice(0, 2)
transit_exposures = slice(4, None)
baseline_apertures = dict(g140m='52x0.2')
cos_consideration_threshold_flux = 2e-14


#%% instrument details

grating = 'g140m'
base_aperture = '52x0.2'
all_apertures = '52x0.5 52x0.2 52x0.1 52x0.05'.split()

acq_exptime_guess = 5
def exptime_fn(aperture):
    """nominal exposure times shortened by peakups. accounts for the tradeoff between
    less airglow contamination and shorter exposures."""
    exptimes_mod =  stis.shorten_exposures_by_peakups(
        aperture,
        acq_exptime_guess,
        exptimes,
        visit_start_indices=[0,2]
    )
    return exptimes_mod


#%% ranges within which to search for integration bands that maximize SNR

normalization_search_rvs = ((-400, -150), (150, 400)) * u.km / u.s
search_model_transit_within_rvs = (-150, 50) * u.km / u.s
search_simple_transit_within_rvs = (-150, 100) * u.km / u.s
simple_transit_range = (-150, 100) * u.km / u.s


#%% assumed jitter and rotation variability as a function of Ro

variability_predictor = tutils.VariabilityPredictor(
    Ro_break=0.1,
    jitter_saturation=0.1,
    jitter_Ro1=0.01,
    rotation_amplitude_saturation=0.25,
    rotation_amplitude_Ro1=0.05,
)


#%% planet and host catalogs

with catutils.catch_QTable_unit_warnings():
    planet_catalog = catutils.load_and_mask_ecsv(staging_area / 'planet_catalog.ecsv')
    planet_catalog = table.QTable(planet_catalog)
    host_catalog = catutils.planets2hosts(planet_catalog)
    planet_catalog.add_index('tic_id')
    host_catalog = table.QTable.read(staging_area / 'host_catalog.ecsv')
    host_catalog.add_index('tic_id')


#%% a few loose closures

def get_transit(planet, host, tst_type: Literal['model', 'flat']):
    if tst_type == 'model':
        return tutils.get_transit_from_simulation(host, planet)
    elif tst_type == 'flat':
        transit_flat = tutils.construct_flat_transit(
            planet, host, obstimes, exptimes,
            rv_grid_span=(-500, 500) * u.km / u.s,
            rv_range=simple_transit_range,
        )
        return transit_flat
    else:
        raise ValueError('tst_type not recognized')

def path_snrs(planet, host, tst_type: Literal['model', 'flat']):
    filenamer = tutils.FileNamer(tst_type, planet, host)
    return filenamer.snr_tbl_full

def load_snr_db(planet, host, tst_type: Literal['model', 'flat']):
    path = path_snrs(planet, host, tst_type)
    return tutils.DetectabilityDatabase.from_file(path)

def load_best_snrs(planet, host, tst_type: Literal['model', 'flat']):
    snrs = load_snr_db(planet, host, tst_type)
    best_snrs = snrs.filter_obs_config(aperture='best', offset='best safe')
    best_snrs = best_snrs.clean_duplicates()
    return best_snrs

def get_lya_flux(host):
    lya = host.lya_reconstruction
    Flya = np.trapz(lya.fluxes[0], lya.wavegrid_earth)
    return Flya

def flag_consider_cos(host):
    Flya = get_lya_flux(host)
    return Flya > cos_consideration_threshold_flux

def consrtuct_snr_samplers(host, transit, tst_type):
    if tst_type == 'flat':
        transit_search_rvs = search_simple_transit_within_rvs
    elif tst_type == 'model':
        transit_search_rvs = search_model_transit_within_rvs
    else:
        raise ValueError
    host_variability = tutils.HostVariability(host, variability_predictor)
    get_snr_iterable, get_snr_single = tutils.build_snr_sampler_fns(
        host,
        host_variability,
        transit,
        exptime_fn,
        obstimes,
        baseline_exposures,
        transit_exposures,
        normalization_search_rvs,
        transit_search_rvs
    )
    return get_snr_iterable, get_snr_single



#%% per-target processing functions

build_snrs = tutils.DetectabilityDatabase.build_db_with_nested_offset_aperture_exploration


def _as_bool(value):
    """Parse common command-line truth values."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def target_from_file(target_file, target_index):
    """Return the 1-indexed target from a plain-text target list."""
    target_file = Path(target_file)
    with target_file.open() as fh:
        target_list = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

    if target_index < 1 or target_index > len(target_list):
        raise IndexError(
            f"target_index={target_index} is outside the range 1..{len(target_list)} "
            f"for {target_file}"
        )
    return target_list[target_index - 1]


def write_default_target_file(path="stage2_eval_targets.txt", overwrite=False):
    """Write the currently configured target list to a text file for Slurm arrays."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use overwrite=True to replace it.")
    path.write_text("\n".join(sorted(targets)) + "\n")
    return path


def calculate_snr_tables_for_target(target, overwrite=False, selected_tst_types=tst_types):
    """Calculate and save all per-planet SNR tables for one host target."""
    print(f"[target={target}] Loading host")
    host = tutils.Host(target, host_catalog, planet_catalog)
    consider_cos = flag_consider_cos(host)

    for planet in utils.printprogress(host.planets, 'dbname', prefix='\tplanet '):
        for tst_type in selected_tst_types:
            output_path = path_snrs(planet, host, tst_type)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if Path(output_path).exists() and not overwrite:
                print(f"[{planet.dbname} {tst_type}] SNR table already exists; skipping: {output_path}")
                continue

            print(f"[{planet.dbname} {tst_type}] Building SNR table")
            transit = get_transit(planet, host, tst_type)
            get_snr_iterable, _ = consrtuct_snr_samplers(host, transit, tst_type)

            def build_planet_snrs(grating, base_aperture, all_apertures):
                snrs = build_snrs(
                    get_snr_iterable,
                    grating,
                    base_aperture,
                    all_apertures,
                    offsets,
                    max_safe_offset,
                    verbose=True,
                )
                return snrs

            snrs = build_planet_snrs(grating, base_aperture, all_apertures)

            # Optionally add COS.
            snrs.meta['COS considered'] = consider_cos
            if consider_cos:
                cos_snrs = build_planet_snrs('g130m', 'psa', ['psa'])
                cos_snrs.snrs.meta = {}
                snrs += cos_snrs

            snrs.write(output_path, overwrite=True)
            print(f"[{planet.dbname} {tst_type}] Wrote {output_path}")


def make_diagnostic_plots_for_target(target, selected_tst_types=tst_types):
    """Make diagnostic plots for one host target, using existing SNR tables."""
    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        for tst_type in selected_tst_types:
            filenamer = tutils.FileNamer(tst_type, planet, host)
            transit = get_transit(planet, host, tst_type)
            _, get_snr = consrtuct_snr_samplers(host, transit, tst_type)
            best_snrs = load_best_snrs(planet, host, tst_type)
            best_snrs.snrs = table.QTable(best_snrs.snrs)

            label_case_pairs = [('median', best_snrs.median_case())]
            if tst_type == 'model':
                label_case_pairs.append(('max', best_snrs.best_case()))
            for label, case_snr_row in label_case_pairs:
                wfig, tfig = tutils.make_diagnostic_plots(planet, transit, get_snr, case_snr_row)
                tutils.save_diagnostic_plots(wfig, tfig, label, host, filenamer)

            plt.close('all')


def make_corner_plots_for_target(target):
    """Make detectability-volume and median-SNR corner plots for one host target."""
    labels = 'log10(eta),log10(T_ion)\n[h],log10(Mdot_star)\n[g s-1],log10(M_planet)\n[Mearth],σ_Lya'.split(',')

    host = tutils.Host(target, host_catalog, planet_catalog)
    for planet in host.planets:
        best_snrs = load_best_snrs(planet, host, 'model')
        filenamer = tutils.FileNamer('model', planet, host)

        # Construct parameter vectors.
        lTion = best_snrs['Tion'].quantity.to_value('dex(h)')
        lMdot = best_snrs['mdot_star'].quantity.to_value('dex(g s-1)')
        leta = np.log10(best_snrs['eta'].data)
        lMp = best_snrs['mass'].quantity.to_value('dex(Mearth)')
        lya_sigma = [tutils.LyaReconstruction.lbl2sig[lbl] for lbl in best_snrs['lya reconstruction case']]
        param_vecs = [leta, lTion, lMdot, lMp, lya_sigma]

        snr_vec = best_snrs['transit sigma']

        cfig, _ = pu.detection_volume_corner(param_vecs, snr_vec, snr_threshold=sigma_threshold, labels=labels)
        cfig.suptitle(planet.dbname)
        utils.save_pdf_png(cfig, host.transit_folder / filenamer.det_vol_corner_basename)

        cfig, _ = pu.median_snr_corner(param_vecs, snr_vec, labels=labels)
        cfig.suptitle(planet.dbname)
        utils.save_pdf_png(cfig, host.transit_folder / filenamer.mdn_snr_corner_basename)

        plt.close('all')


def process_target(target, overwrite=False, selected_tst_types=tst_types, diagnostics=True, corners=True):
    """Run all per-target processing that is safe to parallelize across targets."""
    calculate_snr_tables_for_target(target, overwrite=overwrite, selected_tst_types=selected_tst_types)
    if diagnostics:
        make_diagnostic_plots_for_target(target, selected_tst_types=selected_tst_types)
    if corners and 'model' in selected_tst_types:
        make_corner_plots_for_target(target)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-target Stage 2 SNR products. Designed for trivial "
            "parallelization with one target per process / one Slurm array task."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=False)
    target_group.add_argument('--target', help='One target name to process.')
    target_group.add_argument('--target-file', help='Plain-text target list. Use with --target-index.')

    parser.add_argument(
        '--target-index',
        type=int,
        default=None,
        help='1-indexed row in --target-file. Defaults to SLURM_ARRAY_TASK_ID if available.',
    )
    parser.add_argument(
        '--write-target-file',
        metavar='PATH',
        help='Write the configured target list to PATH and exit.',
    )
    parser.add_argument('--overwrite', action='store_true', help='Recalculate existing SNR tables.')
    parser.add_argument(
        '--tst-types',
        nargs='+',
        choices=['model', 'flat'],
        default=list(tst_types),
        help='Transit signal types to calculate.',
    )
    parser.add_argument(
        '--diagnostics',
        type=_as_bool,
        default=True,
        help='Whether to make per-target diagnostic plots after SNR tables are available.',
    )
    parser.add_argument(
        '--corners',
        type=_as_bool,
        default=True,
        help='Whether to make model corner plots after SNR tables are available.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.write_target_file:
        path = write_default_target_file(args.write_target_file, overwrite=args.overwrite)
        print(f"Wrote {path}")
        return

    if args.target_file:
        target_index = args.target_index
        if target_index is None:
            try:
                target_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
            except KeyError as exc:
                raise ValueError(
                    "--target-index was not supplied and SLURM_ARRAY_TASK_ID is not set."
                ) from exc
        target = target_from_file(args.target_file, target_index)
    elif args.target:
        target = args.target
    else:
        raise ValueError("Supply --target, --target-file, or --write-target-file.")

    process_target(
        target,
        overwrite=args.overwrite,
        selected_tst_types=tuple(args.tst_types),
        diagnostics=args.diagnostics,
        corners=args.corners,
    )


if __name__ == '__main__':
    main()
