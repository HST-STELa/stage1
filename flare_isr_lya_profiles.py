"""Generate ISR flare ETC input spectra with assumed and reconstructed Lya ISM absorption."""

import numpy as np
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt

import synphot

import paths

import catalog_utilities as catutils
from lya_prediction_tools import ism, lya

OUTPUT_DIR = paths.packages / 'isr_spectra'
PLOTS_DIR = OUTPUT_DIR / 'plots'
ISR2017_DIR = paths.reference_files / 'isr2017'

W_GRID = np.arange(1100, 1800, 0.1) * u.AA
NH_COLS = [6e16, 1e17] * u.cm**-2
RECON_CASES = ('median', 'high_1sig', 'high_2sig')
CASE_ROW = dict(zip(RECON_CASES, (2, 3, 4)))
REL_TOL = 0.05
FLUX_FLOOR = 1e-20  # erg s-1 cm-2 AA-1

vegaspec = synphot.SourceSpectrum.from_vega()
u_band = synphot.SpectralElement.from_filter('johnson_u')

_stela_names_tbl = None


def get_stela_names():
    global _stela_names_tbl
    if _stela_names_tbl is None:
        _stela_names_tbl = table.Table.read(paths.stela_name_tbl)
        _stela_names_tbl.add_index('hostname')
    return _stela_names_tbl


def flare_line(w, w0, flux, fwhm):
    sigma = lya.sig_from_fwhm_wave(lya.wlab_H, fwhm)
    amp = lya.gaussian_amp(flux, sigma)
    return lya.gaussian_profile(w, w0, amp, sigma)


def flare_lya_intrinsic(w, flux):
    return flare_line(w, 1215.67 * u.AA, flux, 0.5 * u.AA)


def flare_lya_with_ISM(w, flux, rv_offset, Nh):
    Tism = 1e4 * u.K
    intrinsic = flare_lya_intrinsic(w, flux)
    transmitted = lya.transmission(w, rv_offset, Nh, Tism)
    return intrinsic * transmitted


def flare_lya_with_recon_ism(w, flux, wave_recon, T_recon):
    intrinsic = flare_lya_intrinsic(w, flux)
    T = np.interp(
        w.to_value('AA'),
        wave_recon,
        T_recon,
        left=0,
        right=0,
    )
    return intrinsic * T


def normed_bb(w, Umag):
    Tbb = 9000 * u.K
    bb = synphot.models.BlackBody1D(temperature=Tbb)
    bb = synphot.SourceSpectrum(bb)
    bb_normed = bb.normalize(Umag * synphot.units.VEGAMAG, band=u_band, vegaspec=vegaspec)
    y = bb_normed(w, flux_unit='FLAM')
    return y.to('erg s-1 cm-2 AA-1')


def h1col_tag(h1_col):
    return f'{float(h1_col):.2f}'.replace('.', 'p')


def load_isr_table():
    isr_table = catutils.read_excel(paths.mdwarf_google_sheet_xlsx_export, header=1)
    i_footer, = np.nonzero(isr_table['Target'] == 'EXAMPLES BELOW')
    return isr_table[1:i_footer[0]]


def load_target_parameters():
    parameters = catutils.load_and_mask_ecsv(
        paths.selection_intermediates / 'chkpt3__add-archival_obs_counts.ecsv'
    )
    parameters = catutils.planets2hosts(parameters)
    isr_table = load_isr_table()
    parameters.add_index('hostname')
    in_cat = np.isin(isr_table['Target'], parameters['hostname'])
    missing = isr_table['Target'][~in_cat]
    if len(missing):
        print(f'ISR targets not in exocat ({len(missing)}): {list(missing)}')
    isr_table = isr_table[in_cat]
    target_parameters = parameters.loc[isr_table['Target']]
    return table.hstack((target_parameters, isr_table))


def add_ism_rvs(target_parameters):
    ras, decs = target_parameters['ra'].quantity, target_parameters['dec'].quantity
    ism_rvs = u.Quantity([ism.ism_velocity(ra, dec) for ra, dec in zip(ras, decs)])
    target_parameters['ism_radv'] = ism_rvs
    target_parameters['ism_radv'].format = '.2f'
    return target_parameters


def hostname_file(hostname):
    return get_stela_names().loc['hostname', hostname]['hostname_file']


def resolve_rv(target):
    cat_rv = target['st_radv']
    valid_cat_rv = not np.ma.is_masked(cat_rv)
    xls_rv = target['Stellar RV']
    valid_xls_rv = not (np.ma.is_masked(xls_rv) or (xls_rv in ['', '--']))
    if valid_xls_rv:
        xls_rv = float(xls_rv)
    if not valid_cat_rv and not valid_xls_rv:
        return None
    if valid_cat_rv and not valid_xls_rv:
        return cat_rv
    if not valid_cat_rv and valid_xls_rv:
        return xls_rv
    if not np.isclose(cat_rv, xls_rv, atol=5):
        print(
            f'RVs in the exocat and ISR spreadsheet differ by > 5 km s-1 for {target["hostname"]}'
        )
    return cat_rv


def load_recon_ism_cases(hostname):
    """Return list of (case_label, wave_recon, T_recon, h1_col) or empty if no recon file."""
    try:
        dbname = hostname_file(hostname)
    except KeyError:
        return []
    recon_folder = paths.target_data(dbname) / 'reconstructions'
    files = list(recon_folder.rglob('*lya-recon.csv'))
    if not files:
        return []
    lyarecon = table.Table.read(files[0])
    wave = lyarecon['wave_lya'].astype(float)
    h1_col = lyarecon['h1_col value'].astype(float)
    cases = []
    for case in RECON_CASES:
        row = CASE_ROW[case]
        col = f'lya_ism unconvolved_{case}'
        cases.append((case, wave, lyarecon[col].astype(float), h1_col[row]))
    return cases


def continuum_and_lines(w, target):
    fluxcols = ['f(C IV)6', 'f(Si IV)6', 'f(Ly a)6,7']
    Fc4, Fsi4, Flya = [target[key] * u.Unit('erg s-1 cm-2') for key in fluxcols]
    yc4 = flare_line(w, 1548.2 * u.AA, Fc4, 0.2 * u.AA)
    ysi4 = flare_line(w, 1393.8 * u.AA, Fsi4, 0.2 * u.AA)
    ybb = normed_bb(w, target['U_flare'])
    return ybb, yc4, ysi4, Flya


def save_spectrum(path, w, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.array((w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1'))).T
    np.savetxt(str(path), data)


def plot_target_spectra(hostname, curves, wide=(1100, 1700), lya_zoom=(1190, 1240)):
    """curves: list of (label, w_AA, flux)"""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, xlim in (('wide', wide), ('lya_zoom', lya_zoom)):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, ww, ff in curves:
            mask = (ww >= xlim[0]) & (ww <= xlim[1])
            ax.plot(ww[mask], ff[mask], label=label, lw=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
        ax.set_title(hostname)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(xlim)
        fig.tight_layout()
        safe = hostname.replace(' ', '_')
        fig.savefig(PLOTS_DIR / f'{safe}_{suffix}.png', dpi=150)
        plt.close(fig)


def generate_all_spectra(target_parameters, w=W_GRID):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in target_parameters:
        hostname = target['hostname']
        rv = resolve_rv(target)
        if rv is None:
            print(
                f"No spectrum generated for {hostname} because there is no stellar RV "
                'measurement in the catalog.'
            )
            continue

        ybb, yc4, ysi4, Flya = continuum_and_lines(w, target)
        rv_offset = (rv - target['ism_radv']) * u.km / u.s
        curves = []

        for Nh in NH_COLS:
            ylya = flare_lya_with_ISM(w, Flya, rv_offset, Nh)
            y = ybb + yc4 + ysi4 + ylya
            label = f'assumed-Nh{Nh.value:.0e}'
            name = f'{hostname}_assumed-Nh{Nh.value:.0e}.dat'
            save_spectrum(OUTPUT_DIR / name, w, y)
            curves.append((label, w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1')))

        for case, wave_recon, T_recon, h1_val in load_recon_ism_cases(hostname):
            ylya = flare_lya_with_recon_ism(w, Flya, wave_recon, T_recon)
            y = ybb + yc4 + ysi4 + ylya
            tag = h1col_tag(h1_val)
            label = f'recon-{case}_h1col{tag}'
            name = f'{hostname}_recon-{case}_h1col{tag}.dat'
            save_spectrum(OUTPUT_DIR / name, w, y)
            curves.append((label, w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1')))

        if curves:
            plot_target_spectra(hostname, curves)


def _compare_to_reference(w_ref, f_ref, f_mod, label):
    f_mod_interp = np.interp(w_ref, W_GRID.to_value('AA'), f_mod)
    denom = np.maximum(np.abs(f_ref), FLUX_FLOOR)
    rel_err = np.abs(f_mod_interp - f_ref) / denom
    max_err = float(np.nanmax(rel_err))
    passed = max_err <= REL_TOL
    status = 'PASS' if passed else 'FAIL'
    print(f'  {label}: {status} (max relative error = {max_err:.4f})')
    return passed, w_ref, f_ref, f_mod_interp, max_err


def validate_against_isr2017_reference(w=W_GRID, Umag=10):
    """Compare synthetic components to ISR2017 ETC reference CSVs."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    w_aa = w.to_value('AA')

    Fc4 = 1e-12 * u.Unit('erg s-1 cm-2')
    Fsi4 = 5e-13 * u.Unit('erg s-1 cm-2')
    Flya = 1e-11 * u.Unit('erg s-1 cm-2')

    ybb = normed_bb(w, Umag)
    yc4 = flare_line(w, 1548.2 * u.AA, Fc4, 0.2 * u.AA)
    ysi4 = flare_line(w, 1393.8 * u.AA, Fsi4, 0.2 * u.AA)
    ylya = flare_lya_intrinsic(w, Flya)

    builds = {
        'bb': ybb,
        'c4+bb': ybb + yc4,
        'si4+bb': ybb + ysi4,
        'lya+bb': ybb + ylya,
    }
    ref_files = {
        'bb': 'etc bb continuum Umag 10.csv',
        'c4+bb': 'etc c4 1e-12 and bb Umag 10.csv',
        'si4+bb': 'etc si4 5e-13 and bb Umag 10.csv',
        'lya+bb': 'etc lya 1e-11 and bb Umag 10.csv',
    }

    print('ISR2017 reference validation (5% max relative error):')
    all_passed = True
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (key, ref_name) in zip(axes, ref_files.items()):
        ref_path = ISR2017_DIR / ref_name
        w_ref, f_ref = np.loadtxt(ref_path, delimiter=',').T
        f_mod = builds[key].to_value('erg s-1 cm-2 AA-1')
        passed, _, f_ref_plot, f_mod_plot, _ = _compare_to_reference(
            w_ref, f_ref, f_mod, key
        )
        all_passed &= passed
        ax.plot(w_ref, f_ref_plot, 'k-', lw=1, label='reference', alpha=0.7)
        ax.plot(w_ref, f_mod_plot, 'r--', lw=1, label='model')
        ax.set_yscale('log')
        ax.set_title(key)
        ax.set_xlabel('Wavelength (Å)')
        ax.legend(fontsize=8)

    fig.suptitle('ISR2017 validation')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'isr2017_validation.png', dpi=150)
    plt.close(fig)

    overall = 'PASS' if all_passed else 'FAIL'
    print(f'Overall: {overall}')
    return all_passed


def main():
    target_parameters = add_ism_rvs(load_target_parameters())
    generate_all_spectra(target_parameters)
    validate_against_isr2017_reference()


#%% load and join parameter and ISR tables (live Google Sheet)

if __name__ == '__main__':
    main()
else:
    target_parameters = add_ism_rvs(load_target_parameters())


#%% Check to be sure the targets are in the right order

target_parameters['hostname'].pprint(-1)


#%% print stellar RVs to copy and paste into ISR table

target_parameters['st_radv'].pprint(-1)


#%% print ISM RVs to copy and paste into ISR table

target_parameters['ism_radv'].pprint(-1)
