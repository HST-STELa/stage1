"""Generate ISR flare ETC input spectra with assumed and reconstructed Lya ISM absorption."""

import numpy as np
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt

import synphot
import pandas as pd

import paths

import catalog_utilities as catutils
from lya_prediction_tools import ism, lya

OUTPUT_DIR = paths.packages / 'isr_spectra'
PLOTS_DIR = OUTPUT_DIR / 'plots'
ISR2017_DIR = paths.reference_files / 'isr2017'

FLUX_THRESHOLDS = (
    ('g140m', 1.9e-10),
    ('g140l', 2.7e-11),
)

W_MIN = 1100.0
W_MAX = 1800.0
DW_COARSE = 0.1
DW_FINE = 0.01

# Fine sampling only in narrow windows around emission lines
LINE_WINDOWS_AA = (
    (1215.7, 2.0),  # Lya
    (1393.8, 2.0),  # Si IV
    (1548.2, 2.0),  # C IV
)


def make_w_grid():
    w_coarse = np.arange(W_MIN, W_MAX, DW_COARSE)
    w_fine_parts = []
    for w0, half_width in LINE_WINDOWS_AA:
        w_fine_parts.append(np.arange(w0 - half_width, w0 + half_width + DW_FINE / 2, DW_FINE))
    w_all = np.sort(np.unique(np.hstack([w_coarse] + w_fine_parts)))
    return w_all * u.AA


W_GRID = make_w_grid()
NH_COLS = [6e16, 1e17] * u.cm**-2
RECON_CASES = ('median', 'high_1sig', 'high_2sig')
CASE_ROW = dict(zip(RECON_CASES, (2, 1, 0)))
REL_TOL = 0.1
FLUX_FLOOR = 1e-20  # erg s-1 cm-2 AA-1

vegaspec = synphot.SourceSpectrum.from_vega()
u_band = synphot.SpectralElement.from_filter('johnson_u')

_stela_names_tbl = None


def get_stela_names():
    global _stela_names_tbl
    if _stela_names_tbl is None:
        _stela_names_tbl = table.Table.read(paths.stela_name_tbl)
        _stela_names_tbl.add_index('tic_id')
        if not catutils.has_index(_stela_names_tbl, 'hostname'):
            _stela_names_tbl.add_index('hostname')
    return _stela_names_tbl


def flare_line(w, w0, flux, fwhm):
    sigma = lya.sig_from_fwhm_wave(lya.wlab_H, fwhm)
    amp = lya.gaussian_amp(flux, sigma)
    return lya.gaussian_profile(w, w0, amp, sigma)


def flare_lya_intrinsic(w, flux):
    return flare_line(w, 1215.7 * u.AA, flux, 0.5 * u.AA)


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
        left=1.0,
        right=1.0,
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
    # CSV export tends to preserve evaluated values more reliably than XLSX export.
    # This sheet uses the legacy ISR column names (e.g. "f(C IV)6") directly.
    # The sheet CSV includes a top grouping row and one units row.
    # Row 1 contains the real column names (starting with "Target").
    df = pd.read_csv(paths.mdwarf_google_sheet_csv_export, header=1, keep_default_na=False)
    isr_table = table.Table.from_pandas(df)

    # Drop any completely blank targets, and drop any footer section if present.
    if 'Target' not in isr_table.colnames:
        raise KeyError(f"Expected 'Target' column, got: {isr_table.colnames}")
    targets = np.array([str(x).strip() for x in isr_table['Target']])
    keep = (targets != '') & (targets != '---')
    if np.any(targets == 'EXAMPLES BELOW'):
        i_footer = np.nonzero(targets == 'EXAMPLES BELOW')[0][0]
        keep[i_footer:] = False
    isr_table = isr_table[keep]

    def _coerce_masked_float_column(tbl, colname):
        col = tbl[colname]
        data = np.empty(len(tbl), dtype=float)
        mask = np.zeros(len(tbl), dtype=bool)

        for i, x in enumerate(col):
            if np.ma.is_masked(x) or x is None:
                data[i] = np.nan
                mask[i] = True
                continue

            # Common spreadsheet placeholders / unit strings
            s = str(x).strip()
            if s in ("", "--", "—", "-", "nan", "NaN", "None"):
                data[i] = np.nan
                mask[i] = True
                continue

            # Allow commas in numbers (e.g. "1,2e-13")
            s = s.replace(",", "")
            try:
                v = float(s)
            except Exception:
                data[i] = np.nan
                mask[i] = True
            else:
                data[i] = v
                mask[i] = not np.isfinite(v)

        tbl.replace_column(colname, table.MaskedColumn(name=colname, data=data, mask=mask, dtype=float))

    for numeric_col in ('U_flare', 'f(C IV)6', 'f(Si IV)6', 'f(Ly a)6,7'):
        if numeric_col in isr_table.colnames:
            _coerce_masked_float_column(isr_table, numeric_col)

    # Hard-fail if the export is clearly broken (e.g. all zeros).
    # This prevents generating spectra with silently invalid line flux inputs.
    def _nonzero_count(colname):
        if colname not in isr_table.colnames:
            return 0
        v = isr_table[colname].filled(np.nan)
        return int(np.sum(np.isfinite(v) & (v != 0.0)))

    n_c4 = _nonzero_count('f(C IV)6')
    n_si4 = _nonzero_count('f(Si IV)6')
    n_lya = _nonzero_count('f(Ly a)6,7')
    n_rows = len(isr_table)
    if n_rows and (n_c4 == 0 or n_si4 == 0):
        raise ValueError(
            "ISR flux table appears to contain no nonzero values for one or more required line columns. "
            f"Nonzero counts (of {n_rows} rows): "
            f"f(C IV)6={n_c4}, f(Si IV)6={n_si4}, f(Ly a)6,7={n_lya}. "
            f"Source: {paths.mdwarf_google_sheet_csv_export!r}. "
            "This usually means the Google Sheets XLSX export is not preserving formula values. "
            "Fix the sheet/export so these columns contain numeric fluxes (not all zeros), then retry."
        )

    return isr_table


def load_target_parameters():
    parameters = catutils.load_and_mask_ecsv(
        paths.selection_intermediates / 'chkpt3__add-archival_obs_counts.ecsv'
    )
    parameters = catutils.planets2hosts(parameters)
    isr_table = load_isr_table()

    # ISR "Target" names follow stela_names hostnames (e.g. TOI-1730), but the exocat
    # may use a different hostname for the same star (e.g. LHS 1903). Match via tic_id.
    stela = get_stela_names()
    if not catutils.has_index(parameters, 'tic_id'):
        parameters.add_index('tic_id')

    tic_ids = []
    not_in_stela = []
    for target in isr_table['Target']:
        try:
            tic_ids.append(int(stela.loc['hostname', target]['tic_id']))
        except KeyError:
            tic_ids.append(None)
            not_in_stela.append(target)
    if not_in_stela:
        print(f'ISR targets not in stela_names ({len(not_in_stela)}): {not_in_stela}')

    in_cat = np.array([t is not None and t in parameters['tic_id'] for t in tic_ids])
    missing = isr_table['Target'][~in_cat]
    if len(missing):
        print(f'ISR targets not in exocat ({len(missing)}): {list(missing)}')
    isr_table = isr_table[in_cat]
    tic_ids = [t for t, keep in zip(tic_ids, in_cat) if keep]
    target_parameters = parameters.loc[tic_ids]
    return table.hstack((target_parameters, isr_table))


def add_ism_rvs(target_parameters):
    ras, decs = target_parameters['ra'].quantity, target_parameters['dec'].quantity
    ism_rvs = u.Quantity([ism.ism_velocity(ra, dec) for ra, dec in zip(ras, decs)])
    target_parameters['ism_radv'] = ism_rvs
    target_parameters['ism_radv'].format = '.2f'
    return target_parameters


def target_hostname_file(tic_id):
    return get_stela_names().loc['tic_id', tic_id]['hostname_file']


def resolve_rv(target):
    cat_rv = target['st_radv']
    valid_cat_rv = not np.ma.is_masked(cat_rv)
    valid_xls_rv = False
    xls_rv = None
    if 'Stellar RV' in target.colnames:
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


def load_recon_ism_cases(tic_id, verbose=True):
    """Return list of (case_label, wave_recon, T_recon, h1_col) or empty if no recon file."""
    dbname = target_hostname_file(tic_id)
    recon_folder = paths.target_data(dbname) / 'reconstructions'
    files = list(recon_folder.rglob('*lya-recon.csv'))
    if not files:
        if verbose:
            print(f'{dbname} No lya recon file.')
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


def plot_target_spectra(name_for_plot, name_for_file, curves, wide=(1100, 1700), lya_zoom=(1190, 1240)):
    """curves: list of (label, w_AA, flux)"""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, xlim in (('wide', wide), ('lya_zoom', lya_zoom)):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, ww, ff in curves:
            mask = (ww >= xlim[0]) & (ww <= xlim[1])
            ax.plot(ww[mask], ff[mask], label=label, lw=0.8)
        for lbl, y in FLUX_THRESHOLDS:
            ax.axhline(y, color='0.35', lw=1, ls=':', label=lbl)
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
        ax.set_title(name_for_plot)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(xlim)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f'{name_for_file}_{suffix}.png', dpi=150)
        plt.close(fig)


def generate_all_spectra(target_parameters, w=W_GRID):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in target_parameters:
        hostname = target['hostname']
        tic_id = target['tic_id']
        base = target_hostname_file(tic_id)
        rv = resolve_rv(target)

        ybb, yc4, ysi4, Flya = continuum_and_lines(w, target)
        curves = []

        if rv is None:
            print(
                f"No simple ISM spectra generated for {hostname} because there is no stellar RV "
                'measurement in the catalog.'
            )
        else:
            rv_offset = (rv - target['ism_radv']) * u.km / u.s
            for Nh in NH_COLS:
                ylya = flare_lya_with_ISM(w, Flya, rv_offset, Nh)
                y = ybb + yc4 + ysi4 + ylya
                label = f'assumed-Nh{Nh.value:.0e}'
                name = f'{base}_assumed-Nh{Nh.value:.0e}.dat'
                save_spectrum(OUTPUT_DIR / name, w, y)
                curves.append((label, w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1')))

        for case, wave_recon, T_recon, h1_val in load_recon_ism_cases(tic_id):
            ylya = flare_lya_with_recon_ism(w, Flya, wave_recon, T_recon)
            y = ybb + yc4 + ysi4 + ylya
            tag = h1col_tag(h1_val)
            label = f'recon-{case}_h1col{tag}'
            name = f'{base}_recon-{case}_h1col{tag}.dat'
            save_spectrum(OUTPUT_DIR / name, w, y)
            curves.append((label, w.to_value('AA'), y.to_value('erg s-1 cm-2 AA-1')))

        if curves:
            plot_target_spectra(hostname, base, curves)


def _compare_to_reference(w_ref, f_ref, w_model, f_mod, label):
    f_mod_interp = np.interp(w_ref, w_model, f_mod)
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
            w_ref, f_ref, w_aa, f_mod, key
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


if __name__ == '__main__':
    main()
