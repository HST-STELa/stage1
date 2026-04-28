
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale
from astropy import units as u
from astropy.io import fits

from lya_prediction_tools import lya

import utilities as utils
import paths
import database_utilities as dbutils

from processing import observation_table as obt



#%% setup

star = 'l98-59'
planet = f'{star}-c'
fd = paths.target_data(star)
obs_month = '2026-04'
tst_window = [-10, 10]

rv = -5.7 * u.km/u.s
rv_kms = rv.to_value('km/s')

v_lim = 300  # km s^-1 symmetric window around stellar Lyα

v_integrate = [-200, -10] * u.km/u.s

#%% files

x1d_files = sorted(fd.glob(f'hst/{star}*g140m*{obs_month}*x1d.fits'))
flt_files = [f.parent / f.name.replace('_x1d', '_flt') for f in x1d_files]


#%% load obs tbl to get phases, filter to obs

obstbl = obt.load_obs_tbl(star)
month_mask = [obs_month in start for start in obstbl['start']]
month_mask = np.array(month_mask)
obstbl = obstbl[month_mask]
obstbl.sort('start')
obstbl.add_index('archive id')

phases = obstbl['phase_c']

# get midpt of transit visit
tstmask = (obstbl['phase_c'] > tst_window[0]) & (obstbl['phase_c'] < tst_window[1])
phmid_visit = (np.min(phases[tstmask]) + np.max(phases[tstmask])) / 2

#%% folder to store transit results

tst_fd = fd / 'transit plots' / f'{planet}.midpt{phmid_visit:.1f}h.{obs_month}'
tst_fd.mkdir(parents=True, exist_ok=True)

#%% utils

def get_phase(filepath, planet='c'):
    obsid = dbutils.parse_filename(filepath)['id']
    phase = obstbl.loc[obsid][f'phase_{planet}']
    return phase


#%% plot flats

flats_dir = tst_fd / 'flats'
flats_dir.mkdir(parents=True, exist_ok=True)

for i, flt_file in enumerate(flt_files):
    obsid = dbutils.parse_filename(flt_file)['id']
    phase = get_phase(flt_file, 'c')
    data = np.asarray(fits.getdata(flt_file, 1), dtype=float)
    z = np.cbrt(data)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        z,
        aspect='auto'
    )
    label = f'{planet}.{i}.midpt{phase:.1f}h.{obsid}'
    plt.title(label)
    plt.tight_layout()
    out_path = flats_dir / f'{label}.png'
    plt.savefig(out_path, dpi=200)
    plt.close()


#%% load spectra

spec_records = []
for x1d_file in x1d_files:
    obsid = dbutils.parse_filename(x1d_file)['id']
    phase = get_phase(x1d_file, 'c')
    data = fits.getdata(x1d_file, 1)
    wave = np.ravel(np.asarray(data['wavelength'], dtype=np.float64))
    flux = np.ravel(np.asarray(data['flux'], dtype=np.float64))
    names = getattr(data.dtype, 'names', None) or ()
    flux_err = None
    for err_key in ('error', 'ERROR', 'STAT_ERR'):
        if err_key in names:
            flux_err = np.ravel(np.asarray(data[err_key], dtype=np.float64))
            break
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if flux_err is not None:
        flux_err = flux_err[order]
    v_kms = lya.w2v(wave) - rv_kms
    in_transit = (phase > tst_window[0]) & (phase < tst_window[1])
    spec_records.append(
        dict(
            file=x1d_file,
            obsid=obsid,
            phase=phase,
            in_transit=in_transit,
            wave=wave,
            flux=flux,
            flux_err=flux_err,
            v_kms=v_kms,
        )
    )

spec_records.sort(key=lambda r: r['phase'])


#%% plot spectra


transit_only = [r for r in spec_records if r['in_transit']]
colors_map = {}
# Sequential colormap for phase (time): warm late / cool early is easy to read in order
_cmap_transit = 'Plasma'
phases_only = np.array([r['phase'] for r in transit_only], dtype=float)
p_lo, p_hi = phases_only.min(), phases_only.max()
denom = (p_hi - p_lo) if np.isfinite(p_hi - p_lo) and (p_hi > p_lo) else 1.0
norm = (phases_only - p_lo) / denom
colors_map = {
    r['obsid']: c
    for r, c in zip(transit_only, sample_colorscale(_cmap_transit, norm.tolist()))
}
first_oid = transit_only[0]['obsid']
last_oid = transit_only[-1]['obsid']

fig = go.Figure()
baseline_legend_done = False
for r in spec_records:
    mask = (r['v_kms'] >= -v_lim) & (r['v_kms'] <= v_lim)
    vx = np.asarray(r['v_kms'][mask], dtype=np.float64)
    fx = np.asarray(r['flux'][mask], dtype=np.float64)
    trace_name = f'phase = {r["phase"]:.2f} h'
    if not r['in_transit']:
        linekws = dict(color='rgba(140,140,140,0.85)', width=1)
        showlegend = not baseline_legend_done
        baseline_legend_done = True
    else:
        linekws = dict(color=colors_map[r['obsid']], width=1.6)
        showlegend = (r['obsid'] == first_oid) or (r['obsid'] == last_oid)
    n_pt = vx.shape[0]
    hover_cd = np.empty((n_pt, 2), dtype=object)
    hover_cd[:, 0] = float(r['phase'])
    hover_cd[:, 1] = r['obsid']

    fig.add_trace(
        go.Scatter(x=vx, y=fx, mode='lines', line_shape='hv',
                   name=trace_name,
                   line=linekws, showlegend=showlegend,
                   hovertemplate=(
                       'phase=%{customdata[0]:.2f} h<br>'
                       'obsid=%{customdata[1]}<br>'
                       'v=%{x:.1f} km/s<br>flux=%{y:.4g}<extra></extra>'
                   ),
                   customdata=hover_cd,
        )
    )

fig.update_layout(
    title=f'{planet} transit ({obs_month})',
    xaxis_title='Velocity (km/s)',
    yaxis_title='Flux',
    xaxis=dict(range=[-v_lim, v_lim]),
    yaxis=dict(tickformat='.2~e'),
    height=560,
    width=1000,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02),
)
spectra_html = tst_fd / f'{planet}.{obs_month}.spectra.html'
fig.write_html(str(spectra_html), include_plotlyjs='cdn')


#%% plot lightcurves

v_int_kms = v_integrate.to_value('km s-1')
w_integrate = lya.v2w(v_int_kms + rv_kms)


def integrated_line_flux(record):
    w, f, e = record['wave'], record['flux'], record['flux_err']
    assert w_integrate[0] > w.min()
    assert w_integrate[1] < w.max()
    fi, ferr = utils.flux_integral(w, f, range=w_integrate, e=e)
    return fi, ferr


def exptime_half_width_phase(obsid):
    ex = obstbl.loc[obsid]['exptime']
    sec = float(ex.to(u.s).value) if hasattr(ex, 'to') else float(ex)
    return (sec / 3600.0) / 2.0


def lc_trace_arrays(rows):
    xs, ys, yerr, xhalf = [], [], [], []
    for r in rows:
        fi, fe = integrated_line_flux(r)
        xs.append(float(r['phase']))
        ys.append(fi)
        yerr.append(fe if np.isfinite(fe) else np.nan)
        xhalf.append(exptime_half_width_phase(r['obsid']))
    return xs, ys, yerr, xhalf


def phase_axis_extent(rows):
    ph = [float(r['phase']) for r in rows]
    if not ph:
        return None
    lo, hi = min(ph), max(ph)
    span = hi - lo
    if span <= 0:
        span = 1.0
    pad = 0.05 * span
    return lo - pad, hi + pad, span + 2 * pad


baseline_lc = [r for r in spec_records if not r['in_transit']]
transit_lc = [r for r in spec_records if r['in_transit']]
eb = phase_axis_extent(baseline_lc)
et = phase_axis_extent(transit_lc)
wb = eb[2] if eb else 1.0
wt = et[2] if et else 1.0
wsum = wb + wt
col_widths = [wb / wsum, wt / wsum]

fig_lc = make_subplots(
    rows=1,
    cols=2,
    column_widths=col_widths,
    subplot_titles=(
        'Baseline',
        'In transit',
    ),
    horizontal_spacing=0.07,
    shared_yaxes=True,
)

xb, yb, yeb, xhb = lc_trace_arrays(baseline_lc)
xt, yt, yet, xht = lc_trace_arrays(transit_lc)

_kw_bl = dict(
    mode='markers',
    marker=dict(size=9, color='rgba(120,120,120,0.95)'),
    error_x=dict(type='data', array=xhb, thickness=1),
    name='baseline',
    showlegend=False,
)
if len(yeb) and np.any(np.isfinite(yeb)):
    _kw_bl['error_y'] = dict(type='data', array=np.asarray(yeb, dtype=np.float64))
fig_lc.add_trace(go.Scatter(x=xb, y=yb, **_kw_bl), row=1, col=1)

_kw_tr = dict(
    mode='markers',
    marker=dict(size=9, color='rgba(120,60,140,0.95)'),
    error_x=dict(type='data', array=xht, thickness=1),
    name='in transit',
    showlegend=False,
)
if len(yet) and np.any(np.isfinite(yet)):
    _kw_tr['error_y'] = dict(type='data', array=np.asarray(yet, dtype=np.float64))
fig_lc.add_trace(go.Scatter(x=xt, y=yt, **_kw_tr), row=1, col=2)

if eb:
    fig_lc.update_xaxes(range=[eb[0], eb[1]], row=1, col=1)
if et:
    fig_lc.update_xaxes(range=[et[0], et[1]], row=1, col=2)

fig_lc.update_xaxes(title_text='Phase (h)', row=1, col=1)
fig_lc.update_xaxes(title_text='Phase (h)', row=1, col=2)
fig_lc.update_yaxes(title_text='integrated flux', row=1, col=1)

fig_lc.update_yaxes(tickformat='.2~e')

fig_lc.update_layout(
    title=(
        f'{planet} integrated line flux ({obs_month}); '
        f'v=[{v_int_kms[0]:.1f}, {v_int_kms[1]:.1f}] km/s (stellar frame)'
    ),
    height=480,
    width=1000,
)
lc_html = tst_fd / f'{planet}.{obs_month}.lightcurve.{v_int_kms[0]:.0f}to{v_int_kms[1]:.0f}kms.html'
fig_lc.write_html(str(lc_html), include_plotlyjs='cdn')

