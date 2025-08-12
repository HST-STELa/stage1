from itertools import combinations

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

import paths
import utilities as utils

def get_intrinsic_lya_flux(target_name_file, return_16_50_84=False):
    recon_folder = paths.target_data(target_name_file) / 'reconstructions'
    lya_file, = recon_folder.rglob('*lya-recon.csv')
    lyarecon = Table.read(lya_file)
    suffixes = ['low_1sig', 'median', 'high_1sig'] if return_16_50_84 else ['median']
    Fs = []
    for suffix in suffixes:
        F = np.trapz(lyarecon[f'lya_intrinsic unconvolved_{suffix}'], lyarecon['wave_lya'])
        Fs.append(F)
    if return_16_50_84:
        return Fs
    else:
        return Fs[0]


def detection_sigma_corner(param_vecs, snr_vec, **hist_corner_kws):
    mean_snr_1d = []
    for x in param_vecs:
        unique_x, inverse_idx = np.unique(x, return_inverse=True)
        edges = utils.mids2bins(unique_x)
        mean_snr = np.bincount(inverse_idx, weights=snr_vec) / np.bincount(inverse_idx)
        mean_snr_1d.append((mean_snr, edges))

    mean_snr_2d = {}

    n = len(param_vecs)
    for i, j in combinations(range(n), 2):
        x, y = param_vecs[i], param_vecs[j]
        edges_x = utils.mids2bins(np.unique(x))
        edges_y = utils.mids2bins(np.unique(y))
        keys = np.column_stack((x, y))
        unique_pairs, inverse_idx = np.unique(keys, axis=0, return_inverse=True)
        n = len(edges_y) - 1
        mean_snr = np.bincount(inverse_idx, weights=snr_vec) / np.bincount(inverse_idx)
        mean_snr = mean_snr.reshape((n,n))
        mean_snr_2d[(j,i)] = (mean_snr, edges_x, edges_y)

    return make_hist_corner(mean_snr_1d, mean_snr_2d, **hist_corner_kws)


def make_hist_corner(
    hist1d,                 # list of length D: each is (counts, edges)
    hist2d,                 # dict with keys (i,j) for i>j: value is (H, xedges, yedges)
    labels=None,
    levels=None,            # list, e.g. [0.68, 0.95] or [z1, z2, ...]
    cmap='magma',
    levels_kws = {},
    figsize=(7,7),
    diagonal_fill=False,
    colorbar_label=None,
):

    D = len(hist1d)
    assert all(isinstance(t, tuple) and len(t)==2 for t in hist1d), "hist1d items must be (counts, edges)"
    if labels is None:
        labels = [f"x{i}" for i in range(D)]

    # Figure + axes grid
    fig, axes = plt.subplots(D, D, figsize=figsize, squeeze=False)
    # consistent z-range across all 2D panels
    H_list = [hist2d[key][0] for key in hist2d if isinstance(key, tuple) and len(key)==2]
    if len(H_list):
        vmin = 0.0
        vmax = max(np.nanmax(H) if np.isfinite(np.nanmax(H)) else 0 for H in H_list)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Draw panels
    last_im = None
    for i in range(D-1,-1,-1): # build it bottom up
        for j in range(D):
            ax = axes[i, j]
            if i < j:
                # upper triangle: hide frame
                ax.set_visible(False)
                continue

            if i == j:
                # 1D histogram on diagonal
                counts, edges = hist1d[i]
                # step histogram (edges length = counts length + 1)
                ax.hist((edges[:-1] + edges[1:]) / 2.0, bins=edges, weights=counts,
                        histtype='step', lw=1.5, color='0.4')
                ax2 = ax.twinx()
                ax2.set_ylim(ax.get_ylim())
                if diagonal_fill:
                    ax.fill_between(edges[:-1], 0, counts, step='post', alpha=0.15, color='k')
            else:
                # lower triangle: 2D histogram image + contours
                H, xedges, yedges = hist2d[(i, j)]
                # Note: pcolormesh expects bin edges; transposed so x along columns, y along rows.
                im = ax.pcolormesh(xedges, yedges, H.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                last_im = im

                if levels is not None:

                    # For contours, use bin centers to avoid off-by-half a bin shifts
                    xc = 0.5 * (xedges[:-1] + xedges[1:])
                    yc = 0.5 * (yedges[:-1] + yedges[1:])
                    # Note the transpose: H.T aligns with (xc, yc)
                    cs = ax.contour(xc, yc, H.T, levels=levels, linewidths=1.0, **levels_kws)
                    ax.clabel(cs)

            if i > 0 and i < j: # sharing along column
                ref_ax = axes[-1, j]
                if ref_ax.has_data():
                    ax.set_xlim(ref_ax.get_xlim())
            if j > 0 and i < j: # sharing along row
                ref_ax = axes[i, 0]
                if ref_ax.has_data():
                    ax.set_ylim(ref_ax.get_ylim())
            if i == j: # sharing along diagonal
                if j == D - 1:
                    xlim = axes[-1,0].get_ylim()
                else:
                    xlim = axes[-1,j].get_xlim()
                ax.set_xlim(xlim)

            # Labels / ticks: corner-style minimalist
            if i == D - 1:
                ax.set_xlabel(labels[j])
            if i < D - 1:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i])
            if j > 0:
                ax.set_yticklabels([])
            if j == 0 and i == 0:
                ax.set_yticklabels([])

    # Global colorbar (if at least one 2D panel was drawn)
    if last_im is not None:
        # colorbar next to the grid
        cbar = fig.colorbar(last_im, ax=axes, orientation='horizontal', fraction=0.03, pad=0.02, location='top',
                            label=colorbar_label)

    # Set axis labels neatly on diagonal x and first-column y
    for k in range(D):
        axes[-1, k].set_xlabel(labels[k])
        axes[k, 0].set_ylabel(labels[k])

    return fig, axes