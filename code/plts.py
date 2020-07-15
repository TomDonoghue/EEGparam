"""Plotting functions for EEG-FOOOF analysis."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, sem

from fooof.core.funcs import gaussian_function, expo_nk_function

from utils import get_intersect, get_pval_shades, calc_ap_comps
from settings import *

###################################################################################################
###################################################################################################

def plot_comp_boxplot(data, save_fig=False, save_name=None):
    """Plot comparison between groups, as a boxplot.

    `data` should be a 1d array, with data that can be split up by YOUNG & OLD _INDS.
    """

    # Initialize figure & set settings
    fig, ax = plt.subplots(figsize=[2, 4])
    lw = 2

    # Create the plot
    bplot = plt.boxplot([data[YNG_INDS], data[OLD_INDS]], whis=1.0, widths=0.2, showfliers=False,
                        boxprops={'linewidth': lw}, capprops={'linewidth': lw},
                        whiskerprops={'linewidth': lw},
                        medianprops={'linewidth': lw, 'color':'black'},
                        patch_artist=True,
                        labels=['Young', 'Old'])

    # Set colors of the boxes
    colors = [YNG_COL, OLD_COL]
    for box, color in zip(bplot['boxes'], colors):
        box.set_facecolor(color)

    # Set tick font-sizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    _set_lr_spines(ax, 2)
    _save_fig(save_fig, save_name)


def plot_comp(data, save_fig=False, save_name=None):
    """Plot comparison between groups, as a mean value with an errorbar.

    `data` should be a 1d array, with data that can be split up by YOUNG & OLD _INDS.
    """

    fig, ax = plt.subplots(figsize=[2, 4])

    # Split up data
    you_data = data[YNG_INDS][~np.isnan(data[YNG_INDS])]
    old_data = data[OLD_INDS][~np.isnan(data[OLD_INDS])]

    means = [np.mean(you_data), np.mean(old_data)]
    sems = [sem(you_data), sem(old_data)]

    plt.errorbar([1, 2], means, yerr=sems, xerr=None, fmt='.',
                 markersize=22, capsize=10, elinewidth=2, capthick=2)

    ax.set_xlim([0.5, 2.5])
    plt.xticks([1, 2], ['Young', 'Old'])

    _set_lr_spines(ax, 2)
    _save_fig(save_fig, save_name)


def plot_comp_scatter(data, label=None, save_fig=False, save_name=None):
    """Create a scatter plot comparing the two groups."""

    fig, ax = plt.subplots(figsize=[2, 4])

    x1, x2 = 0.5, 1.5
    d1 = data[YNG_INDS]
    d2 = data[OLD_INDS]

    # Create x-axis data, with small jitter for visualization purposes
    x_data_1 = np.ones_like(d1) * x1 + np.random.normal(0, 0.025, d1.shape)
    x_data_2 = np.ones_like(d2) * x2 + np.random.normal(0, 0.025, d2.shape)

    ax.scatter(x_data_1, d1, s=36, alpha=0.5, c=YNG_COL)
    ax.plot([x1-0.2, x1+0.2], [np.mean(d1), np.mean(d1)], lw=5, c=YNG_COL)

    ax.scatter(x_data_2, d2, s=36, alpha=0.5, c=OLD_COL)
    ax.plot([x2-0.2, x2+0.2], [np.mean(d2), np.mean(d2)], lw=5, c=OLD_COL)

    if label:
        ax.set_ylabel(label, fontsize=16)

    plt.xlim([0, 2])

    plt.xticks([x1, x2], ['Young', 'Old'])

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    _set_lr_spines(ax)
    _save_fig(save_fig, save_name)


def plot_behav_loads(loads_data, save_fig=False, save_name=None):
    """Plot behavioural performance across loads.

    loads_data : list of summary data across behavioural loads.
    """

    fig, ax = plt.subplots(figsize=[6, 4])

    plt.plot([1, 2, 3], [data['yng_mean'] for data in loads_data],
             '.', markersize=15, label='YNG', c=YNG_COL)
    plt.plot([1, 2, 3], [data['old_mean'] for data in loads_data],
             '.', markersize=15, label='OLD', c=OLD_COL)
    plt.legend()

    plt.xlabel('Load', fontsize=14)
    plt.xticks([1, 2, 3], ['1', '2', '3'])
    plt.ylabel("d'", fontsize=14);

    _set_lr_spines(ax, lw=2)
    _save_fig(save_fig, save_name)


def plot_oscillations(alphas, save_fig=False, save_name=None):
    """Plot a group of (flattened) oscillation definitions."""

    n_subjs = alphas.shape[0]

    # Initialize figure
    fig, ax = plt.subplots(figsize=[6, 6])

    # Get frequency axis (x-axis)
    fs = np.arange(6, 16, 0.1)

    # Create the oscillation model from parameters
    osc_psds = np.empty(shape=[n_subjs, len(fs)])
    for ind, alpha in enumerate(alphas):
        osc_psds[ind, :] = gaussian_function(fs, *alphas[ind, :])

    # Plot each individual subject
    for ind in range(n_subjs):
        ax.plot(fs, osc_psds[ind, :], alpha=0.3, linewidth=1.5)

    # Plot the average across all subjects
    avg = np.nanmean(osc_psds, 0)
    ax.plot(fs, avg, 'k', linewidth=3)

    ax.set_ylim([0, 1.75])

    ax.set_xlabel('Frequency', {'fontsize': 14})
    ax.set_ylabel('Power', {'fontsize': 14})

    # Set tick font-sizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    _set_lr_spines(ax, 2)
    _save_fig(save_fig, save_name)


def plot_aperiodic(aps, control_offset=False, save_fig=False, save_name=None, return_vals=False):
    """Plot aperiodic components, comparing between groups."""

    n_subjs = aps.shape[0]

    # Set offset to be zero across all PSDs
    taps = np.copy(aps)
    if control_offset:
        taps[:, 0] = 1

    fig, ax = plt.subplots(figsize=[8, 6])

    # Get frequency axis (x-axis)
    fs = np.arange(1, 45, 0.1)

    # Create the aperiodic model from parameters
    ap_psds = np.empty(shape=[n_subjs, len(fs)])
    for ind, ap in enumerate(taps):
        ap_psds[ind, :] = expo_nk_function(fs, *taps[ind, :])

    # Set whether to plot x-axis in log
    plt_log = False
    fs = np.log10(fs) if plt_log else fs

    # Plot each individual subject
    for ind in range(n_subjs):
        lc = YNG_COL if ind in YNG_INDS else OLD_COL
        ax.plot(fs, ap_psds[ind, :], lc, alpha=0.2, linewidth=1.8)

    # Plot the average across all subjects, split up by age group
    you_avg = np.mean(ap_psds[YNG_INDS, :], 0)
    old_avg = np.mean(ap_psds[OLD_INDS, :], 0)
    ax.plot(fs, you_avg, YNG_COL, linewidth=4, label='Young')
    ax.plot(fs, old_avg, OLD_COL, linewidth=4, label='Old')

    # Shade regions of significant difference
    avg_diffs, p_vals = calc_ap_comps(fs, ap_psds)
    sh_starts, sh_ends = get_pval_shades(fs, p_vals)
    _plt_shade_regions(sh_starts, sh_ends)

    # Plot limits & labels
    ax.set_xlim([min(fs)-0.5, max(fs) + 0.5])
    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight':'bold'})
    ax.set_ylabel('Power', {'fontsize': 14, 'fontweight':'bold'})

    # Set tick font-sizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    _set_lr_spines(ax, 2)
    plt.legend()
    _save_fig(save_fig, save_name)

    if return_vals:
        return fs, avg_diffs, p_vals


def plot_ap_band_diff(freqs, avg_diffs, p_vals, save_fig=False, save_name=None):
    """Plot a comparison between specific frequency values from generated aperiodic components.

    freqs : 1d array of frequency values
    avg_diffs : 1d array of differences per frequency value (same length as freqs)
    p_vals : 1d array of p-values for each comparison  (same length as freqs)
    """

    fig, ax = plt.subplots(figsize=[8, 6])

    ax.set_xlim([0, max(freqs) + 0.5])
    ax.set_ylim([-0.75, 0.75])

    plt.plot(freqs, avg_diffs, '.', markersize=12)

    plt.title('\'Periodic\' Differences from Aperiodic Component',
              {'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight': 'bold'})
    ax.set_ylabel('Difference in Power (au)', {'fontsize': 14, 'fontweight': 'bold'})

    # Set tick font-sizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Add shading for statistically significant different regions
    sh_starts, sh_ends = get_pval_shades(freqs, p_vals)

    _set_lr_spines(ax, 2)
    _plt_shade_regions(sh_starts, sh_ends)
    _save_fig(save_fig, save_name)


def plot_overlap(m1, m2, std1, std2, col='#2ba848', save_fig=False, save_name=None):
    """Visualize the overlap of two gaussians.

    m1 & std1 : define the average alpha
    m2 & std2 : define the canonical alpha
    """

    # Initialize plot & settings
    fig, ax = plt.subplots(figsize=[5, 6])
    ax.set_xlim([0, 20.])
    ax.set_ylim([0, 0.21])

    # Set up for x-axis
    x_vals = np.linspace(0, 20, 1000)
    step = x_vals[1] - x_vals[0]

    # Plot the gaussians
    plot1 = plt.plot(x_vals, norm.pdf(x_vals, m1, std1), 'grey', lw=2.5, label='Average')
    plot2 = plt.plot(x_vals, norm.pdf(x_vals, m2, std2), 'black', lw=2.5, label='Canonical')

    # Get point of overlap
    r_pt = get_intersect(m1, m2, std1, std2)

    # Shade in overlapping areas
    alpha = 0.6
    if m1 < m2:
        _ = plt.fill_between(x_vals[x_vals > r_pt - step], 0,
                             norm.pdf(x_vals[x_vals > r_pt - step], m1, std1),
                             alpha=alpha, color=col, lw=0)
        _ = plt.fill_between(x_vals[x_vals < r_pt], 0,
                             norm.pdf(x_vals[x_vals < r_pt], m2, std2),
                             alpha=alpha, color=col, lw=0)
        _ = plt.fill_between(x_vals[x_vals < r_pt], norm.pdf(x_vals[x_vals < r_pt], m2, std2),
                             norm.pdf(x_vals[x_vals < r_pt], m1, std1),
                             alpha=alpha, color='#d10c29', lw=0)
    else:
        _ = plt.fill_between(x_vals[x_vals < r_pt + step], 0,
                             norm.pdf(x_vals[x_vals < r_pt + step], m1, std1),
                             alpha=alpha, color=col, lw=0)
        _ = plt.fill_between(x_vals[x_vals > r_pt], 0,
                             norm.pdf(x_vals[x_vals > r_pt], m2, std2),
                             alpha=alpha, color=col, lw=0)
        _ = plt.fill_between(x_vals[x_vals > r_pt], norm.pdf(x_vals[x_vals > r_pt], m2, std2),
                             norm.pdf(x_vals[x_vals > r_pt], m1, std1),
                             alpha=0.6, color='#d10c29', lw=0)

    _set_lr_spines(ax, 2)
    _save_fig(save_fig, save_name)

###################################################################################################
###################################################################################################

def _save_fig(save_fig, save_name):
    """Save out current figure."""

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=600)


def _set_lr_spines(ax, lw=None):
    """Set the spines to drop top & right box & set linewidth."""

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    if lw:
        ax.spines['left'].set_linewidth(lw)
        ax.spines['bottom'].set_linewidth(lw)


def _plt_shade_regions(shade_starts, shade_ends):
    """Shade in regions of a plot."""

    for sh_st, sh_en in zip(shade_starts, shade_ends):
        plt.axvspan(sh_st, sh_en, color='r', alpha=0.2, lw=0)
