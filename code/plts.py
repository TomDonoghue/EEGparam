"""Plotting functions for EEG-FOOOF analysis."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, sem
from scipy.stats import norm

from fooof.core.funcs import gaussian_function, expo_nk_function

from utils import get_intersect, get_pval_shades, calc_ap_comps
from settings import *

###################################################################################################
###################################################################################################

def plot_comp_boxplot(dat, save_fig=False, save_name=None):
    """Plot comparison between groups, as a boxplot.

    Dat should be 1D vector, with data that can be split up by YOUNG & OLD _INDS.
    """

    # Initialize figure
    fig, ax = plt.subplots(figsize=[2, 4])

    # Settings
    lw = 2

    # Create the plot
    bplot = plt.boxplot([dat[YNG_INDS], dat[OLD_INDS]], whis=1.0, widths=0.2, showfliers=False,
                        boxprops={'linewidth': lw}, capprops={'linewidth': lw}, whiskerprops={'linewidth': lw},
                        medianprops={'linewidth': lw, 'color':'black'},
                        patch_artist=True,
                        labels=['Young', 'Old'])

    # Set colors of the boxes
    colors = [YNG_COL, OLD_COL]
    for box, color in zip(bplot['boxes'], colors):
        box.set_facecolor(color)

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    _set_lr_spines(ax, 2)

    _save_fig(save_fig, save_name)


def plot_comp(dat, save_fig=False, save_name=None):
    """Plot comparison between groups, as a mean value with an errorbar.

    Dat should be 1D vector, with data that can be split up by YOUNG & OLD _INDS.
    """

    fig, ax = plt.subplots(figsize=[2, 4])

    # Split up data
    you_dat = dat[YNG_INDS][~np.isnan(dat[YNG_INDS])]
    old_dat = dat[OLD_INDS][~np.isnan(dat[OLD_INDS])]

    means = [np.mean(you_dat), np.mean(old_dat)]
    sems = [sem(you_dat), sem(old_dat)]

    plt.errorbar([1, 2], means, yerr=sems, xerr=None, fmt='.',
                 markersize=22, capsize=10, elinewidth=2, capthick=2)

    ax.set_xlim([0.5, 2.5])
    plt.xticks([1, 2], ['Young', 'Old'])

    # Titles & Labels
#    ax.set_title('Data')
#    ax.set_xlabel('Noise Levels')
#    ax.set_ylabel('Error')

    # Set the top and right side frame & ticks off
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

    plt.xticks([x1, x2], ["Young", "Old"])

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    _set_lr_spines(ax)

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

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
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

    # Shade regions of siginificant difference
    avg_diffs, p_vals = calc_ap_comps(fs, ap_psds)
    sh_starts, sh_ends = get_pval_shades(fs, p_vals)
    _plt_shade_regions(sh_starts, sh_ends)

    # Plot limits & labels
    ax.set_xlim([min(fs)-0.5, max(fs) + 0.5])
    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight':'bold'})
    ax.set_ylabel('Power', {'fontsize': 14, 'fontweight':'bold'})

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    _set_lr_spines(ax, 2)

    plt.legend()

    _save_fig(save_fig, save_name)

    if return_vals:
        return fs, avg_diffs, p_vals


def plot_ap_band_diff(freqs, avg_diffs, p_vals, save_fig=False, save_name=None):
    """Plot a comparison between specific frequency values from generated aperiodic components.

    freqs - 1d vector of frequency values
    avg_diffs - 1d vector of differences per frequency value (same len as freqs)
    p_vals - 1d vector of p-values for each comparison  (same len as freqs)
    """

    fig, ax = plt.subplots(figsize=[8, 6])

    ax.set_xlim([0, max(freqs) + 0.5])
    ax.set_ylim([-0.75, 0.75])

    plt.plot(freqs, avg_diffs, '.', markersize=12)

    plt.title('\'Periodic\' Differences from Aperiodic Component', {'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight': 'bold'})
    ax.set_ylabel('Difference in Power (au)', {'fontsize': 14, 'fontweight': 'bold'})

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    _set_lr_spines(ax, 2)

    # Add shading for statistically significant different regions
    sh_starts, sh_ends = get_pval_shades(freqs, p_vals)
    _plt_shade_regions(sh_starts, sh_ends)

    _save_fig(save_fig, save_name)


def plot_overlap(m1, m2, std1, std2, save_fig=False, save_name=None):
    """Visualize the overlap of two gaussians.

    m1 & std1 : define the average alpha
    m2 & std2 : define the average alpha
    """

    # Get point of overlap
    r = get_intersect(m1, m2, std1, std2)

    # Initialize plot
    fig, ax = plt.subplots(figsize=[6, 6])

    ax.set_xlim([0, 20.])
    ax.set_ylim([0, 0.21])

    x = np.linspace(0, 20, 1000)
    step = x[1] - x[0]

    plot1 = plt.plot(x, norm.pdf(x, m1, std1), 'grey', lw=2.5, label='Average')
    plot2 = plt.plot(x, norm.pdf(x, m2, std2), 'black', lw=2.5, label='Canonical')
    #plot3 = plt.plot(r, norm.pdf(r, m1, std1), markersize=22)#, 'o')

    # Shade in overlapping area
    _ = plt.fill_between(x[x>r-step], 0, norm.pdf(x[x>r-step], m1, std1), alpha=0.7, color='#2ba848', lw=0)
    _ = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r], m2, std2), alpha=0.7, color='#2ba848', lw=0)
    _ = plt.fill_between(x[x<r], norm.pdf(x[x<r], m2, std2), norm.pdf(x[x<r], m1, std1),
                     alpha=0.7, color='#d10c29', lw=0)

    # Set the top and right side frame & ticks off
    _set_lr_spines(ax, 2)

    plt.legend(fontsize=12)

    _save_fig(save_fig, save_name)


###################################################################################################
###################################################################################################

def _save_fig(save_fig, save_name):
    """Save out current figure."""

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


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

    for shs, she in zip(shade_starts, shade_ends):
        plt.axvspan(shs, she, color='r', alpha=0.2, lw=0)
