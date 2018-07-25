"""Plotting functions for EEG-FOOOF analysis."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

from fooof.core.funcs import gaussian_function, expo_nk_function

###################################################################################################
###################################################################################################

YNG_INDS = range(14, 31)
OLD_INDS = range(0, 14)

###################################################################################################
###################################################################################################

def plot_alpha_response_compare(canonical_alpha, fooofed_alpha, t_win, srate):
    """Plot the alpha response results.

    Note: both inputs should [n_conds, n_times] matrices.
    """

    # Plot alpha response between different alpha filters
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[16, 6])
    times = np.arange(t_win[0], t_win[1], 1/srate)

    # Canonical alpha
    ax1.set_title('Canonical Alpha')
    ax1.plot(times, canonical_alpha[0, :], 'b', label='Load-1')
    ax1.plot(times, canonical_alpha[1, :], 'g', label='Load-2')
    ax1.plot(times, canonical_alpha[2, :], 'y', label='Load-3')

    ax2.set_title('FOOOFed Alpha')
    ax2.plot(times, fooofed_alpha[0, :], 'b', label='Load-1')
    ax2.plot(times, fooofed_alpha[1, :], 'g', label='Load-2')
    ax2.plot(times, fooofed_alpha[2, :], 'y', label='Load-3');

    # Restrict x-axis plotting
    ax1.set_xlim([-0.5, 1.0])
    ax2.set_xlim([-0.5, 1.0])

    ax1.legend(); ax2.legend();


def plot_comp_boxplot(dat):
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

    # Fill boxplots with colors
    you_col = "#0d82c1"
    old_col = "#239909"

    # Set colors of the boxes
    colors = [you_col, old_col]
    for box, color in zip(bplot['boxes'], colors):
        box.set_facecolor(color)

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


def plot_background(bgs, save_fig=False, save_name=None):
    """Plot background components, comparing between groups."""

    n_subjs = bgs.shape[0]

    you_col = "#0d82c1"
    old_col = "#239909"

    # Set offset to be zero across all PSDs
    tbgs = np.copy(bgs)
    tbgs[:, 0] = 1

    # Initialize figure
    fig, ax = plt.subplots(figsize=[6, 6])

    # Get frequency axis (x-axis)
    fs = np.arange(3, 40, 0.1)

    # Create the background model from parameters
    bg_psds = np.empty(shape=[n_subjs, len(fs)])
    for ind, bg in enumerate(tbgs):
        bg_psds[ind, :] = expo_nk_function(fs, *tbgs[ind, :])

    plt_log = False
    fs = np.log10(fs) if plt_log else fs

    # Plot each individual subject
    for ind in range(n_subjs):
        lc = you_col if ind in YNG_INDS else old_col
        ax.plot(fs, bg_psds[ind, :], lc, alpha=0.2, linewidth=1.5)

    # Plot the average across all subjects, split up by age group
    you_avg = np.mean(bg_psds[YNG_INDS, :], 0)
    old_avg = np.mean(bg_psds[OLD_INDS, :], 0)
    ax.plot(fs, you_avg, you_col, linewidth=4, label='Young')
    ax.plot(fs, old_avg, old_col, linewidth=4, label='Old')

    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight':'bold'})
    ax.set_ylabel('Power', {'fontsize': 14, 'fontweight':'bold'})

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.legend()

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


def plot_sl_band_diff(freqs, model_bgs, save_fig=False, save_name=None):
    """Plot a comparison between specific frequency values from generated backgrounds.

    freqs - 1d vector of frequency values
    model_bgs - 2d vector of [n_subjs, n_freqs]
    """

    avg_diff = []
    p_vals = []

    for f_val in model_bgs.T:
        avg_diff.append(np.mean(f_val[YNG_INDS] - np.mean(f_val[OLD_INDS])))
        p_vals.append(ttest_ind(f_val[YNG_INDS], f_val[OLD_INDS])[1])

    # Initialize plot
    fig, ax = plt.subplots(figsize=[8, 6])

    ax.set_xlim([0, max(freqs) + 0.5])
    ax.set_ylim([-0.75, 0.75])

    plt.plot(freqs, avg_diff, '.', markersize=12)

    pst = None
    pen = None

    # Find p-value ranges to shade in
    #  Note: this presumes starts significant, gets unsignificant, gets significant again
    for f_val, p_val in zip(freqs, p_vals):
        if p_val < 0.05:
            if not pst and pen:
                pst = f_val
        else:
            if not pen:
                pen = f_val

    plt.title('\'Periodic\' Differences from Aperiodic Component', {'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Frequency', {'fontsize': 14, 'fontweight': 'bold'})
    ax.set_ylabel('Difference in Power (au)', {'fontsize': 14, 'fontweight': 'bold'})

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add shading for statistically significant different regions
    sh_starts = [0, pst-0.5]
    sh_ends = [pen+0.5, max(freqs) + 0.5]
    for shs, she in zip(sh_starts, sh_ends):
        plt.axvspan(shs, she, color='r', alpha=0.2, lw=0)

    if save_fig:
        save_name = 'plts/' + save_name + '.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
