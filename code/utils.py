""""Helper / utility functions for EEG-FOOOF."""

from math import sqrt
from statistics import mean, stdev

import numpy as np
from scipy.stats import pearsonr, norm, ttest_ind

from settings import YNG_INDS, OLD_INDS

###################################################################################################
###################################################################################################

def cohens_d(d1, d2):
    """Calculate cohens-D: (u1 - u2)/SDpooled."""

    return (mean(d1) - mean(d2)) / (sqrt((stdev(d1) ** 2 + stdev(d2) ** 2) / 2))


def mean_diff(d1, d2):
    """Helper function to calculate mean differences."""

    return np.mean(d1) - np.mean(d2)


def check_outliers(data, thresh):
    """Calculate indices of outliers, as defined by a standard deviation threshold."""

    return list(np.where(np.abs(data - np.mean(data)) > thresh * np.std(data))[0])


def calc_diff(data, i1, i2):
    """Calculate element-wise differences between columns of an array."""

    return data[:, i1] - data[:, i2]


def drop_nan(arr):
    """Drop any NaN indices of an array."""

    return arr[~np.isnan(arr)]


def print_stat(label, stat_val, p_val):
    """Helper function to print out statistical tests."""

    print(label + ': \t {: 5.4f} \t{: 5.4f}'.format(stat_val, p_val))


def nan_corr(vec1, vec2):
    """Correlation of two vectors with NaN values.

    Note: assumes the vectors have NaN in the same indices.
    """

    nan_inds = np.isnan(vec1)

    return pearsonr(vec1[~nan_inds], vec2[~nan_inds])


def nan_ttest(vec1, vec2):
    """Run an independent samples t-test on vectors with NaNs in them."""

    d1 = vec1[~np.isnan(vec1)]
    d2 = vec2[~np.isnan(vec2)]

    return ttest_ind(d1, d2)


def calc_ap_comps(freqs, model_aps):
    """Calculate point by point comparison of power per frequency, from aperiodic components.

    freqs: vector of frequency values
    model_aps: power spectra generated as just the aperiodic component
    """

    avg_diffs = []
    p_vals = []

    for f_val in model_aps.T:
        avg_diffs.append(np.mean(f_val[YNG_INDS] - np.mean(f_val[OLD_INDS])))
        p_vals.append(ttest_ind(f_val[YNG_INDS], f_val[OLD_INDS])[1])

    return avg_diffs, p_vals


def get_intersect(m1, m2, std1, std2):
    """Gets the point of intersection of two gaussians defined by (m1, std1) & (m2, std2)"""

    a = 1. / (2.*std1**2) - 1. / (2.*std2**2)
    b = m2 / (std2**2) - m1 / (std1**2)
    c = m1**2 / (2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)

    return np.roots([a, b, c])[0]


def get_overlap(intersect, m1, m2, std1, std2):
    """Get the percent overlap of two gaussians, given their definitions, and intersection point."""

    return norm.cdf(intersect, m2, std2) + (1. - norm.cdf(intersect, m1, std1))


def get_pval_shades(freqs, p_vals):
    """Find p-value ranges to shade in.

    Note: This approach assumes starts significant, becomes non-significant, gets significant again.
    """

    pst, pen = None, None

    for f_val, p_val in zip(freqs, p_vals):
        if p_val < 0.05:
            if not pst and pen:
                pst = f_val
        else:
            if not pen:
                pen = f_val

    sh_starts = [0, pst-0.5]
    sh_ends = [pen+0.5, max(freqs) + 0.5]

    return sh_starts, sh_ends
