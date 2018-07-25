""""   """

import numpy as np
from scipy.stats import norm

###################################################################################################
###################################################################################################

def drop_nan(vec):
    return vec[~np.isnan(vec)]


def print_stat(label, stat_val, p_val):
    print(label + ': \t {: 5.4f} \t{:5.4f}'.format(stat_val, p_val))


def get_intersect(m1, m2, std1, std2):
    """Gets the point of intersection of two gaussians defined by (m1, std1) & (m2, std2)"""

    a = 1. / (2.*std1**2) - 1. / (2.*std2**2)
    b = m2 / (std2**2) - m1 / (std1**2)
    c = m1**2 / (2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)

    return np.roots([a,b,c])[0]


def get_overlap(intersect, m1, m2, std1, std2):
    """Get the percent overlap of two gaussians, given their definitions, and intersection point."""

    return norm.cdf(intersect, m2, std2) + (1. - norm.cdf(intersect, m1, std1))
