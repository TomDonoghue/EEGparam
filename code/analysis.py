"""Functions to run pieces of analyses for EEG-FOOOF."""

import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import get_intersect, get_overlap

###################################################################################################
###################################################################################################

def calc_overlaps(alphas):
    """
    The approach to do this is taken from:
    https://stackoverflow.com/questions/32551610/overlapping-probability-of-two-normal-distribution-with-scipy
    """

    # Note: current approach presumes no NaNs
    overlaps = []
    mean, std = 10, 2

    for alpha in alphas:

        # Get individial CF - keep BW @ 2
        ind_mean, ind_std = alpha[0], 2

        # Normalize all deviations from canonical to be lower than 10 Hz
        if ind_mean <= mean:
            m1, std1, m2, std2 = ind_mean, ind_std, mean, std
        else:
            m1, std1, m2, std2 = mean, std, ind_mean, ind_std

        intersect = get_intersect(m1, m2, std1, std2)
        overlap = get_overlap(intersect, m1, m2, std1, std2)

        overlaps.append(overlap)

    overlaps = np.array(overlaps)

    return overlaps


def run_model(model_def, data_def, print_model=True):
    """Helper function to use a DataFrame & patsy to set up & run a model.

    model_def: str, passed into patsy.
    data_def: dictionary of labels & data to use.
    """

    df = pd.DataFrame()

    for label, data in data_def.items():
        df[label] = data

    outcome, predictors = patsy.dmatrices(model_def, df)
    model = sm.OLS(outcome, predictors).fit()

    if print_model:
        print(model.summary())

    return model