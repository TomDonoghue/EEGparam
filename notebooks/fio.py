"""Load and data organization functions."""

from os.path import join as pjoin

import numpy as np

from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_group

###################################################################################################
###################################################################################################

def load_fooof_task(data_path, side='Contra'):
    """Loads task data in for all subjects, across loads and trial times.
    Returns matrix of slope & alpha data.

    data_path : path to where data
    side: 'Ipsi' or 'Contra'
    """

    # Settings
    n_loads, n_subjs, n_times = 3, 31, 3

    # Collect measures together from FOOOF results into matrices
    all_exps = np.zeros(shape=[n_loads, n_subjs, n_times])
    all_alphas = np.zeros(shape=[n_loads, n_subjs, n_times])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        # Load the FOOOF analyses of the average
        pre, early, late = FOOOFGroup(), FOOOFGroup(), FOOOFGroup()
        pre.load('Group_' + load + '_' + side + '_Pre', pjoin(data_path, 'FOOOF'))
        early.load('Group_' + load + '_' + side +  '_Early', pjoin(data_path, 'FOOOF'))
        late.load('Group_' + load + '_' + side + '_Late', pjoin(data_path, 'FOOOF'))

        for ind, fg in enumerate([pre, early, late]):
            all_exps[li, :, ind] = fg.get_all_data('aperiodic_params', 'exponent').T
            temp_alphas = get_band_peak_group(fg.get_all_data('peak_params'), [7, 14], len(fg))

            temp_alphas = temp_alphas[:, 1]
            #temp_alphas = all_areas(temp_alphas)

            all_alphas[li, :, ind] = temp_alphas

    # Replace alpha NaN's with 0
    #all_alphas[np.isnan(all_alphas)] = 0

    return all_exps, all_alphas
