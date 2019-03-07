"""Load and data organization functions."""

from os.path import join as pjoin

import numpy as np

from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_group

###################################################################################################
###################################################################################################

def load_fooof_task_ap(data_path, side='Contra', folder='FOOOF'):
    """Loads task data in for all subjects, selects and return aperiodic FOOOF outputs.

    data_path : path to where data
    side: 'Ipsi' or 'Contra'
    """

    # Settings
    n_loads, n_subjs, n_times = 3, 31, 3

    # Collect measures together from FOOOF results into matrices
    all_exps = np.zeros(shape=[n_loads, n_subjs, n_times])
    all_offsets = np.zeros(shape=[n_loads, n_subjs, n_times])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        # Load the FOOOF analyses of the average
        pre, early, late = FOOOFGroup(), FOOOFGroup(), FOOOFGroup()
        pre.load('Group_' + load + '_' + side + '_Pre', pjoin(data_path, folder))
        early.load('Group_' + load + '_' + side +  '_Early', pjoin(data_path, folder))
        late.load('Group_' + load + '_' + side + '_Late', pjoin(data_path, folder))

        for ind, fg in enumerate([pre, early, late]):
            all_exps[li, :, ind] = fg.get_all_data('aperiodic_params', 'exponent')#.T
            all_offsets[li, :, ind] = fg.get_all_data('aperiodic_params', 'offset')#.T

    return all_offsets, all_exps


def load_fooof_task_pe(data_path, side='Contra', param_ind=1, folder='FOOOF'):
    """Loads task data in for all subjects, selects and return periodic FOOOF outputs.

    data_path : path to where data
    side: 'Ipsi' or 'Contra'
    """

    # Settings
    n_loads, n_subjs, n_times = 3, 31, 3

    # Collect measures together from FOOOF results into matrices
    all_alphas = np.zeros(shape=[n_loads, n_subjs, n_times])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        # Load the FOOOF analyses of the average
        pre, early, late = FOOOFGroup(), FOOOFGroup(), FOOOFGroup()
        pre.load('Group_' + load + '_' + side + '_Pre', pjoin(data_path, folder))
        early.load('Group_' + load + '_' + side +  '_Early', pjoin(data_path, folder))
        late.load('Group_' + load + '_' + side + '_Late', pjoin(data_path, folder))

        for ind, fg in enumerate([pre, early, late]):
            temp_alphas = get_band_peak_group(fg.get_all_data('peak_params'), [7, 14], len(fg))
            all_alphas[li, :, ind] = temp_alphas[:, param_ind]

    return all_alphas
