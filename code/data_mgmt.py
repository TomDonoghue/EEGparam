"""Functions for loading and data management for EEG-FOOOF."""

from os.path import join as pjoin

import numpy as np

from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg

from settings import BANDS, YNG_INDS, OLD_INDS
from settings import N_LOADS, N_SUBJS, N_TIMES

###################################################################################################
###################################################################################################

def reshape_dat(dat):

    yng_dat = np.vstack([dat[0, YNG_INDS, :], dat[1, YNG_INDS, :], dat[2, YNG_INDS, :]])
    old_dat = np.vstack([dat[0, OLD_INDS, :], dat[1, OLD_INDS, :], dat[2, OLD_INDS, :]])

    return yng_dat, old_dat


def load_fooof_task_md(data_path, side='Contra', folder='FOOOF'):
    """Load task data in for all subjects, selects & return metadata."""

    # Collect measures together from FOOOF results into matrices
    all_r2s = np.zeros(shape=[N_LOADS, N_SUBJS, N_TIMES])
    all_errs = np.zeros(shape=[N_LOADS, N_SUBJS, N_TIMES])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        pre, early, late = _load_fgs(data_path, folder, side, load)

        for ind, fg in enumerate([pre, early, late]):
            all_r2s[li, :, ind] = fg.get_params('r_squared')
            all_errs[li, :, ind] = fg.get_params('error')

    return all_r2s, all_errs


def load_fooof_task_ap(data_path, side='Contra', folder='FOOOF'):
    """Loads task data in for all subjects, selects and return aperiodic FOOOF outputs.

    data_path : path to where data
    side: 'Ipsi' or 'Contra'
    """

    # Collect measures together from FOOOF results into matrices
    all_exps = np.zeros(shape=[N_LOADS, N_SUBJS, N_TIMES])
    all_offsets = np.zeros(shape=[N_LOADS, N_SUBJS, N_TIMES])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        pre, early, late = _load_fgs(data_path, folder, side, load)

        for ind, fg in enumerate([pre, early, late]):
            all_exps[li, :, ind] = fg.get_params('aperiodic_params', 'exponent')
            all_offsets[li, :, ind] = fg.get_params('aperiodic_params', 'offset')

    return all_offsets, all_exps


def load_fooof_task_pe(data_path, side='Contra', param_ind=1, folder='FOOOF'):
    """Loads task data in for all subjects, selects and return periodic FOOOF outputs.

    data_path : path to where data
    side: 'Ipsi' or 'Contra'
    """

    # Collect measures together from FOOOF results into matrices
    all_alphas = np.zeros(shape=[N_LOADS, N_SUBJS, N_TIMES])

    for li, load in enumerate(['Load1', 'Load2', 'Load3']):

        pre, early, late = _load_fgs(data_path, folder, side, load)

        for ind, fg in enumerate([pre, early, late]):
            temp_alphas = get_band_peak_fg(fg, BANDS.alpha)
            all_alphas[li, :, ind] = temp_alphas[:, param_ind]

    return all_alphas


def _load_fgs(data_path, folder, side, load):
    """Helper to load FOOOFGroups."""

    # Load the FOOOF analyses of the average
    pre, early, late = FOOOFGroup(), FOOOFGroup(), FOOOFGroup()
    pre.load('Group_' + load + '_' + side + '_Pre', pjoin(data_path, folder))
    early.load('Group_' + load + '_' + side +  '_Early', pjoin(data_path, folder))
    late.load('Group_' + load + '_' + side + '_Late', pjoin(data_path, folder))

    return pre, early, late
