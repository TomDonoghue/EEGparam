"""Analysis script for the EEG project.

Notes
-----
- Open question: baselining?
    - Baseline: -500 - 350
- All trials average (all subjects), or average of subject averages?
- Filter individual channels or single frequency estimate per subject?
"""

from os import listdir
from os.path import join as pjoin

import numpy as np

# MNE & Associated Code
import mne
from autoreject import LocalAutoRejectCV

# FOOOF & Associated Code
from fooof import FOOOF, FOOOFGroup
from fooof.analysis import get_band_peak
from fooof.utils import combine_fooofs

# Other custom code
from om.core.utils import clean_file_list

###################################################################################################
###################################################################################################

# Set which group to run
GROUP = 'G2'

# Set up event code dictionary, with key labels for each event type
EV_DICT = {'LeLo1': [201, 202], 'LeLo2': [205, 206], 'LeLo3': [209, 210],
           'RiLo1': [203, 204], 'RiLo2': [207, 208], 'RiLo3': [211, 212]}

# Event codes for correct and incorrect codes
CORR_CODES = [2, 1]
INCO_CODES = [102, 101]

# Set channel to use to derive FOOOFed alpha band
CHL = 'Oz'

# Set FOOOF frequency range
FREQ_RANGE = [3, 25]

# Set paths
DAT_PATH = pjoin('/Users/tom/Documents/Data/Voytek_WMData/', GROUP)
RES_PATH = pjoin('/Users/tom/Documents/Research/1-Projects/fooof/2-Data/Results/', GROUP)

# Group specific things
EXT = '.set' if GROUP == 'G1' else '.bdf'

###################################################################################################
###################################################################################################

def main():

    # Get list of subject files
    subj_files = listdir(DAT_PATH)
    subj_files = clean_file_list(subj_files, EXT)

    # Initialize FOOOF model, used for second FOOOFing
    fm = FOOOF(peak_width_limits=[1, 6], min_peak_amplitude=0.1)
    fg_dict = {'Load1' : [], 'Load2' : [], 'Load3' : []}

    # Initialize FOOOFGroup object, and save out settings file
    fg = FOOOFGroup(peak_width_limits=[1, 8])
    fg.save('0-FOOOF_Settings', pjoin(RES_PATH, 'FOOOF'), save_settings=True)

    # Initialize group level data stores
    n_subjs, n_conds, n_times = len(subj_files), 3, 436
    group_fooofed_alpha_freqs = np.zeros(shape=[n_subjs])
    canonical_group_avg_dat = np.zeros(shape=[n_subjs, n_conds, n_times])
    fooofed_group_avg_dat = np.zeros(shape=[n_subjs, n_conds, n_times])
    dropped_trials = np.zeros(shape=[n_subjs, 500])

    for s_ind, subj_file in enumerate(subj_files):

        # Get subject label and print status
        subj_label = subj_file.split('.')[0]
        print('\nCURRENTLY RUNNING SUBJECT: ', subj_label, '\n')

        ## LOAD / ORGANIZE / SET-UP DATA

        # Load subject of data - and specific fixes for channels and so on depending on group
        if GROUP == 'G1':

            eeg_dat = mne.io.read_raw_eeglab(pjoin(DAT_PATH, subj_file), preload=True, verbose=False)

            # Set non-EEG channel types
            ch_types = {'LO1' : 'eog', 'LO2' : 'eog', 'IO1' : 'eog', 'A1' : 'misc', 'A2' : 'misc'}

        if GROUP == 'G2':

            eeg_dat = mne.io.read_raw_edf(pjoin(DAT_PATH, subj_file), preload=True)

            # Fix channel name labels
            eeg_dat.info['ch_names'] = [chl[2:] for chl in eeg_dat.ch_names[:-1]] + [eeg_dat.ch_names[-1]]
            for ind, chi in enumerate(eeg_dat.info['chs']):
                eeg_dat.info['chs'][ind]['ch_name'] = eeg_dat.info['ch_names'][ind]

            # Set channel types
            ch_types = {'LHor' : 'eog', 'RHor' : 'eog', 'IVer' : 'eog', 'SVer' : 'eog',
                        'LMas' : 'misc', 'RMas' : 'misc', 'Nose' : 'misc', 'EXG8' : 'misc'}

        # Update channel types
        eeg_dat.set_channel_types(ch_types)

        # Set to keep current reference
        eeg_dat.set_eeg_reference(ref_channels=[], verbose=False)

        # Set channel montage
        chs = mne.channels.read_montage('standard_1020', eeg_dat.ch_names)
        eeg_dat.set_montage(chs)

        # Get event information & check all used event codes
        evs = mne.find_events(eeg_dat, shortest_event=1, verbose=False)
        ev_codes = np.unique(evs[:, 2])

        # Pull out sampling rate
        srate = eeg_dat.info['sfreq']

        ## SORT OUT EVENT CODES

        # Extract a list of all the event labels
        all_trials = [it for it2 in EV_DICT.values() for it in it2]

        # Create list of new event codes to be used to label correct trials (300s)
        all_trials_new = [it + 100 for it in all_trials]
        # This is an annoying way to collapse across the doubled event markers from above
        all_trials_new = [it - 1 if not ind%2 == 0 else it for ind, it in enumerate(all_trials_new)]
        # Get labelled dictionary of new event names
        ev_dict2 = {k:v for k, v in zip(EV_DICT.keys(), set(all_trials_new))}

        # Initialize variables to store new event definitions
        evs2 = np.empty(shape=[0, 3], dtype='int64')
        lags = np.array([])

        # Loop through, creating new events for all correct trials
        t_min, t_max = -0.4, 3.0
        for ref_id, targ_id, new_id in zip(all_trials, CORR_CODES * 6, all_trials_new):

            t_evs, t_lags = mne.event.define_target_events(evs, ref_id, targ_id, srate,
                                                           t_min, t_max, new_id)

            if len(t_evs) > 0:
                evs2 = np.vstack([evs2, t_evs])
                lags = np.concatenate([lags, t_lags])

        ## FOOOF

        # Set channel of interest
        ch_ind = eeg_dat.ch_names.index(CHL)

        # Calculate PSDs over ~ first 2 minutes of data, for specified channel
        psds, freqs = mne.time_frequency.psd_welch(eeg_dat, fmin=2, fmax=40, tmin=0 ,tmax=120,
                                                   n_fft=int(2*srate), n_overlap=int(srate), verbose=False)

        # Fit FOOOF to across all channels
        fg.fit(freqs, psds, FREQ_RANGE, n_jobs=-1)

        # Save out FOOOF results
        fg.save(subj_label + '_fooof', pjoin(RES_PATH, 'FOOOF'), save_results=True)

        # Extract individualized CF from specified channel, add to group collection
        fm = fg.get_fooof(ch_ind, False)
        fooof_freq, _, fooof_bw = get_band_peak(fm.peak_params_, [7, 13])
        group_fooofed_alpha_freqs[s_ind] = fooof_freq

        ## ALPHA FILTERING - CANONICAL

        # Filter data to canonical alpha band: 8-12 Hz
        alpha_dat = eeg_dat.copy()
        alpha_dat.filter(8, 12, fir_design='firwin', verbose=False)
        alpha_dat.apply_hilbert(envelope=True, verbose=False)

        ## ALPHA FILTERING - FOOOF

        # Filter data to FOOOF derived alpha band
        fooof_dat = eeg_dat.copy()
        fooof_dat.filter(fooof_freq-2, fooof_freq+2, fir_design='firwin')
        #fooof_dat.filter(fooof_freq-1.5*fooof_bw, fooof_freq+1.5*fooof_bw, fir_design='firwin', verbose=False)
        fooof_dat.apply_hilbert(envelope=True)

        ## EPOCH TRIALS

        # Set epoch timings
        #  Note: not sure of exact / best range to run on. This is the range auto-reject sees.
        tmin, tmax = -0.5, 1.2

        # Epoch trials - raw data for trial rejection
        epochs = mne.Epochs(eeg_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                            baseline=(-0.5, -0/25), preload=True, verbose=False)

        # Epoch trials - filtered version
        epochs_alpha = mne.Epochs(alpha_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                                  baseline=(-0.5, -0.35), preload=True, verbose=False);
        epochs_fooof = mne.Epochs(fooof_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                                  baseline=(-0.5, -0.35), preload=True, verbose=False);

        ## PRE-PROCESSING: AUTO-REJECT

        # Initialize and run autoreject across epochs
        ar = LocalAutoRejectCV()
        epochs = ar.fit_transform(epochs)
        dropped_trials[s_ind, 0:len(ar.bad_epochs_idx)] = ar.bad_epochs_idx

        # Drop same trials from filtered data
        epochs_alpha.drop(ar.bad_epochs_idx)
        epochs_fooof.drop(ar.bad_epochs_idx)

        ## SET UP CHANNEL CLUSTERS

        # Set channel clusters - take channels contralateral to stimulus presentation
        #  Note: channels will be used to extract data contralateral to stimulus presentation
        le_chs = ['P3', 'P5', 'P7', 'P9', 'O1', 'PO3', 'PO7']       # Left Side Channels
        le_inds = [epochs.ch_names.index(chn) for chn in le_chs]
        ri_chs = ['P4', 'P6', 'P8', 'P10', 'O2', 'PO4', 'PO8']      # Right Side Channels
        ri_inds = [epochs.ch_names.index(chn) for chn in ri_chs]

        ## TRIAL-RELATED ANALYSIS: CANONICAL vs. FOOOF

        # Pull out channel of interest for each load level - canonical data
        #  Channels extracted are those contralateral to stimulus presentation
        lo1_a = np.concatenate([epochs_alpha['LeLo1']._data[:, ri_inds, :],
                                epochs_alpha['RiLo1']._data[:, le_inds, :]], 0)
        lo2_a = np.concatenate([epochs_alpha['LeLo2']._data[:, ri_inds, :],
                                epochs_alpha['RiLo2']._data[:, le_inds, :]], 0)
        lo3_a = np.concatenate([epochs_alpha['LeLo3']._data[:, ri_inds, :],
                                epochs_alpha['RiLo3']._data[:, le_inds, :]], 0)
        # Pull out channel of interest for each load level - fooofed data
        #  Channels extracted are those contralateral to stimulus presentation
        lo1_f = np.concatenate([epochs_fooof['LeLo1']._data[:, ri_inds, :],
                                epochs_fooof['RiLo1']._data[:, le_inds, :]], 0)
        lo2_f = np.concatenate([epochs_fooof['LeLo2']._data[:, ri_inds, :],
                                epochs_fooof['RiLo2']._data[:, le_inds, :]], 0)
        lo3_f = np.concatenate([epochs_fooof['LeLo3']._data[:, ri_inds, :],
                                epochs_fooof['RiLo3']._data[:, le_inds, :]], 0)

        # Calculate average across trials and channels - add to group data collection
        # Canonical data
        canonical_group_avg_dat[s_ind, 0, :] = np.mean(lo1_a, 1).mean(0)
        canonical_group_avg_dat[s_ind, 1, :] = np.mean(lo2_a, 1).mean(0)
        canonical_group_avg_dat[s_ind, 2, :] = np.mean(lo3_a, 1).mean(0)
        # FOOOFed data
        fooofed_group_avg_dat[s_ind, 0, :] = np.mean(lo1_f, 1).mean(0)
        fooofed_group_avg_dat[s_ind, 1, :] = np.mean(lo2_f, 1).mean(0)
        fooofed_group_avg_dat[s_ind, 2, :] = np.mean(lo3_f, 1).mean(0)

        ## FOOOFING TRIAL AVERAGED DATA

        # Settings for trial average FOOOFing
        fmin, fmax = 3, 25
        tmin, tmax = -0.1, 1.1
        n_fft, n_overlap, n_per_seg = 4*srate, srate/2, srate*2

        # Calculate PSDs across trials, fit FOOOF models to averages
        for le_label, ri_label, load in zip(['LeLo1', 'LeLo2', 'LeLo3'],
                                            ['RiLo1', 'RiLo2', 'RiLo3'],
                                            ['Load1', 'Load2', 'Load3']):

            # Calculate trial wise PSDs - left side trials
            le_trial_psds, trial_freqs = mne.time_frequency.psd_welch(epochs[le_label], fmin, fmax, tmin=tmin, tmax=tmax,
                                                                      n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                                                                      verbose=False)
            le_avg_psd = np.mean(le_trial_psds[:, ri_inds, :], 0).mean(0)

            # Calculate trial wise PSDs - right side trials
            ri_trial_psds, trial_freqs = mne.time_frequency.psd_welch(epochs[ri_label], fmin, fmax, tmin=tmin, tmax=tmax,
                                                                      n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                                                                      verbose=False)
            ri_avg_psd = np.mean(ri_trial_psds[:, le_inds, :], 0).mean(0)

            # Collapse PSD across left & right trials for given load
            avg_psd = np.mean(np.vstack([le_avg_psd, ri_avg_psd]), 0)

            # FOOOF
            fm.fit(trial_freqs, avg_psd)
            fg_dict[load].append(fm.copy())

    # Save out group data
    np.save(pjoin(RES_PATH, 'Group', 'alpha_freqs_group'), group_fooofed_alpha_freqs)
    np.save(pjoin(RES_PATH, 'Group', 'canonical_group'), canonical_group_avg_dat)
    np.save(pjoin(RES_PATH, 'Group', 'fooofed_group'), fooofed_group_avg_dat)
    np.save(pjoin(RES_PATH, 'Group', 'dropped_trials'), dropped_trials)

    # Save out second round of FOOOFing
    for load in ['Load1', 'Load2', 'Load3']:
        fg = combine_fooofs(fg_dict[load])
        fg.save('Group_' + load, pjoin(RES_PATH, 'FOOOF'), save_results=True)


if __name__ == "__main__":
    main()
