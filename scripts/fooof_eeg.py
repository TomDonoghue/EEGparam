"""Analysis script for the EEG project.

Notes
-----
- Baseline: -500 - 350
- All trials average (all subjects), or average of subject averages?
- Filter individual channels or single frequency estimate per subject?
"""

from os import listdir
from os.path import join as pjoin

import numpy as np
from scipy.signal import periodogram

# MNE & Associated Code
import mne
from mne.preprocessing import ICA, read_ica
from mne.utils import _time_mask

from autoreject import AutoReject, read_auto_reject
from autoreject.autoreject import _apply_interp

# FOOOF & Associated Code
from fooof import FOOOF, FOOOFGroup
from fooof.analysis import get_band_peak
from fooof.funcs import combine_fooofs

# Other custom code
from om.core.utils import clean_file_list

###################################################################################################
###################################################################################################

#################################################
## SETTINGS

# Processing Options
#   Note: by default, if set to false, this will apply a saved solution for ICA & AR
RUN_ICA = True
RUN_AUTOREJECT = True

# Set up event code dictionary, with key labels for each event type
EV_DICT = {'LeLo1': [201, 202], 'LeLo2': [205, 206], 'LeLo3': [209, 210],
           'RiLo1': [203, 204], 'RiLo2': [207, 208], 'RiLo3': [211, 212]}

# Set labels & definitions
LOAD_LABELS = ['Load1', 'Load2', 'Load3']
SIDE_LABELS = ['Contra', 'Ipsi']
SEG_LABELS = ['Pre', 'Early', 'Late']
SEG_TIMES = [(-0.85, -0.35), (0.1, 0.6), (0.5, 1.0)]

# Event codes for correct and incorrect codes
CORR_CODES = [2, 1]
INCO_CODES = [102, 101]

# Set channel to use to derive FOOOFed alpha band
CHL = 'Oz'

# Set names for EOG channels
EOG_CHS = ['LHor', 'RHor', 'IVer', 'SVer']

# Set FOOOF frequency range
FREQ_RANGE = [3, 25]

# FOOOF Settings
PEAK_WIDTH_LIMITS = [1, 6]
MAX_N_PEAKS = 6
MIN_PEAK_AMP = 0.05
PEAK_THRESHOLD = 1.5

# Set paths
DAT_PATH = '/Users/tom/Documents/Data/Voytek_WMData/G2/'
RES_PATH = '/Users/tom/Documents/Research/1-Projects/fooof/2-Data/Results/'

# Data settings
EXT = '.bdf'
N_TIMES = 999

###################################################################################################
###################################################################################################

def main():

    #################################################
    ## SETUP

    # Get list of subject files
    subj_files = listdir(DAT_PATH)
    subj_files = clean_file_list(subj_files, EXT)

    # Initialize FOOOFGroup object, and save out settings file
    #  This is the one used for first FOOOFing (across all channels - 2 minute 'rest data')
    fg = FOOOFGroup(peak_width_limits=PEAK_WIDTH_LIMITS, max_n_peaks=MAX_N_PEAKS,
                    min_peak_amplitude=MIN_PEAK_AMP, peak_threshold=PEAK_THRESHOLD)
    fg.save('0-FOOOF_Settings', pjoin(RES_PATH, 'FOOOF'), save_settings=True)

    # Initialize FOOOF model, used for second FOOOFing
    #  This is the one used for the task related, epoched data
    fm = FOOOF(peak_width_limits=PEAK_WIDTH_LIMITS, max_n_peaks=MAX_N_PEAKS,
               min_peak_amplitude=MIN_PEAK_AMP, peak_threshold=PEAK_THRESHOLD)

    # Set up the dictionary to store all the FOOOF results
    fg_dict = dict()
    for load_label in LOAD_LABELS:
        fg_dict[load_label] = dict()
        for side_label in SIDE_LABELS:
            fg_dict[load_label][side_label] = dict()
            for seg_label in SEG_LABELS:
                fg_dict[load_label][side_label][seg_label] = []

    # Initialize group level data stores
    n_subjs, n_conds, n_times = len(subj_files), 3, N_TIMES
    group_fooofed_alpha_freqs = np.zeros(shape=[n_subjs])
    dropped_components = np.ones(shape=[n_subjs, 50]) * 999
    dropped_trials = np.ones(shape=[n_subjs, 1500]) * 999
    canonical_group_avg_dat = np.zeros(shape=[n_subjs, n_conds, n_times])
    fooofed_group_avg_dat = np.zeros(shape=[n_subjs, n_conds, n_times])

    # Set channel types
    ch_types = {'LHor' : 'eog', 'RHor' : 'eog', 'IVer' : 'eog', 'SVer' : 'eog',
                'LMas' : 'misc', 'RMas' : 'misc', 'Nose' : 'misc', 'EXG8' : 'misc'}

    #################################################
    ## RUN ACROSS ALL SUBJECTS

    # Run analysis across each subject
    for s_ind, subj_file in enumerate(subj_files):

        # Get subject label and print status
        subj_label = subj_file.split('.')[0]
        print('\nCURRENTLY RUNNING SUBJECT: ', subj_label, '\n')

        #################################################
        ## LOAD / ORGANIZE / SET-UP DATA

        # Load subject of data, apply apply fixes for channels, etc
        eeg_dat = mne.io.read_raw_edf(pjoin(DAT_PATH, subj_file),
                                      preload=True, verbose=False)

        # Fix channel name labels
        eeg_dat.info['ch_names'] = [chl[2:] for chl in \
            eeg_dat.ch_names[:-1]] + [eeg_dat.ch_names[-1]]
        for ind, chi in enumerate(eeg_dat.info['chs']):
            eeg_dat.info['chs'][ind]['ch_name'] = eeg_dat.info['ch_names'][ind]

        # Update channel types
        eeg_dat.set_channel_types(ch_types)

        # Set reference - average reference
        eeg_dat = eeg_dat.set_eeg_reference(ref_channels='average',
                                            projection=False, verbose=False)

        # Set channel montage
        chs = mne.channels.read_montage('standard_1020', eeg_dat.ch_names)
        eeg_dat.set_montage(chs)

        # Get event information & check all used event codes
        evs = mne.find_events(eeg_dat, shortest_event=1, verbose=False)
        ev_codes = np.unique(evs[:, 2])

        # Pull out sampling rate
        srate = eeg_dat.info['sfreq']

        #################################################
        ## Pre-Processing: ICA
        if RUN_ICA:

            print("\nICA: CALCULATING SOLUTION\n")

            # ICA settings
            method = 'fastica'
            n_components = 0.99
            random_state = 47
            reject = {'eeg': 20e-4}

            # Initialize ICA object
            ica = ICA(n_components=n_components, method=method,
                      random_state=random_state)

            # High-pass filter data for running ICA
            eeg_dat.filter(l_freq=1., h_freq=None, fir_design='firwin');

            # Fit ICA
            ica.fit(eeg_dat, reject=reject)

            # Save out ICA solution
            ica.save(pjoin(RES_PATH, 'ICA', subj_label + '-ica.fif'))

        # Otherwise: load previously saved ICA to apply
        else:
            print("\nICA: USING PRECOMPUTED\n")
            ica = read_ica(pjoin(RES_PATH, 'ICA', subj_label + '-ica.fif'))

        # Find components to drop, based on correlation with EOG channels
        drop_inds = []
        for chi in EOG_CHS:
            inds, scores = ica.find_bads_eog(eeg_dat, ch_name=chi, threshold=2.5,
                                             l_freq=1, h_freq=10, verbose=False)
            drop_inds.extend(inds)
        drop_inds = list(set(drop_inds))

        # Set which components to drop, and collect record of this
        ica.exclude = drop_inds
        dropped_components[s_ind, 0:len(drop_inds)] = drop_inds

        # Apply ICA to data
        eeg_dat = ica.apply(eeg_dat);

        #################################################
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

        #################################################
        ## FOOOF

        # Set channel of interest
        ch_ind = eeg_dat.ch_names.index(CHL)

        # Calculate PSDs over ~ first 2 minutes of data, for specified channel
        fmin, fmax = 1, 50
        tmin, tmax = 5, 125
        psds, freqs = mne.time_frequency.psd_welch(eeg_dat, fmin=fmin, fmax=fmax,
                                                   tmin=tmin ,tmax=tmax,
                                                   n_fft=int(2*srate), n_overlap=int(srate),
                                                   n_per_seg=int(2*srate),
                                                   verbose=False)

        # Fit FOOOF across all channels
        fg.fit(freqs, psds, FREQ_RANGE, n_jobs=-1)

        # Save out FOOOF results
        fg.save(subj_label + '_fooof', pjoin(RES_PATH, 'FOOOF'), save_results=True)

        # Extract individualized CF from specified channel, add to group collection
        fm = fg.get_fooof(ch_ind, False)
        fooof_freq, _, fooof_bw = get_band_peak(fm.peak_params_, [7, 14])
        group_fooofed_alpha_freqs[s_ind] = fooof_freq

        # If not FOOOF alpha extracted, reset to 10
        if np.isnan(fooof_freq):
            fooof_freq = 10

        #################################################
        ## ALPHA FILTERING

        # CANONICAL: Filter data to canonical alpha band: 8-12 Hz
        alpha_dat = eeg_dat.copy()
        alpha_dat.filter(8, 12, fir_design='firwin', verbose=False)
        alpha_dat.apply_hilbert(envelope=True, verbose=False)

        # FOOOF: Filter data to FOOOF derived alpha band
        fooof_dat = eeg_dat.copy()
        fooof_dat.filter(fooof_freq-2, fooof_freq+2, fir_design='firwin')
        fooof_dat.apply_hilbert(envelope=True)

        #################################################
        ## EPOCH TRIALS

        # Set epoch timings
        tmin, tmax = -0.85, 1.1

        # Epoch trials - raw data for trial rejection
        epochs = mne.Epochs(eeg_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose=False)

        # Epoch trials - filtered version
        epochs_alpha = mne.Epochs(alpha_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                                  baseline=(-0.5, -0.35), preload=True, verbose=False);
        epochs_fooof = mne.Epochs(fooof_dat, evs2, ev_dict2, tmin=tmin, tmax=tmax,
                                  baseline=(-0.5, -0.35), preload=True, verbose=False);

        #################################################
        ## PRE-PROCESSING: AUTO-REJECT
        if RUN_AUTOREJECT:

            print('\nAUTOREJECT: CALCULATING SOLUTION\n')

            # Initialize and run autoreject across epochs
            ar = AutoReject(n_jobs=4, verbose=False)
            ar.fit(epochs)

            # Save out AR solution
            ar.save(pjoin(RES_PATH, 'AR', subj_label + '-ar.hdf5'), overwrite=True)

        # Otherwise: load & apply previously saved AR solution
        else:
            print('\nAUTOREJECT: USING PRECOMPUTED\n')
            ar = read_auto_reject(pjoin(RES_PATH, 'AR', subj_label + '-ar.hdf5'))
            ar.verbose = 'tqdm'

        # Apply autoreject to the original epochs object it was learnt on
        epochs, rej_log = ar.transform(epochs, return_log=True)

        # Apply autoreject to the copies of the data - apply interpolation, then drop same epochs
        _apply_interp(rej_log, epochs_alpha, ar.threshes_, ar.picks_, ar.verbose)
        epochs_alpha.drop(rej_log.bad_epochs)
        _apply_interp(rej_log, epochs_fooof, ar.threshes_, ar.picks_, ar.verbose)
        epochs_fooof.drop(rej_log.bad_epochs)

        # Collect which epochs were dropped
        dropped_trials[s_ind, 0:sum(rej_log.bad_epochs)] = np.where(rej_log.bad_epochs)[0]

        #################################################
        ## SET UP CHANNEL CLUSTERS

        # Set channel clusters - take channels contralateral to stimulus presentation
        #  Note: channels will be used to extract data contralateral to stimulus presentation
        le_chs = ['P3', 'P5', 'P7', 'P9', 'O1', 'PO3', 'PO7']       # Left Side Channels
        le_inds = [epochs.ch_names.index(chn) for chn in le_chs]
        ri_chs = ['P4', 'P6', 'P8', 'P10', 'O2', 'PO4', 'PO8']      # Right Side Channels
        ri_inds = [epochs.ch_names.index(chn) for chn in ri_chs]

        #################################################
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

        #################################################
        ## FOOOFING TRIAL AVERAGED DATA

        # Settings for trial average FOOOFing
        fmin, fmax = 4, 25

        # Loop loop loads & trials segments
        for seg_label, seg_time in zip(SEG_LABELS, SEG_TIMES):
            tmin, tmax = seg_time[0], seg_time[1]

            # Calculate PSDs across trials, fit FOOOF models to averages
            for le_label, ri_label, load_label in zip(['LeLo1', 'LeLo2', 'LeLo3'],
                                                      ['RiLo1', 'RiLo2', 'RiLo3'],
                                                      LOAD_LABELS):

                # Calculate trial wise PSDs - left side trials
                trial_freqs, le_trial_psds = periodogram(
                    epochs[le_label]._data[:, :, _time_mask(epochs.times, tmin, tmax, srate)],
                    srate, window='hann', nfft=4*srate)

                le_avg_psd_contra = np.mean(le_trial_psds[:, ri_inds, :], 0).mean(0)
                le_avg_psd_ipsi = np.mean(le_trial_psds[:, le_inds, :], 0).mean(0)

                # Calculate trial wise PSDs - right side trials
                trial_freqs, ri_trial_psds = periodogram(
                    epochs[ri_label]._data[:, :, _time_mask(epochs.times, tmin, tmax, srate)],
                    srate, window='hann', nfft=4*srate)

                ri_avg_psd_contra = np.mean(ri_trial_psds[:, le_inds, :], 0).mean(0)
                ri_avg_psd_ipsi = np.mean(ri_trial_psds[:, ri_inds, :], 0).mean(0)

                # Collapse PSD across left & right trials for given load
                try:
                    avg_psd_contra = np.mean(np.vstack([le_avg_psd_contra, ri_avg_psd_contra]), 0)
                    avg_psd_ipsi = np.mean(np.vstack([le_avg_psd_ipsi, ri_avg_psd_ipsi]), 0)

                    # Fit FOOOF & collect results
                    fm.fit(trial_freqs, avg_psd_contra, [fmin,fmax])
                    fg_dict[load_label]['Contra'][seg_label].append(fm.copy())
                    fm.fit(trial_freqs, avg_psd_ipsi, [fmin, fmax])
                    fg_dict[load_label]['Ipsi'][seg_label].append(fm.copy())

                except:
                    continue

    #################################################
    ## SAVE OUT RESULTS

    # Save out group data
    np.save(pjoin(RES_PATH, 'Group', 'alpha_freqs_group'), group_fooofed_alpha_freqs)
    np.save(pjoin(RES_PATH, 'Group', 'canonical_group'), canonical_group_avg_dat)
    np.save(pjoin(RES_PATH, 'Group', 'fooofed_group'), fooofed_group_avg_dat)
    np.save(pjoin(RES_PATH, 'Group', 'dropped_trials'), dropped_trials)
    np.save(pjoin(RES_PATH, 'Group', 'dropped_components'), dropped_components)

    # Save out second round of FOOOFing
    for load_label in LOAD_LABELS:
        for side_label in SIDE_LABELS:
            for seg_label in SEG_LABELS:
                fg = combine_fooofs(fg_dict[load_label][side_label][seg_label])
                fg.save('Group_' + load_label + '_' + side_label + '_' + seg_label,
                    pjoin(RES_PATH, 'FOOOF'), save_results=True)

if __name__ == "__main__":
    main()
