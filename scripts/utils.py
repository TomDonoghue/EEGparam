"""Utility functions for EEG-FOOOF analysis."""

import numpy as np
import matplotlib.pyplot as plt

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
