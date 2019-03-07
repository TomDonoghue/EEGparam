import numpy as np

from fooof import FOOOF
from fooof.data import FOOOFResults
from fooof.analysis import get_band_peak_group

##
##

def avg_fg(fg, avg='mean'):
    """Average across a FOOOFGroup object."""

    if avg == 'mean':
        avg_func = np.nanmean
    elif avg == 'median':
        avg_func = np.nanmedian

    ap_params = avg_func(fg.get_all_data('aperiodic_params'), 0)

    peak_params = avg_func(get_band_peak_group(fg.get_all_data('peak_params'), [7, 14], len(fg)), 0)
    peak_params = peak_params[np.newaxis, :]

    gaussian_params = avg_func(get_band_peak_group(fg.get_all_data('gaussian_params'), [7, 14], len(fg)), 0)
    gaussian_params = gaussian_params[np.newaxis, :]

    r2 = avg_func(fg.get_all_data('r_squared'))
    error = avg_func(fg.get_all_data('error'))

    results = FOOOFResults(ap_params, peak_params, r2, error, gaussian_params)

    # Create the new FOOOF object, with settings, data info & results
    fm = FOOOF()
    fm.add_settings(fg.get_settings())
    fm.add_data_info(fg.get_data_info())
    fm.add_results(results)

    return fm
