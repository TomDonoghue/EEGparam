"""   """

import numpy as np

from settings import YNG_INDS, OLD_INDS

###################################################################################################
###################################################################################################

def reshape_dat(dat):

    yng_dat = np.vstack([dat[0, YNG_INDS, :], dat[1, YNG_INDS, :], dat[2, YNG_INDS, :]])
    old_dat = np.vstack([dat[0, OLD_INDS, :], dat[1, OLD_INDS, :], dat[2, OLD_INDS, :]])

    return yng_dat, old_dat
