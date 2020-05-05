"""Settings."""

from fooof.bands import Bands

###################################################################################################
###################################################################################################

# Set paths
DATA_PATH = '/Users/tom/Documents/Data/02-Shared/Voytek_WMData/G2/'
RESULTS_PATH = '/Users/tom/Documents/Research/2-Projects/1-Current/fooof/2-Data/Results/'

# Define default band definitions
BANDS = Bands({'alpha' : [7, 14]})

# Group index information
YNG_INDS = list(range(14, 31))
OLD_INDS = list(range(0, 14))

# Group colour settings
YNG_COL = "#0d82c1"
OLD_COL = "#239909"

# Group data information
N_LOADS = 3
N_SUBJS = 31
N_TIMES = 3
