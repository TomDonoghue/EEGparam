""".  """

import patsy
import statsmodels.api as sm
import pandas as pd

###################################################################################################
###################################################################################################

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