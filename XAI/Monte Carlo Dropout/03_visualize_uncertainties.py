#This file uses the uncertainties created in 02_get_uncertainties to create boxplots from the results

import pandas as pd
import numpy as np
import plotly.express as px

"""
Function which takes list of uncertanties and normalizes it to range 0 - 1
call with: 
    uncertainties: pandas series of uncertainties

Returns: 
    uncertainties_norm: pandas series of normalized uncertainties
"""
def uncertainty_norm(uncertainties):
    uncertainties_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
    return uncertainties_norm

uncertainties = pd.read_csv('ModelUncertainties.csv')
uncertainties['is_correct'] = uncertainties['class_pred'] == uncertainties['true_class']
uncertainties['class_un_norm'] = uncertainty_norm(uncertainties['class_un'])

px.box(uncertainties, x = 'is_correct', y ='class_un', title = 'Uncertainties of Xception Model')
