import pandas as pd
import numpy as np
import plotly.express as px


uncertainties = pd.read_csv('ModelUncertainties.csv')
uncertainties['is_correct'] = uncertainties['class_pred'] == uncertainties['true_class']

px.box(uncertainties, x = 'is_correct', y ='class_un', title = 'Uncertainties of Xception Model')
