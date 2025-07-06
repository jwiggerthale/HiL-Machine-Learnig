import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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


#Get uncertaintties from both models and create new useful columns
uns = pd.read_csv('ModelUncertainties.csv')
uns['vgg_class_un_norm'] = uncertainty_norm(uns['vgg_class_un'])
uns['vgg_is_correct'] = uns['vgg_class_pred'] == uns['vgg_true_class']
uns['xc_class_un_norm'] = uncertainty_norm(uns['xc_class_un'])
uns['xc_is_correct'] = uns['xc_class_pred'] == uns['xc_true_class']
uns['all_correct'] = uns['xc_is_correct'] & uns['vgg_is_correct']
uns['both_wrong'] = ~(uns['xc_is_correct'] | uns['vgg_is_correct'])#False when both models predict wrongly
uns['vgg_xc_identical'] =  uns['xc_class_pred'] == uns['vgg_class_pred']
uns['not_warned'] = uns['both_wrong'] & uns['vgg_xc_identical']#True when both models predict wrongly and both models predict identical class

#Examine how false positives and false negatives develop for different uncertainty weightings
weights_xc = np.arange(-1, 1.1, 0.1)
model_stats= []
for weight_xc in weights_xc:
    uns['combined_uncertainty'] = uns['xc_class_un_norm'] * weight_xc + uns['vgg_class_un_norm'] * (1 - weight_xc)
    uns['combined_uncertainty'] = uncertainty_norm(uns['combined_uncertainty'])
    both_wrong = uns[uns['both_wrong'] == True]
    both_correct = uns[uns['all_correct'] == True]
    not_warned = both_wrong[both_wrong['not_warned'] == True]#Operator not warned and both models predict identical (wrong) class
    q1 = both_wrong['combined_uncertainty'].quantile(0.25)
    num_false_negatives = len(not_warned[not_warned['combined_uncertainty'] < q1])#Operator still not warned and both models predict identical (wrong) class
    num_false_positives = len(both_correct[both_correct['combined_uncertainty'] > q1])#Operator warned althoug both models predict correctly
    fig = px.box(both_wrong, y = 'combined_uncertainty', title = f'Combined Uncertainty - Exception Weight: {weight_xc} - {num_false_positives} False Positive -  {num_false_negatives} False Negatives')
    fig.show()
    model_stats.append([weight_xc, num_false_positives, num_false_negatives])

model_stats = pd.DataFrame(model_stats, columns = ['exception_weight', 'false_positives', 'false_negatives'])

#Plot results to determine best weighting
trace1 = go.Scatter(x = model_stats['exception_weight'], y = model_stats['false_positives'], name='False Positives', marker=dict(color='blue'))
trace2 = go.Scatter(x = model_stats['exception_weight'], y = model_stats['false_negatives'], name='False Negatives', marker=dict(color='red'))
fig = go.Figure(data=[trace1, trace2])
fig.update_layout(title = 'False Positives and False Negatives for Different Weights for Xception Uncertainty', 
                  xaxis_title = 'Exception Weight', 
                  yaxis_title = 'Count'
                 )
fig.write_image("CombinedUncertainty.png")
