#This file uses the uncertainties calculated in 02 to define the best thresholds for:
#    a) vgg16-uncertainty 
#    b) combined uncertainty
#in order to minimize fpr and tpr
#Note: the exact values to be chosen depend on the criticality of the use case

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


uns = pd.read_csv('ModelUncertainties.csv')
uns['vgg_class_un_norm'] = uncertainty_norm(uns['vgg_class_un'])
uns['vgg_is_correct'] = uns['vgg_class_pred'] == uns['vgg_true_class']

uns['xc_class_un_norm'] = uncertainty_norm(uns['xc_class_un'])
uns['xc_is_correct'] = uns['xc_class_pred'] == uns['xc_true_class']

uns['all_correct'] = uns['xc_is_correct'] & uns['vgg_is_correct']

uns['both_wrong'] = ~(uns['xc_is_correct'] | uns['vgg_is_correct'])#False when both models predict wrongly

uns['vgg_xc_identical'] =  uns['xc_class_pred'] == uns['vgg_class_pred']

uns['not_warned'] = uns['all_correct'] | (uns['both_wrong'] & uns['vgg_xc_identical'])#True when both models predict wrongly and both models predict identical class

weight_xc = -0.7 #Value found during experimentation in script CombinedUncertainty.py
uns['combined_uncertainty'] = uns['xc_class_un_norm'] * weight_xc + uns['vgg_class_un_norm'] * (1 - weight_xc)
uns['combined_uncertainty'] = uncertainty_norm(uns['combined_uncertainty'])


thresholds_combined = np.arange(0.7, 1, 0.05)#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresholds_vgg = np.arange(0.06, 0.2, 0.02)#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model_stats= []
both_wrong = uns[uns['both_wrong'] == True]
both_correct = uns[uns['all_correct'] == True]
not_warned = both_wrong[both_wrong['not_warned'] == True]#Operator not warned and both models predict identical (wrong) class
for t_combined in thresholds_combined:
    for t_vgg in thresholds_vgg:
        not_warned_1 = not_warned[not_warned['combined_uncertainty'] < t_combined] 
        not_warned_1 = not_warned_1[not_warned_1['vgg_class_un'] < t_vgg ]
        warned_1 = both_correct[(both_correct['combined_uncertainty'] > t_combined) | (both_correct['vgg_class_un'] > t_vgg)]
        num_false_negatives = len(not_warned_1)#Operator still not warned and both models predict identical (wrong) class
        num_false_positives = len(warned_1)#Operator warned althoug both models predict correctly
        model_stats.append([t_combined, t_vgg, num_false_positives, num_false_negatives])


model_stats = pd.DataFrame(model_stats, columns = ['threshold_combined', 'threshold_vgg', 'false_positives', 'false_negatives'])

for t in thresholds_combined:
    trace1 = go.Scatter(x = model_stats[model_stats['threshold_combined'] ==t]['threshold_vgg'], y = model_stats[model_stats['threshold_combined'] ==t]['false_positives'], name='False Positives', marker=dict(color='blue'))
    trace2 = go.Scatter(x = model_stats[model_stats['threshold_combined'] ==t]['threshold_vgg'], y = model_stats[model_stats['threshold_combined'] ==t]['false_negatives'], name='False Negatives', marker=dict(color='red'))
    
    # Combine traces into a figure
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(title = f'False Positives and False Negatives for Different Thresholds of VGG Uncertainty - Threshold Combined Uncertainty: {t}', 
                      xaxis_title = 'Threshold VGG', 
                      yaxis_title = 'Count'
                     )
    fig.show()
    fig.write_image(f"FindThreshold_T_Combined_{int(t*100)}.png")


model_stats.to_excel('ModelStatsCombined.xlsx')


