# XAI and UQ for more safety and trustworthiness

In this folder, we implement XAI-techniques and uncertainty-quantification (UQ) methods to make the model output more interpretable and explainable. 

## Saliency Map
In terms of XAI, we implement saliency maps using GradCAM. Thereby, we obtain heatmaps showing which pixels in the image have a high impact on the model's prediction. 
Beyond that, we implement integrated gradients allowing detailed inference on pixel attributions. 

## Monte Carlo Dropout
For UQ, we implement Monte Carlo (MC) dropout. The function is implemented in the folder under point 00. 

Based on MC dropout, we visualize some models predictions in a first step (files with prefix 01). Eemplary model predictions of the VGG16 model are shown below: 

![alt text](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/XAI/Monte%20Carlo%20Dropout/01_model_predictions_VGG16.png)

Using the implementation, we calculate the uncertainty of models for all samples of the training set and normalizing scores to range 0 - 1 (files with prefix 02). 


We then visualize the results (files with prefix 03). The distribution of uncertainties of the VGG16-model is shown below: 

![alt text](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/XAI/Monte%20Carlo%20Dropout/03_VGG16Uncertainties.png)

It can be seen that the uncertainties on wrong predictions are in general way higher than on corect predictions. This property allows for good seperation between correct and wrong predictions by looking at the uncertainty. 

In contrast, model uncertainties of Xception model do not show such a clear trend: 

![alt text](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/XAI/Monte%20Carlo%20Dropout/03_XceptionUncertainties.png)

The figure shows that uncertainties for wrong prediction tend to be higher but the differences between wrong and correct predictions are not as clear. Therefore, we use the VGG16-uncertainty scores as indicator when to inform the operator. 

## Model-Combination for More Reliability
As an additional measure, we inform the operator when the two models predict differently. Beyond that, we use a combined model uncertainty score. The combined score and its basic examination is implemented in files with prefix 04.

Finally, we identify appropriate threholds for VGG16 model uncertainty and the combined uncertainty. Here, we strive to minimize false postives and false negatives. The exact thresholds should be adapted to the criticality of the use case. Implementation and results of the step can be found in files with prefix 05. 
