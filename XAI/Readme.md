# XAI and UQ for more safety and trustworthiness

In this folder, we implement XAI-techniques and uncertainty-quantification (UQ) methods to make the model output more interpretable and explainable. 

## Saliency Map
In terms of XAI, we implement saliency maps using GradCAM. Thereby, we obtain heatmaps showing which pixels in the image have a high impact on the model's prediction. 
Beyond that, we implement integrated gradients allowing detailed inference on pixel attributions. 

## Monte Carlo Dropout
For UQ, we implement Monte Carlo (MC) dropout. By calculating the uncertainty of model for all samples of the training set and normalizing scores to range 0 - 1, we are able to define borders for when operator intervention is necessary.

The distribution of uncertainties of the VGG16-model is shown below: 

![alt text](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/XAI/Monte%20Carlo%20Dropout/VGG16Uncertainties.png)

It can be seen that the uncertainties on wrong predictions are in general way higher than on corect predictions. This property allows for good seperation between correct and wrong predictions by looking at the uncertainty. 

In contrast, model uncertainties of Xception model do not show such a clear trend: 

![alt text](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/XAI/Monte%20Carlo%20Dropout/XceptionUncertainties.png)

The figure shows that uncertainties for wrong prediction tend to be higher but the differences between wrong and correct predictions are not as clear. Therefore, we use the VGG16-uncertainty scores as indicator when model to inform the operator. 

## Model-Combination for More Reliability
As an additional measure, we inform the operator when the two models predict differently. Beyond that, we use a combined model uncertainty score. To identify an appropriate threshold for that score, we identify all images where both models predict wrongly. This is based on the assumption that images where one of the two models predicts wrongly are already identified as the other model predicts correctly and the operator is therfore warned due to the conflicting predictions.

