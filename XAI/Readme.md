In this folder, we implement XAI-techniques and uncertainty-quantification (UQ) methods to make the model output more interpretable and explainable. 

In terms of XAI, we implement saliency maps using GradCAM. Thereby, we obtain heatmaps showing which pixels in the image have a high impact on the model's prediction. 
Beyond that, we implement integrated gradients allowing detailed inference on pixel attributions. 

For UQ, we implement Monte Carlo (MC) dropout. By calculating the uncertainty of model for all samples of the training set and normalizing scores to range 0 - 1, we are able to define borders for when operator intervention is necessary.
