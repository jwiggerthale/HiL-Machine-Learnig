In this folder, we implement XAI-techniques and uncertainty-quantification (UQ) methods to make the model output more interpretable and explainable. 

In terms of XAI, we implement saliency maps using GradCAM. Thereby, we obtain heatmaps showing which pixels in the image have a high impact on the model's prediction. 
Beyond that, we implement integrated gradients allowing detailed inference on pixel attributions. 

For UQ, we implement Monte Carlo (MC) dropout. By calculating the entropy within the predictions and dividing by log(k), 
we obtain a normalized uncertainty score allowing define limits for when to check the model's output manually. 
