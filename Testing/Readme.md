# Overview
This folder contain the code and files for testing phases two and three of the workflow. 
To test, we took the baseline models and applied them to labeled test images. 

# Cross-Checking
In the phase *cross-checking*, we apply the model to new images without label. However, the operator remains entirely in the process and labels every single image.  
Whenever the model predicts wrongly on a given image or is uncertain regarding its prediction, the operator draws a new bounding box and assigns the correct label to the image. 
The image is then automatically added to the database. If the model predicts correctly, the image is added to the dataset as well. That way, the database grows continuously. 

As soon as a sufficient amount of new images is available, the model is retrained. Performance metrics are visualized shwoing a comparison of the modle before and after retraining. 

Below, we show how accuracy can increase whenadding just 300 new images to the dataset. 

<img src="Acc Train Samples.jpg" alt="Structure of Development of accuracy when increasing the number of available training samples" style="width:500px; vertical-align:middle;">

It can be seen that accuracy increases by ~ 14 percentage points by adding just 300 images to the dataset. In many real world scenarios, 300 images are generated within a few minutes. 
A major advantage of the approach is the fact that it a self accelerating process. At first, the operator has to drwa bounding boxes for ~ 30 of 100 images. After 300 images, he only has to label ~ 15 of 100 images. This example shows how efficient the approach is and how quick very reliable models can be created using the approach. 

# Operator Validation
Once the model achieves predefined performance, our tests enter the last phase of the workflow. 
Here, the model acts on its own for the first time. In particular, we apply the model to unlabeled images in process. Only uncertain predictions are retained and checked by the operator. 
Beyond that, random images are selected for operator validation with a probability of 0.05. 
In order to show how long it takes until performance degradation is recognized, we introduce bias into the model making it prefer predicting a certain class. 

