# Overview
This folder contain the code and files for testing phases two and three of the workflow. 
To test, we took the baseline models and applied them to labeled test images. 

# Cross-Checking
In the phase *cross-checking*, we apply the model to new images without label. However, the operator remains entirely in the process and labels every single image.  
Whenever the model predicts wrongly on a given image or is uncertain regarding its prediction, the operator draws a new bounding box and assigns the correct label to the image. 
The image is then automatically added to the database. 

As soon as 100 new images are available, the model is retrained. Performance metrics are visualized shwoing a comparison of the modle before and after retraining. 

Regarding the process in the experiments conducted, it has to be mentioned that it is slightly different from the real process. In our test, labels are already available. 
In real world scenarios, they would be created by the operator during inspection. 

# Operator Validation
Once the model achieves predefined performance, our tests enter the last phase of the workflow. 
Here, the model acts on its own for the first time. In particular, we apply the model to unlabeled images in process. Only uncertain predictions are retained and checked by the operator. 
Beyond that, random images are selected for operator validation with a probability of 0.05. 
In order to show how long it takes until performance degradation is recognized, we introduce bias into the model making it prefer predicting a certain class. 

