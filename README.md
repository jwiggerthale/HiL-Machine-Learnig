# HiL-Machine-Learnig
This repo contains best practices for human in the loop (HiL) Machine Learning targeted at creating safe ML-applications for visual inspection. 
It is based on a process proposed by us recently involving XAI and uncertainty quantifiation (UQ).

# Process
The process consists of different phases in which a ML model is initially created and improved iteratively 
in a HiL approach afterwadrds until its performance is sufficient for standalone application. The process is visualized below: 

![image](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/Process%20Overview.png)


## Baseline Model
The first phase is about creating a baseline model. In this phase, labeled images for the given inspetion task have to be created firstly. Based on that, the model is initially trained. The GUI implemented in this repo contains a labeling tool allowing for efficient generation of annotaed images (class and bounding box). The scripts in the folder model_training implement different CNNs in PyTorch. 

## Cross Checking
In the second phase, the initially created models are applied in the inspection task along with manual inspection by the operator. In this phase, the operator as well as the model inspect every single element. When it comes to contradictory assessments or high model uncertainty, the operator receives an information and has to relabel the image in order to generate new training data. To get a better understanding of why the model predicted wrongly or is uncertain, he can visualize the saliency map showing what regions in the image are of high relevance for the model's prediction. 

To facilitate these tasks, the repo implements: 
1) Saliency map (in folder XAI)
2) GUI integrating:
     a) Visualization of image with predicted bounding box and saliency map
     b) Visualiation of uncertainty
     c) Efficient tool for labeling
     d) Trigger for retraining the model with performance visualization

## Operator Validation
The final phase of the process consists of operator validation. The phase can be entered when the model achieves the desired performance on the givem task. In this phase, the operator's workload is efficiently reduced by the model. Here the model inspects every single object. Whenever the model is uncertain or an object is classified as defect, the image is added to a task queue. Additionally, random images are added to that queue. Whenever the operator has time, he checks images from the task queue and relabels them if necessary. The task queue is implemented in the GUI. By randomly adding images to the queue although the model is not uncertain, performance degradation is avoided. 



