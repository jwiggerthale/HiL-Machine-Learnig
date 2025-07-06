This folder contains code for training different models on the *defects location for metal surface* dataset (available on kaggle: https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location)
Models are trained in PyTorch. In particular, we implement the following models: 
- VGG16
- YOLOv11
- Xception
- ResNet50

The training scripts as well as links to model weights in Google Cloud can be found in the apprpriate folders. Regarding ResNet50, it should be noted that the model does not achieve sufficient performance and is therefore neglected in the further course of the project. Alos, YOLOv11 will not be utilized further as the framwework that was used to implement the model is not flexible enough for our needs.
