This folder contains code for training different models on the *defects location for metal surface* dataset (available on kaggle: https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location)
Models are trained in Tensorflow. In particular, we implement the following models: 
- VGG16
- Xception
- ResNet50

The training scripts can be found in the apprpriate folders. Regarding ResNet50, it should be noted that the model does not achieve sufficient performance and is therefore neglected in the further course of the project. The weights of VGG16 and Xception model are available in Colab: https://drive.google.com/drive/folders/1j99Ip_hUOnBTSxRTWITkeuwc6jCYtLVN?usp=drive_link

In addition to the above mentioned models, we train YOLOv11 using ultralytics. However, YOLOv11 will not be utilized further as the framwework that was used to implement the model is not flexible enough for our needs.
