#This file uses the mc_predict function from 00_monte_carlo_dropout to make predictions with uncertainty on some exemplary images
#Model predictions are plotted and saved as image

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.data_utils import get_dataset
from 00_monte_carlo_dropout import mc_predict


#Get dataset
im_path  = "/data/HiL_XAI/SteelLocation/images/*/*.jpg"
xml_path = "/data/HiL_XAI/SteelLocation/label/*.xml"
batch_size = 32
train_split = 0.8

train_dataset, test_dataset, Class_dict, train_count, test_count = get_dataset(im_path = im_path, 
                                                                                label_path=xml_path, 
                                                                                batch_size = batch_size, 
                                                                                train_share  = train_split)

class_dict = {v: k for k, v in Class_dict.items()}

#Create base model, adapt to the model architecture you use as baseline
base_model = tf.keras.applications.ResNet50(weights='resnet50_base.h5',
                                            include_top=False,
                                            input_shape=(224,224,3),
                                            pooling='avg')

base_model.trainable = True
inputs = keras.Input(shape = (224,224,3))
x = base_model(inputs)

#Add layers for classification and bounding box prediction
x1 = keras.layers.Dense(1024, activation = "relu")(x)
x1 = keras.layers.Dense(512, activation = "relu")(x1)
out1 = keras.layers.Dense(1, name = "xmin")(x1)
out2 = keras.layers.Dense(1, name = "ymin")(x1)
out3 = keras.layers.Dense(1, name = "xmax")(x1)
out4 = keras.layers.Dense(1, name = "ymax")(x1)

x2 = keras.layers.Dense(1024,activation = "relu")(x)
x2 = keras.layers.Dropout(0.5)(x2)
x2 = keras.layers.Dense(512,activation = "relu")(x2)
out_class = keras.layers.Dense(10,activation = "softmax", name = "class")(x2)

out = [out1, out2, out3, out4, out_class]

model = keras.models.Model(inputs = inputs, outputs = out)

#load model weights, adapt filepath
model.load_weights('best_model_05.h5')

#Get predictions and uncertainties for prediction
for ims, labels in test_dataset.take(1):
    im = ims[0]
    x_min_pred, x_min_un, y_min_pred, y_min_un, x_max_pred, x_max_un, y_max_pred, y_max_un, pred_classes, pred_un = mc_predict(model = vgg16, 
                                                                                                                             x = ims, 
                                                                                                                            num_samples = 20)

#Visualize model performance for 3 images
plt.figure(figsize = (10, 24))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.imshow(keras.preprocessing.image.array_to_img(ims[i]))
    pred_class = class_dict[pred_classes[i] + 1]
    true_class = class_dict[np.argmax(label[4][i]) + 1]
    plt.title(f'True Class: {true_class} - Predicted Class: {pred_class} - Uncertainty: {pred_un[i]}')
    x_min, y_min, x_max, y_max = x_min_pred[i]*224, y_min_pred[i]*224, x_max_pred[i]*224, y_max_pred[i]*224
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color = "red", label = 'Predicted Bounding Box')
    x_min, y_min, x_max, y_max = labels[0][i]*224, labels[1][i]*224, labels[2][i]*224, labels[3][i]*224
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color = "green", label = 'True Bounding Box')
    plt.legend()
    plt.savefig('Model predictions.png')



