
#This file uses the mc_predict function to get model uncertainties for all data points from the test set
#Model predictions and uncertainties are stored for each model in a csv file
#Also, we create a csv-file with predictions and uncertainties of both models 
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.data_utils import get_dataset
import pandas as pd
from 00_monte_carlo_dropout import mc_predict

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
base_model = tf.keras.applications.xception.Xception(weights=None,
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
model.load_weights('best_model_do_05_2.h5')

#Get predictions and uncertainties for prediction
x_min_preds = []
x_min_uns = []
y_min_preds = []
y_min_uns = []
x_max_preds = []
x_max_uns = []
y_max_preds = []
y_max_uns = []
pred_classes = []
pred_uns = []
all_classes = []
for ims, labels in test_dataset:
    x_min_pred, x_min_un, y_min_pred, y_min_un, x_max_pred, x_max_un, y_max_pred, y_max_un, pred_class, pred_un = mc_predict(model = model, 
                                                                                                                             x = ims, 
                                                                                                                            num_samples = 20)
    x_min_preds.extend(x_min_pred.reshape(-1).tolist())
    x_min_uns.extend(x_min_un.reshape(-1).tolist())
    y_min_preds.extend(y_min_pred.reshape(-1).tolist())
    y_min_uns.extend(y_min_un.reshape(-1).tolist())
    x_max_preds.extend(x_max_pred.reshape(-1).tolist())
    x_max_uns.extend(x_max_un.reshape(-1).tolist())
    y_max_preds.extend(y_max_pred.reshape(-1).tolist())
    y_max_uns.extend(y_max_un.reshape(-1).tolist())
    pred_classes.extend(pred_class)
    pred_uns.extend(pred_un)
    all_classes.extend(int(elem.numpy().argmax()) for elem in labels[4])


results = pd.DataFrame()
results['x_min_pred'] = x_min_preds
results['x_min_un'] = x_min_uns
results['y_min_pred'] = y_min_preds
results['y_min_un'] = y_min_uns
results['x_max_pred'] = x_max_preds
results['x_max_un'] = x_max_uns
results['y_max_pred'] = y_max_preds
results['y_max_un'] = y_max_uns
results['class_pred'] = pred_classes
results['class_un'] = pred_uns
results['true_class'] = all_classes

results.to_csv('ModelUncertainties.csv', index = False)
