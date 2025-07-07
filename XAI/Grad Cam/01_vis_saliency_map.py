#This script implements a basic usage of the function compute_grads from script calculate_saliency_map
#Gradients are calculated and results are visualized: 
#  1) Visualization of basic image
#  2) Visualization of saliency map
#  3) Visualization of image with overlay of saliency map
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import glob
import random
from lxml import etree
from utils.data_utils import get_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
from compute_saliency_map import compute_grads


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)




im_path  = "/data/HiL_XAI/SteelLocation/images/*/*.jpg"
xml_path = "/data/HiL_XAI/SteelLocation/label/*.xml"
batch_size = 32
train_split = 0.8

train_dataset, test_dataset, Class_dict, train_count, test_count = get_dataset(im_path = im_path, 
                                                                                label_path=xml_path, 
                                                                                batch_size = batch_size, 
                                                                                train_share  = train_split)
class_dict = {v:k for k,v in Class_dict.items()}



base_model = tf.keras.applications.xception.Xception(weights='xception_base.h5',
                                            include_top=False,
                                            input_shape=(224,224,3),
                                            pooling='avg')


base_model.trainable = True

inputs = keras.Input(shape = (224,224,3))
x = base_model(inputs)

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

ims, labels = test_dataset.take(1)
pred, grads = compute_grads(model, ims)
