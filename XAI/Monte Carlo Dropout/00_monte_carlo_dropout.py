#This file implenets monte carlo dropout
#Basis for other scripts in this folder

import numpy as np


'''
Function which takes model for monte carlo dropout prediction
Call with: 
    model: model to use
    x: sample to make prediction on
    num_samples: number of forward passes to use for monte carlo dropout
Returns: 
    Mean predictions --> bounding box and class
    Std of predictions --> bounding box and class
'''
def mc_predict(model, 
               x, 
              num_samples):
    class_preds = []
    x_min_preds = []
    y_min_preds = []
    x_max_preds = []
    y_max_preds = []

    for _ in range(num_samples):
        x_min, y_min, x_max, y_max, pred_class = model(x,training=True)
        class_preds.append(pred_class)
        x_min_preds.append(x_min)
        y_min_preds.append(y_min)
        x_max_preds.append(x_max)
        y_max_preds.append(y_max)

    class_preds = np.array(class_preds)
    x_min_preds = np.array(x_min_preds)
    y_min_preds = np.array(y_min_preds)
    x_max_preds = np.array(x_max_preds)
    y_max_preds = np.array(y_max_preds)

    pred_classes = [elem.argmax() for elem in class_preds.mean(axis = 0)]
    pred_std = class_preds.std(axis = 0)
    pred_un = [pred_std[i][c] for i, c in enumerate(pred_classes)]

    x_min_pred = x_min_preds.mean(axis = 0)
    x_min_un = x_min_preds.std(axis = 0)
    y_min_pred = y_min_preds.mean(axis = 0)
    y_min_un = y_min_preds.std(axis = 0)
    x_max_pred = x_max_preds.mean(axis = 0)
    x_max_un = x_max_preds.std(axis = 0)
    y_max_pred = y_max_preds.mean(axis = 0)
    y_max_un = y_max_preds.std(axis = 0)

    return x_min_pred, x_min_un, y_min_pred, y_min_un, x_max_pred, x_max_un, y_max_pred, y_max_un, pred_classes, pred_un
