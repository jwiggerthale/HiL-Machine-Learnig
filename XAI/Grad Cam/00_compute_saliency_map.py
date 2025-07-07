
#This file implements grad cam to calculate saliency maps for the images
#Function implemented in the script is basis for further project 
import tensorflow as tf

'''
Function which calculates gradients in a model with respect to predicted class
Call with: 
    model: model to be used
    dataset
'''
def compute_grads(model,  
                  ims: images to examine (batch of dataset):
    with tf.GradientTape() as tape:
        tape.watch(ims)
        result = model(ims)
        max_idx = tf.argmax(result[4],axis = 1)
        max_score = result[4][0,max_idx[0]]
    grads = tape.gradient(max_score, ims)
    return result, grads
