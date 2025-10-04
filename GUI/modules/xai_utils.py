'''
This script implements different utils for xai and uq 
'''
import numpy as np
from PIL import Image, ImageTk
from .utils import SharedState
import matplotlib.pyplot as plt
import tensorflow as tf


'''
Fnction which computes gradients of model predictions (vis frame)
Call with: 
  model --> model to use 
  ims --> list of image(s) 
Returns: 
  result --> model predictions
  grads --> gradients with respect to predicted class
'''
def compute_grads(model, ims):
    with tf.GradientTape() as tape:
        tape.watch(ims)
        result = model(ims)
        max_idx = tf.argmax(result[4], axis=1)
        max_score = result[4][0, max_idx[0]]

    grads = tape.gradient(max_score, ims)
    return result, grads

'''
Function which makes predictions on image using mc dropout
call with: 
  model --> model to use
  x --> sample to predict on 
  num_samples: int --> number of mc samples
returns: 
  pred --> tuple of image predictions with bounding box and class
'''
def mc_predict(model, x, num_samples: int = 20):
    class_preds, x_min_preds, y_min_preds, x_max_preds, y_max_preds = [], [], [], [], []

    for _ in range(num_samples):
        x_min, y_min, x_max, y_max, pred_class = model(x, training=True)
        class_preds.append(pred_class)
        x_min_preds.append(x_min)
        y_min_preds.append(y_min)
        x_max_preds.append(x_max)
        y_max_preds.append(y_max)

    # Convert to numpy arrays
    class_preds = np.array(class_preds)
    x_min_preds = np.array(x_min_preds)
    y_min_preds = np.array(y_min_preds)
    x_max_preds = np.array(x_max_preds)
    y_max_preds = np.array(y_max_preds)

    # Compute mean predictions and uncertainties
    pred_classes = [elem.argmax() for elem in class_preds.mean(axis=0)]
    pred_std = class_preds.std(axis=0)
    pred_un = [pred_std[i][c] for i, c in enumerate(pred_classes)]

    return (
        x_min_preds.mean(axis=0), x_min_preds.std(axis=0),
        y_min_preds.mean(axis=0), y_min_preds.std(axis=0),
        x_max_preds.mean(axis=0), x_max_preds.std(axis=0),
        y_max_preds.mean(axis=0), y_max_preds.std(axis=0),
        pred_classes, pred_un
    )


'''
Function which generates and shows saliency image (vis frame)
Call with: 
  state: SharedState --> shared state between frames
  alpha1: float --> alpha for overlay (vgg)
  alpha2: float --> alpha for overlay (xception)
  brightness: float --> brightness of plot
  display_size: tuple --> size of images
'''
def generate_saliency_image(state: SharedState, 
                            alpha1: float = 0.5,
                            alpha2: float = 0.5, 
                            brightness: float = 0.6, 
                            display_size: tuple = (400, 400)):

    if state.imageu is None or state.vgg_grad is None or state.xcep_grad is None:
        return None

    image_np = state.imageu[0].numpy()
    grads_vgg = state.vgg_grad[0].numpy()
    grads_xcep = state.xcep_grad[0].numpy()

    saliency_vgg = np.max(np.abs(grads_vgg), axis=-1)
    saliency_xcep = np.max(np.abs(grads_xcep), axis=-1)

    if np.all(saliency_vgg == 0):
        return ImageTk.PhotoImage(Image.fromarray((image_np * 255).astype(np.uint8)))

    saliency_vgg = (saliency_vgg - saliency_vgg.min()) / (saliency_vgg.max() - saliency_vgg.min() + 1e-8)
    saliency_xcep = (saliency_xcep - saliency_xcep.min()) / (saliency_xcep.max() - saliency_xcep.min() + 1e-8)

    saliency_vgg *= brightness
    dimmed_img = image_np * brightness

    cmap_vgg = plt.get_cmap('jet')
    cmap_xcep = plt.get_cmap('hot')
    sal_color_vgg = cmap_vgg(saliency_vgg)[..., :3]
    sal_color_xcep = cmap_xcep(saliency_xcep)[..., :3]
    overlay = dimmed_img + (sal_color_vgg * alpha1) + (sal_color_xcep * alpha2)
    overlay = np.clip(overlay, 0, 1)

    img_pil = Image.fromarray((overlay * 255).astype(np.uint8)).resize(display_size)
    return ImageTk.PhotoImage(img_pil)
