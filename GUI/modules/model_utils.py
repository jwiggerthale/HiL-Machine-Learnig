'''
This script implements different utils for model handling
'''

from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import tensorflow as tf # pyright: ignore[reportMissingImports]
import datetime
from tensorflow import keras # type: ignore
import os
import pandas as pd # type: ignore
from .settings import *
from .utils import SharedState
from .xai_utils import compute_grads, mc_predict


'''
Function which creates tensorflow model 
Call with: 
  num_classes: int --> number of classes to predict
  mode: str --> whether to create vgg or xception
returns
  model --> compiled model
'''
def build_model(num_classes: int = 10, 
                mode: str = "Vgg"):
    if mode == "Vgg":
        base = tf.keras.applications.VGG16(weights=None, include_top=False,
                                           input_shape=(224,224,3), pooling='avg')
    elif mode == "Xception":
        base = tf.keras.applications.Xception(weights=None, include_top=False,
                                              input_shape=(224,224,3), pooling='avg')
    else:
        raise ValueError("mode must be 'Vgg' or 'Xception'")

    base.trainable = True
    inputs = keras.Input((224,224,3))
    x = base(inputs)

    # Bounding box branch
    x1 = keras.layers.Dense(1024, activation="relu")(x)
    x1 = keras.layers.Dropout(0.5)(x1)
    x1 = keras.layers.Dense(512, activation="relu")(x1)
    x1 = keras.layers.Dropout(0.5)(x1)

    out1 = keras.layers.Dense(1, name="xmin")(x1)
    out2 = keras.layers.Dense(1, name="ymin")(x1)
    out3 = keras.layers.Dense(1, name="xmax")(x1)
    out4 = keras.layers.Dense(1, name="ymax")(x1)

    # Classification branch
    x2 = keras.layers.Dense(1024, activation="relu")(x)
    x2 = keras.layers.Dropout(0.5)(x2)
    x2 = keras.layers.Dense(512, activation="relu")(x2)
    x2 = keras.layers.Dropout(0.5)(x2)
    out_class = keras.layers.Dense(num_classes, activation="softmax", name="class")(x2)

    model = keras.models.Model(inputs=inputs, outputs=[out1, out2, out3, out4, out_class])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss={
            'xmin': 'mse',
            'ymin': 'mse',
            'xmax': 'mse',
            'ymax': 'mse',
            'class': 'categorical_crossentropy'
        },
        metrics={
            'xmin': 'mae',
            'ymin': 'mae',
            'xmax': 'mae',
            'ymax': 'mae',
            'class': 'accuracy'
        }
    )

    return model


'''
Fucntion for training a model
call with: 
  model --> model to be trained 
  callback_update --> function for callback_updates
  train_dataset --> dataset with train images
  test_dataset --> dataset with test images
  train_count: int --> number of train images
  test_count: int --> number of test images
  save_path: str --> path to save model
  mode: str --> whether to train xception or vgg
returns: 
  save_path: str --> path where model is saved
'''
def train_model(model, 
                callback_update,
                train_dataset, 
                test_dataset,
                train_count: int,
                test_count: int, 
                save_path: str, 
                mode: str ="Vgg"):
    lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if save_path is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"{mode.lower()}_{now}.h5"
    if mode.lower() == 'vgg':
        save_dir = vgg_dir
    else:
        save_dir = xception_dir
    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, save_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=full_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    
    class CancelableConsoleLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if callback_update:
                msg = f"Epoch {epoch+1}: " + ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
                callback_update(msg)

            # 🔴 Check cancel event
            if stop_training_event.is_set():
                print("🛑 Training cancellation requested. Stopping...")
                self.model.stop_training = True

    
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_count // batch_size,
        validation_data=test_dataset,
        validation_steps=test_count // batch_size,
        epochs=1,
        callbacks=[checkpoint, CancelableConsoleLogger()]
    )

    # now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pd.DataFrame(history.history).to_csv(os.path.join(save_dir, f'TrainingHistory_{now}.csv'), index=False)
    return full_save_path


'''
Fucntion which loads vgg model 
Call with: 
  location: str --> path to model
'''
def vggfun(location: str):
    model = load_model(location, compile=False)
    return model            

'''
Fucntion which loads xception model 
Call with: 
  location: str --> path to model
'''
def xcepfun(location: str):
    model = load_model(location, compile=False)
    return model
    


'''
Fucntion for predicting using tow models (vgg and xception)
call with: 
  vgg_model --> vgg model to use
  xception_model --> xception model to use
  img_tensor --> image to predict on 
  shared_state: SharedState --> shared information between frames
'''
def predict_with_both_models(vgg_model,
                             xception_model, 
                             img_tensor, 
                             shared_state: SharedState):
    
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    print('VGG')
    print('Computing Grads')
    # Prediction using VGG16
    pred_vgg, grads_vgg = compute_grads(vgg_model, img_tensor)
    print('Predicting MC')
    x_min_pred_vgg, _, y_min_pred_vgg, _, x_max_pred_vgg, _, y_max_pred_vgg, _, pred_classes_vgg, pred_un_vgg = mc_predict(vgg_model, img_tensor, 1)
    print('Xception')
    print('Computing Grads')
    # Prediction using Xception
    pred_xcep, grads_xcep = compute_grads(xception_model, img_tensor)
    print('Predicting MC')
    x_min_pred_xcep, _, y_min_pred_xcep, _, x_max_pred_xcep, _, y_max_pred_xcep, _, pred_classes_xcep, pred_un_xcep = mc_predict(xception_model, img_tensor, 1)
    print('Predicted MC')
    # Store both results in shared_state
    shared_state.imageu = img_tensor
    shared_state.vgg_result = [x_min_pred_vgg, y_min_pred_vgg, x_max_pred_vgg, y_max_pred_vgg, pred_classes_vgg]
    shared_state.xcep_result = [x_min_pred_xcep, y_min_pred_xcep, x_max_pred_xcep, y_max_pred_xcep, pred_classes_xcep]
    shared_state.vgg_grad = grads_vgg
    shared_state.xcep_grad = grads_xcep
    shared_state.vgg_uncertainty = pred_un_vgg[0]
    shared_state.xcep_uncertainty = pred_un_xcep[0]

    print("Prediction done.")
    return shared_state



'''
Class map for model prediction
'''
class_map_vgg16 = {
    'punching_hole': 1, 'welding_line': 2, 'crescent_gap': 3,
    'water_spot': 4, 'oil_spot': 5, 'silk_spot': 6,
    'inclusion': 7, 'rolled_pit': 8, 'crease': 9,
    'waist folding': 10
}

'''
Class map for model prediction (reverse)
'''
class_dictvgg16 = {v: k for k, v in class_map_vgg16.items()}

'''
All class names
'''
class_names = list(class_dictvgg16.values())

