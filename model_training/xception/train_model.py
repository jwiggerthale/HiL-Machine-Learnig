
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import glob
import random
from lxml import etree
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer



#Enable GPU usage if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#Replace by your path to images and labels
image_path = glob.glob("/data/HiL_XAI/SteelLocation/images/*/*.jpg")
xmls_path = glob.glob("/data/HiL_XAI/SteelLocation/label/*.xml")

print(f'Found {len(image_path)} images and {len(xmls_path)} paths')


#Sort paths to make sure train and test images are identical each time
xmls_path.sort(key = lambda x:x.split("/")[-1].split(".xml")[0])
image_path.sort(key = lambda x:x.split("/")[-1].split(".jpg")[0])

#Only take the images for which labels are available
xmls_train = [path.split("/")[-1].split(".")[0] for path in xmls_path]
imgs_train = [img for img in image_path if (img.split("/")[-1].split)(".jpg")[0] in xmls_train]
print(f'Found {len(imgs_train)} images and {len(xmls_path)} labels')


#Get dictionary for labels to convert class prediction to defect type
labels = [label.split("/")[-2] for label in imgs_train]
labels = pd.DataFrame(labels, columns = ["Defect Type"])
Class = labels["Defect Type"].unique()
Class_dict = dict(zip(Class, range(1,len(Class) + 1)))
labels["Class"] = labels["Defect Type"].apply(lambda x: Class_dict[x])

#Binarize labels
lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
# Convert multi-class labels to binary labels (belong or does not belong to the class)
transformed_labels = lb.transform(labels["Class"])
y_bin_labels = []

for i in range(transformed_labels.shape[1]):
    y_bin_labels.append("Class" + str(i))
    labels["Class" + str(i + 1)] = transformed_labels[:, i]

#Drop unnecessary columns from df with binarized labels
labels.drop("Class", axis = 1, inplace = True)
labels.drop("Defect Type", axis = 1, inplace = True)

#Get labels from xml file
#Call with path to xml file
def to_labels(path):
    # Read the annotation file
    xml = open("{}".format(path)).read()
    sel = etree.HTML(xml)
    # Obtain the image width
    width = int(sel.xpath("//size/width/text()")[0])
    # Obtain the image height
    height = int(sel.xpath("//size/height/text()")[0])
    # Extract the bounding box coordinates
    xmin = int(sel.xpath("//bndbox/xmin/text()")[0])
    xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
    ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
    ymax = int(sel.xpath("//bndbox/ymax/text()")[0])
    # Return the relative coordinates
    return [xmin/width, ymin/height, xmax/width, ymax/height]



#Get coordinates for biunding boxes
coors = [to_labels(path) for path in xmls_path]
xmin, ymin, xmax, ymax = list(zip(*coors))
xmin = np.array(xmin)
ymin = np.array(ymin)
xmax = np.array(xmax)
ymax = np.array(ymax)
label = np.array(labels.values, dtype = np.float16)

#Dataset with labels
labels_dataset = tf.data.Dataset.from_tensor_slices((xmin, ymin, xmax, ymax, label))


#Function to load image from path
#Call with path to image
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,3)
    image = tf.image.resize(image,[224,224])
    image = tf.cast(image,tf.float32)
    image = image / 255
    return image

#Dataset for images
dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
dataset = dataset.map(load_image)

#Dataset with images and labels
dataset_label = tf.data.Dataset.zip((dataset, labels_dataset))


#Create dataset with prefetching
batch_size = 32
dataset_label = dataset_label.repeat().shuffle(500).batch(batch_size)
dataset_label = dataset_label.prefetch(tf.data.experimental.AUTOTUNE)

#Split dataset into train and test
train_count = int(len(imgs_train) * 0.8)
test_count = int(len(imgs_train) * 0.2)
train_dataset = dataset_label.skip(test_count)
test_dataset = dataset_label.take(test_count)

#Inverse class dict
class_dict = {v:k for k,v in Class_dict.items()}

#Create model
base_vgg16 = keras.applications.xception.Xception(weights='xception_base.h5',#PAth to pretrained weights (if local) otherwise 'imagenet'
                                            include_top=False,
                                            input_shape=(224,224,3),
                                            pooling='avg')
base_vgg16.trainable = True

inputs = keras.Input(shape = (224,224,3))
x = base_vgg16(inputs)

x1 = keras.layers.Dense(1024, activation = "relu")(x)
x1 = keras.layers.Dropout(0.5)(x1)
x1 = keras.layers.Dense(512, activation = "relu")(x1)
x1 = keras.layers.Dropout(0.5)(x1)
out1 = keras.layers.Dense(1, name = "xmin")(x1)
out2 = keras.layers.Dense(1, name = "ymin")(x1)
out3 = keras.layers.Dense(1, name = "xmax")(x1)
out4 = keras.layers.Dense(1, name = "ymax")(x1)

x2 = keras.layers.Dense(1024,activation = "relu")(x)
x2 = keras.layers.Dropout(0.5)(x2)
x2 = keras.layers.Dense(512,activation = "relu")(x2)
x2 = keras.layers.Dropout(0.5)(x2)
out_class = keras.layers.Dense(10,activation = "softmax", name = "class")(x2)

out = [out1, out2, out3, out4, out_class]

vgg16 = keras.models.Model(inputs = inputs, outputs = out)

#Compile and train model
vgg16.compile(keras.optimizers.Adam(0.0003),
              loss={'xmin':'mse',
                    'ymin':'mse',
                    'xmax':'mse',
                    'ymax':'mse',
                    'class':'categorical_crossentropy'},
              metrics=['mae', 'mae', 'mae', 'mae','acc'])

lr_reduce = keras.callbacks.ReduceLROnPlateau("val_loss", patience = 5, factor = 0.5, min_lr = 1e-6)
     

checkpoint = ModelCheckpoint(
    filepath=f'best_model_do_05_2.h5',  # Filepath to save the model
    save_best_only=True,       # Save only the best model based on validation loss
    monitor='val_loss',        # Metric to monitor
    mode='min'                 # Save when 'val_loss' decreases
)


history = vgg16.fit(train_dataset,
                    steps_per_epoch = train_count//batch_size,
                    epochs = 10,
                    validation_data = test_dataset,
                    validation_steps = test_count//batch_size,
                    callbacks = [checkpoint])

pd.DataFrame(history.history).to_csv('History2.csv', index = False)




    


