
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
from lxml import etree
from sklearn.preprocessing import LabelBinarizer



def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,3)
    image = tf.image.resize(image,[224,224])
    image = tf.cast(image,tf.float32)
    image = image / 255
    return image


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



def get_dataset(im_path: str = "/data/HiL_XAI/SteelLocation/images/*/*.jpg", 
                label_path: str = "/data/HiL_XAI/SteelLocation/label/*.xml", 
                batch_size: int = 32, 
                train_share: float = 0.8):
    image_path = glob.glob(im_path)
    xmls_path = glob.glob(label_path)

    print(f'Found {len(image_path)} images and {len(xmls_path)} paths')



    xmls_path.sort(key = lambda x:x.split("/")[-1].split(".xml")[0])
    image_path.sort(key = lambda x:x.split("/")[-1].split(".jpg")[0])

    xmls_train = [path.split("/")[-1].split(".")[0] for path in xmls_path]

    imgs_train = [img for img in image_path if (img.split("/")[-1].split)(".jpg")[0] in xmls_train]

    labels = [label.split("/")[-2] for label in imgs_train]
    labels = pd.DataFrame(labels, columns = ["Defect Type"])

    # Obtain training labels without duplication
    Class = labels["Defect Type"].unique()
    # Store data values in key:value pairs with Python dictionaries
    Class_dict = dict(zip(Class, range(1,len(Class) + 1)))
    labels["Class"] = labels["Defect Type"].apply(lambda x: Class_dict[x])

    lb = LabelBinarizer()
    # Fit label binarizer
    lb.fit(list(Class_dict.values()))
    # Convert multi-class labels to binary labels (belong or does not belong to the class)
    transformed_labels = lb.transform(labels["Class"])
    y_bin_labels = []

    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append("Class" + str(i))
        labels["Class" + str(i + 1)] = transformed_labels[:, i]


    labels.drop("Class", axis = 1, inplace = True)
    labels.drop("Defect Type", axis = 1, inplace = True)

    coors = [to_labels(path) for path in xmls_path]


    xmin, ymin, xmax, ymax = list(zip(*coors))

    # Convert to Numpy array
    xmin = np.array(xmin)
    ymin = np.array(ymin)
    xmax = np.array(xmax)
    ymax = np.array(ymax)
    label = np.array(labels.values, dtype = np.float16)


    labels_dataset = tf.data.Dataset.from_tensor_slices((xmin, ymin, xmax, ymax, label))


    dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
    dataset = dataset.map(load_image)

    dataset_label = tf.data.Dataset.zip((dataset, labels_dataset))

    dataset_label = dataset_label.repeat().shuffle(500).batch(batch_size)

    # Creates a Dataset that prefetches elements from this dataset
    # Most dataset input pipelines should end with a call to prefetch
    # This allows later elements to be prepared while the current element is being processed
    # This often improves latency and throughput, at the cost of using additional memory to store prefetched elements

    dataset_label = dataset_label.prefetch(tf.data.experimental.AUTOTUNE)


    train_count = int(len(imgs_train) * train_share)
    test_count = int(len(imgs_train) * (1 - train_share))

    train_dataset = dataset_label.skip(test_count)
    test_dataset = dataset_label.take(test_count)
    return train_dataset, test_dataset, Class_dict, train_count, test_count
