import os
from tensorflow import keras # type: ignore
import tensorflow as tf # type: ignore


import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, ImageDraw, ImageFont # pyright: ignore[reportMissingImports]
import glob
from sklearn.preprocessing import LabelBinarizer # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]


from .utils import SharedState, to_labels, ground_class
from .model_utils import class_dictvgg16, class_names
from .settings import *


 """
 Function for start frame 
 opens a dialog to select a single image and set the global image list
 """
def open_single_image():
    image_path = filedialog.askopenfilename(
        title="Select a single image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if image_path:
        selected_folder = os.path.dirname(image_path)
        image_list = [image_path]
        current_index = 0
        print(f"Selected image: {image_path}")
    else:
        # Clear previous selection
        selected_folder = None
        image_list = []
        current_index = 0
        print("No image selected.")
    return image_list


'''
 Function for start frame 
 loads image to tf format
 call with: 
   path: str --> path to image
  returns: 
    img: tf.image --> image selected
'''
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,3)
    img = tf.image.resize(img,[224,224])
    img = tf.cast(img,tf.float32) / 255.0
    return img



'''
Function for vis frame 
draws image with bounding box(es)
call with: 
  index: int --> index of image in list
  image_list: list --> list of image paths
  shared_stat: SharedState --> shared state of vis frame
  zoom: float = 1.0 --> zoom factor to use
'''
def load_image_with_boxes(index: int,  
                          image_list: list, 
                          shared_state: SharedState,
                          zoom: float =1.0):

    img_path = image_list[index]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    image_pil = keras.preprocessing.image.array_to_img(shared_state.imageu[0])
    base_size = 224
    zoomed_size = int(base_size * zoom)
    image_resized = image_pil.resize((zoomed_size, zoomed_size))

    draw = ImageDraw.Draw(image_resized)
    font_size = max(12, int(20 * zoom))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    pred_class_namev = pred_class_namex = None

    if shared_state.vgg_result:
        x_min, y_min, x_max, y_max, class_index = shared_state.vgg_result
        
        if x_min > x_max:
            temp = x_max
            x_max = x_min
            x_min = temp
        if y_min > y_max:
            temp = y_max
            y_max = y_min
            y_min = temp
        coords = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        coords = [c * zoomed_size for c in coords]
        pred_class_namev = class_dictvgg16[class_index[0] + 1]
        draw.rectangle(coords, outline="#FF0000", width=2)

    if shared_state.xcep_result:
        x_min, y_min, x_max, y_max, class_index = shared_state.xcep_result
        if x_min > x_max:
            temp = x_max
            x_max = x_min
            x_min = temp
        if y_min > y_max:
            temp = y_max
            y_max = y_min
            y_min = temp
        coords = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        coords = [c * zoomed_size for c in coords]
        pred_class_namex = class_dictvgg16[class_index[0] + 1]
        draw.rectangle(coords, outline="#0000FF", width=2)

    gt_label_path = os.path.join(ground_truth_folder, base_name + ".xml")
    gt_class_names = []
    if os.path.exists(gt_label_path):
        global grimg
        out1, out2, out3, out4 = to_labels(gt_label_path)
        grimg=[out1, out2, out3, out4]
        coords = [out1 * zoomed_size, out2 * zoomed_size, out3 * zoomed_size, out4 * zoomed_size]
        gt_class_names = ground_class(PARENT_FOLDER, base_name)
        draw.rectangle(coords, outline="#00FF00", width=2)

    return ImageTk.PhotoImage(image_resized), pred_class_namev, pred_class_namex, gt_class_names


'''
Function for vis frame 
resizes image 
call with: 
  img --> image to resize
  w: int --> current width
  h: int --> current height
  zoom_factot --> factor to use for resizing
returns: 
  img --> resized image
'''
def resize_image(img, 
                 w: int, 
                 h: int, 
                 zoom_factor: float):
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    return img.resize((new_w, new_h))

'''
Function for vis frame (relabel popup)
redraws all boxes in image
call with: 
  canvas --> cavas for relabeling popup
  boxes: list --> list of bounding boxes
'''
def redraw_boxes(canvas: tk.Canvas, 
                 boxes: list):
    canvas.delete("box")
    for box in boxes:
        x1, y1, x2, y2, cls_id = box
        canvas.create_rectangle(x1, y1, x2, y2, outline="cyan", width=2, tags="box")
        class_text = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        canvas.create_text(x1 + 5, y1 + 10, text=class_text, fill="cyan", anchor="nw", font=("Arial", 12), tags="box")


'''
Function for vis frame (relabel popup)
redraws all boxes in image
call with:
  img_display --> image diapls
  canvas --> cavas for relabeling popup
  boxes: list --> list of bounding boxes
  image_id: int --> id of image in canvas
'''
def update_canvas_image(img_display, 
                        canvas: tk.Canvas, 
                        boxes: list,
                        image_id: int = 0):
    img_tk = ImageTk.PhotoImage(img_display)
    canvas.itemconfig(image_id, image=img_tk)
    canvas.config(scrollregion=(0, 0, img_display.width, img_display.height))
    redraw_boxes(canvas=canvas, 
                 boxes=boxes)


'''
Function for vis frame (relabel popup)
resizes image in relabel popup
call with: 
   img_display --> current image display
   relabel_popup --> popup window
   h: float --> height of image
   w: float --> width of image
   canvas: tk.Canvas --> canvas with image
   boxes: list --> list of bounding boces
   factor: float --> factor for rescaling
   zoom_factor: float --> current zoom scale of window
returns: 
  new_zoom --> new zoom factor for frame
'''
def zoom_relabel(img_display, 
                 relabel_popup, 
                 h: float, 
                 w: float,
                 canvas: tk.Canvas,
                 boxes: list, 
                 factor: float, 
                 zoom_factor: float):
    new_zoom = zoom_factor * factor
    if new_zoom < MIN_ZOOM or new_zoom > MAX_ZOOM:
        return zoom_factor
    
    # Bild neu skalieren
    new_img_display = resize_image(img=img_display, h=h, w=w, zoom_factor=new_zoom)
    
    # Neues PhotoImage erstellen
    new_img_tk = ImageTk.PhotoImage(new_img_display)
    
    # Canvas-Bild aktualisieren
    canvas.itemconfig(canvas.find_all()[0], image=new_img_tk)  # Erstes Item ist das Bild
    canvas.image = new_img_tk  # Referenz halten!
    
    # Scrollregion anpassen
    canvas.config(scrollregion=(0, 0, new_img_display.width, new_img_display.height))
    
    
    scale = new_zoom / zoom_factor
    zoom_factor = new_zoom
    for i, (x1, y1, x2, y2, cls_id) in enumerate(boxes):
        boxes[i] = (x1 * scale, y1 * scale, x2 * scale, y2 * scale, cls_id)
    update_canvas_image(img_display=img_display, canvas=canvas, boxes=boxes)
    relabel_popup.geometry(f"{int(img_display.width) + 40}x{int(img_display.height) + 150}")
    return new_zoom



'''
Function to get dataset from path
call with: 
  im_path: str 
  label_path: str
  batch_size: int = 32
  train_share: float = 0.8,
  verbose: bool = True
returns: 
  train_dataset --> tf dataset with train samples
  test_dataset --> tf dataset with test samples
  Class_dict: dict --> dictionary with mapping for classes
  train_count --> number of train samples
  test_count --> number of test samples
Note: 
  there has to be one folder for each class in the image folder
'''
def get_dataset(im_path: str = r"C:\Users\sagar\images\images\*\*.jpg", 
                label_path: str = r"C:\Users\sagar\label\*.xml",
                batch_size: int = 32, 
                train_share: float = 0.8,
                verbose: bool = True):
    image_path = glob.glob(im_path)
    xmls_path = glob.glob(label_path)
    if verbose:   
      print(f'Found {len(image_path)} images and {len(xmls_path)} paths')



    xmls_path.sort(key = lambda x:x.split("/")[-1].split(".xml")[0])
    image_path.sort(key = lambda x:x.split("/")[-1].split(".jpg")[0])

    # xmls_train = [path.split("/")[-1].split(".")[0] for path in xmls_path]
    xmls_train = {os.path.splitext(os.path.basename(path))[0]: path for path in xmls_path}

    imgs_train = []
    xml_files = []
    for img in image_path:
        base_name = os.path.splitext(os.path.basename(img))[0]
        if base_name in xmls_train:
            imgs_train.append(img)
            xml_files.append(xmls_train[base_name])

    if verbose:
       print(f'Found {len(imgs_train)} images_train and {len(xmls_train)} paths_train')
    labels = [os.path.basename(os.path.dirname(label)) for label in imgs_train]
    # labels = [label.split("/")[-2] for label in imgs_train]
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

    coors = [to_labels(path) for path in xml_files]


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
    dataset_label = dataset_label.prefetch(tf.data.experimental.AUTOTUNE)


    train_count = int(len(imgs_train) * train_share)
    test_count = int(len(imgs_train) * (1 - train_share))

    train_dataset = dataset_label.skip(test_count)
    test_dataset = dataset_label.take(test_count)
    return train_dataset, test_dataset, Class_dict, train_count, test_count

