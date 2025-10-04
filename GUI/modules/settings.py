'''
This script holds the main variables to be set before starting the GUI workflow
'''
import threading 

im_path = "C:/Users/jwiggerthale/Desktop/Promotion/03_Human in the Loop XAI/GUI/images/images/*/*.jpg"  # insert your image path
xml_path = "C:/Users/jwiggerthale/Desktop/Promotion/03_Human in the Loop XAI/BaselineModels/SteelLocation/label/*.xml"   #insert your label paths
PARENT_FOLDER = "./images/images"  # give folder name which stores image
ground_truth_folder = "./images/label"  # give name of folder which stores ground truth labels
vgg_dir = "./retrained_models/vgg_models"
xception_dir = "./retrained_models/xception_models" 
save_predictedlabloc="newlabel"    #it stores new labels so name as perference
bg_colour = "#3d6466"
accuracy_column = "class_accuracy"
history_filename = "TrainingHistory.csv"
last_n_epochs = 5    # accuracy to determine which model is better is done by averg of last 5 epochs
UNCERTAINTY_THRESHOLD = 0.1 # threshold for uncertainty 
BAR_WIDTH = 200
BAR_HEIGHT = 20
MIN_ZOOM, MAX_ZOOM = 0.2, 5.0
ZOOM_IN_FACTOR = 1.25

ZOOM_OUT_FACTOR = 0.8
batch_size = 32    
train_split = 0.8
warning_shown = False
stop_training_event = threading.Event()
training_thread = None
