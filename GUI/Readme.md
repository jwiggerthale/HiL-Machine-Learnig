This folder contains the code implementing GUI and labeling tool allowing the operator to efficiently: 
1) Label and relabel images
2) Compare model predictions with his labels
4) Recognize how uncertain the model is
5) Get explanations for model predictions
6) Enlarge the dataset
7) Retrain the model

The GUI workflow can be started running main.py

# The GUI

## Roles



### Operator	 
**Tasks**
-   Inspect objects
-   Inspect images
-   Label images
-   Draw bounding boxes	

**Needs** 
- Clear visualization
- Qucik labeling and drawing of bounding boxes
- Intuitive hotkeys

	       

### Data Scientist	
**Tasks**
- Supervise model
- Retrain model
- Analyze performance

**Needs** 
- Aceess to annotation versions of images
- Logs
- Operators' feedback
      
	       
### Admin  	        
**Tasks**
- Permission management
- Maintenance	Backup 

**Needs** 
- Audit trail

	

## Workflows

### Phase: Coexistence 	

### Workflow:

1) Inspection of object
2) Labeling and classification
3) Visualization of: 
    - Image
    - Bounding box
    - Model uncertainty
    - Explanation
4) Accept or correct model output
    - Draw new bounding box
    - Change label (Dropdown menu)
    - Add image to dataset

### Phase: Retraining of model	

### Workflow:  
 
  1) Trigger retraining
  2) Visualization of progress
  3) Visualization of performance metrics (before and after)

 
### Phase: Cross checking

### Workflow: 

  1) Selection of image to be validated (random or high uncertainty)
	2) Information of operator
	3) Accept or correct model output
      - Draw new bounding box
      - Change label (Dropdown menu)
      - Add image to dataset"


## Requirements

1) Image visualization	
  - Show original image 
  - Zoom function
2) Bounding box
  - Visualize predicted and drwan bounding box
  - Color depending on class
3) Class visualization
  - Predicted class
  - Dropown menu to change class
4) Uncertainty visualization
  - Uncertainty with color scale + bar
5) Explanation
  - Grad-CAM Heatmap 
  - Slider to control transparency
6) Annotation tool
  - Draw bounding box
  - Copy bounding box
  - Resize bounding box
  - Move bounding box
7) Dataset extension
  - Create xml entry for new data point automatically
  - Task queue for cross checking	Order images by priority (uncertainty)
8) Retraining
  - Automatic start after x new images
  - Start by operator
  - Visualization of performance metrics
  - Visualization of progress"

## Impressions of the GUI

### Start Frame
![image]([(https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/GUI%20Start.jpg)])

![image](https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/Process%20Overview.png)


### Visualization Frame
Initial visualization: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/Prediction%20Good.jpg])

Saliency overlay: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/Saliency1.jpg])

Relabeling: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/Relabel2.jpg])

Save image: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/ImageSaved.jpg])

### Retrain Frame

Main retrain frame:
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/Retraining.jpg])

Model copmarison: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/Model%20comparison.jpg])

History comparison: 
![image]([https://github.com/jwiggerthale/HiL-Machine-Learnig/blob/main/GUI/images/History%20Comparison.jpg])








