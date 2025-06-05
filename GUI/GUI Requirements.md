# Requirements for GUI

## Roles



Operator	 
		|   Tasks                   |    	  Needs

		|  Inspect objects	      |       Clear visualization

      	        |  Inspect images	      |      Qucik labeling and drawing of bounding boxes
	       
      	        |  Label images	              |      Intuitive hotkeys
	       
      	        |  Draw bounding boxes	      |
	       
---------------------------------------------------------------------------------------------------

Data Scientist	
		|   Tasks                   |    	  Needs

		|  Supervise model	        |         Aceess to annotation versions of images

	        |  Retrain model	          |         Logs
	       
	        |  Analyze performance	    |         Operators feedback
	       
---------------------------------------------------------------------------------------------------
Admin  	        
		|   Tasks                   |    	  Needs

		|  Permission management	  |          Audit trail

                |  Maintenance	Backup      |
---------------------------------------------------------------------------------------------------
	

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
7) Hotkeys
  - Accept
  - Correct
  - Retrain
  - One per class to change class
8) Dataset extension
  - Create json/ coco entry for new data point automatically
  - Task queue for cross checking	Order images by priority (uncertainty)
9) Retraining
  - Automatic start after x new images
  - Start by operator
  - Visualization of performance metrics
  - Visualization of progress"

