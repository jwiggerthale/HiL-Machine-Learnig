'''
Function which implements different utilities
'''
from lxml import etree # type: ignore
import os
import tkinter as tk

'''
Function to get color for uncertainty bar (see plot_utils.py)
Call with: 
  value --> uncertainty value for model
'''
def get_uncertainty_color(value):
        r = int(255 * (value / 100.0))
        g = int(255 * (1 - value / 100.0))
        return f'#{r:02x}{g:02x}00'

'''
Functon which gest class for image name (by search in folder)
call with: 
  parent_folder --> folder with all images 
  base_name --> base name of image
returns: 
  found: list --> list of all folders found 
'''
def ground_class(parent_folder, base_name):
    found = []
    for sub in os.listdir(parent_folder):
        subp = os.path.join(parent_folder, sub)
        if os.path.isdir(subp) and os.path.exists(os.path.join(subp, base_name + ".jpg")):
            found.append(sub)
    return found or []

'''
Function which converts xml from path to label
call with: 
  xml_path: str --> path to xml file
'''
def to_labels(xml_path: str):
    xml = open(xml_path).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath("//size/width/text()")[0])
    height = int(sel.xpath("//size/height/text()")[0])
    xmin = int(sel.xpath("//bndbox/xmin/text()")[0])
    ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
    xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
    ymax = int(sel.xpath("//bndbox/ymax/text()")[0])
    return [xmin/width, ymin/height, xmax/width, ymax/height]

'''
Function which clears all widgets from frame
call with: 
  frame --> frame to be cleared
'''
def clear_widgets(frame):
    for widget in frame.winfo_children():
        widget.destroy()


'''
Class which stores results to be shared among frames
'''
class SharedState:
    def __init__(self):
        self.imageu = None
        self.vgg_result = None
        self.xcep_result = None
        self.vgg_grad = None
        self.xcep_grad = None
        self.vgg_uncertainty = None
        self.xcep_uncertainty = None


'''
Class which handles mouse events in relabel popup (vis frame)
'''
class MouseHandler:
    def __init__(self, canvas, class_names, boxes):
        self.canvas = canvas
        self.class_names = class_names
        self.boxes = boxes
        
        # Zustandsvariablen
        self.start_x = None
        self.start_y = None
        self.rect = None
    
    def redraw_boxes(self):
            self.canvas.delete("box")
            for box in self.boxes:
                x1, y1, x2, y2, cls_id = box
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="cyan", width=2, tags="box")
                class_text = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                self.canvas.create_text(x1 + 5, y1 + 10, text=class_text, fill="cyan", anchor="nw", font=("Arial", 12), tags="box")

    def on_mouse_down(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline="yellow", width=2, tags="current_box"
        )
    
    def on_mouse_drag(self, event):
        if self.rect is not None:
            cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
    
    def on_mouse_up(self, event, cls_name):
        if self.start_x is None or self.start_y is None:
            return
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        cls_id = self.class_names.index(cls_name) if cls_name in self.class_names else 0
        box = (min(self.start_x, end_x), min(self.start_y, end_y), 
               max(self.start_x, end_x), max(self.start_y, end_y), cls_id)
        self.boxes.append(box)
        self.canvas.delete("current_box")
        self.redraw_boxes()
        
        # Zustand zurücksetzen
        self.start_x = None
        self.start_y = None
        self.rect = None

    def undo_last_box(self, event):
        if self.boxes:
            self.boxes.pop()
            self.redraw_boxes()

'''
Class which streams messages 
Initialze with: 
  callback --> callback for stream
'''
class StreamRedirector:
    def __init__(self, callback):
        self.callback = callback

    def write(self, message):
        if message.strip():
            self.callback(message.strip())

    def flush(self):
        pass
