'''
This frame implements the visualization frame of the user interface 
  - Show predicted result and uncertainty
  - Show saliency map
  - Relabel image
'''

from ..settings import *
import tkinter as tk
from ..image_utils import (load_image_with_boxes, 
                           zoom_relabel,  
                           resize_image, 
                           class_dictvgg16
                           )
from ..plot_utils import draw_uncertainty_bar
from ..xai_utils import generate_saliency_image
from ..utils import get_uncertainty_color, MouseHandler

import os
from tensorflow import keras # pyright: ignore[reportMissingImports]
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk # pyright: ignore[reportMissingImports]
from .retrain_frame import retrain_frame, clear_widgets

'''
Frame for visualization:
  - Show predicted result and uncertainty
  - Show saliency map
  - Relabel image
initialize with: 
  parent --> start frame who called vis frame
  zoom_scale --> scale for showing image in relabel popup (use 1.0)
'''
class vis_frame: 
    def __init__ (self, 
                  parent,
                  zoom_scale: float = 1.0):
        self.retrain_frame = None
        self.vis_frame = tk.Frame(parent.root, width=500, height=600, bg=bg_colour)
        self.vis_frame.grid(row=0, column=0, sticky="nesw")
        self.vis_frame.tkraise()
        self.vis_frame.pack_propagate(False)
        self.vis_frame.grid_rowconfigure(0, weight=1)
        self.vis_frame.grid_columnconfigure(0, weight=1)
        self.vis_frame.grid_columnconfigure(1, weight=0)
        self.image_label = None
        self.zoom_scale = zoom_scale
        self.parent = parent
        self.warning_shown = False
        self.mouse_handler = None
        # Image Frame
        self.image_frame = tk.Frame(self.vis_frame, bg=bg_colour)
        self.image_frame.grid(row=0, column=0, sticky='nsew')

        self.image_label = tk.Label(self.image_frame, bg=bg_colour)
        self.image_label.pack()

        self.class_label_frame = tk.Frame(self.image_frame, bg=bg_colour)
        self.class_label_frame.pack(pady=(5, 0))

        self.pred_class_labelvg = tk.Label(self.class_label_frame, text="Predicted: N/A", fg="#FF0000", bg=bg_colour, font=("Arial", 12, "bold"))
        self.pred_class_labelxc = tk.Label(self.class_label_frame, text="Predicted: N/A", fg="#0000FF", bg=bg_colour, font=("Arial", 12, "bold"))
        self.gt_class_label = tk.Label(self.class_label_frame, text="Ground Truth: N/A", fg="#00FF00", bg=bg_colour, font=("Arial", 12, "bold"))

        self.pred_class_labelvg.pack()
        self.pred_class_labelxc.pack()
        self.gt_class_label.pack()

        # Control Frame
        control_frame = tk.Frame(self.vis_frame, bg=bg_colour)
        control_frame.grid(row=0, column=1, sticky='ns', padx=20, pady=20)
        tk.Button(control_frame, text="Relabel", command=self.relabel_image, width=12).pack(pady=5)
        tk.Button(control_frame, text="Saliency Map", command=self.open_saliency_window, width=12).pack(pady=5)
        tk.Button(control_frame, text="Use Predicted Labels of VGG", command=self.save_predicted_labels_vgg, width=18).pack(pady=5)
        tk.Button(control_frame, text="Use Predicted Labels of Xception", command=self.save_predicted_labels_xception, width=18).pack(pady=5)
        tk.Button(control_frame, text="BACK", command=self.load_main_frame, width=12).pack(pady=5)

        tk.Button(control_frame, text="Retraining", command=self.load_retrain_frame, width=12).pack(pady=5)

        tk.Label(control_frame, text="VGG16 Uncertainty", fg="white", bg=bg_colour, font=("Arial", 12, "bold")).pack(pady=(10, 0))

        self.vgg_bar = tk.Canvas(control_frame, width=BAR_WIDTH, height=BAR_HEIGHT, bg=bg_colour, highlightthickness=0)
        self.vgg_bar.pack(pady=5)
        tk.Label(control_frame, text="Xception Uncertainty", fg="white", bg=bg_colour, font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.xcep_bar = tk.Canvas(control_frame, width=BAR_WIDTH, height=BAR_HEIGHT, bg=bg_colour, highlightthickness=0)
        self.xcep_bar.pack(pady=10)

        self.vis_frame.bind_all("<MouseWheel>", lambda e: self.zoom_main(1.25) if e.delta > 0 else self.zoom_main(0.8))

        if self.parent.image_list:
            self.update_image()

    '''
    Function which closes vis_frame and opens start_frame again
    '''
    def load_main_frame(self):
        clear_widgets(self.vis_frame)
        del self.vis_frame
        self.vis_frame = None
        self.parent.load_frame()

    '''
    Function which closes vis_frame and loads frame for retraining mode
    '''
    def load_retrain_frame(self):
        clear_widgets(self.vis_frame)
        self.vis_frame.unbind_all("<MouseWheel>")
        del self.vis_frame
        self.vis_frame = None
        self.retrain_frame = retrain_frame(parent=self)
            
    '''
    Function which zooms in main vis_frame and updates image
    '''
    def zoom_main(self, 
                  factor: float):
        self.zoom_scale = max(0.2, min(self.zoom_scale * factor, 5.0))
        self.update_image()


    '''
    Functio which updates image
      - updates image
      - shows bounding box
      - shows uncertainty
    '''
    def update_image(self):
        if not self.parent.image_list:
            return
        img_tk, pred_classv, pred_classx, gt_classes = load_image_with_boxes(index = self.parent.current_index, 
                                                                             image_list= self.parent.image_list, 
                                                                             shared_state=self.parent.state,
                                                                             zoom = self.zoom_scale)
        if self.image_label and self.image_label.winfo_exists():
            self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        if self.pred_class_labelvg and self.pred_class_labelvg.winfo_exists():
            self.pred_class_labelvg.config(text=f"Predicted Vgg16: {pred_classv}")
        if self.pred_class_labelxc and self.pred_class_labelxc.winfo_exists():
            self.pred_class_labelxc.config(text=f"Predicted Xception: {pred_classx}")
        self.gt_class_label.config(text=f"Ground Truth: {', '.join(gt_classes) if gt_classes else 'N/A'}")

        if self.parent.state.vgg_uncertainty is not None and self.parent.state.xcep_uncertainty is not None:
            percent_vgg = self.parent.state.vgg_uncertainty * 100
            percent_xcep = self.parent.state.xcep_uncertainty * 100

            if (percent_vgg > UNCERTAINTY_THRESHOLD or percent_xcep > UNCERTAINTY_THRESHOLD) and not self.warning_shown:
                self.warning_shown = True
                messagebox.showwarning("High Uncertainty", "Uncertainty for this image is high.\nPlease consider relabeling it.")

            draw_uncertainty_bar(
                canvas = self.vgg_bar,
                bar_height=BAR_HEIGHT, 
                bar_width=BAR_WIDTH, 
                percent = percent_vgg, 
                color = get_uncertainty_color(percent_vgg)
                )
            draw_uncertainty_bar(
                self.xcep_bar, 
                bar_height=BAR_HEIGHT, 
                bar_width=BAR_WIDTH, 
                percent = percent_xcep, 
                color = get_uncertainty_color(percent_xcep)
                )

    '''
    Function which opens saliency map 
    '''
    def open_saliency_window(self):
        if self.parent.state.vgg_grad is None or self.parent.state.imageu is None or self.parent.state.xcep_grad is None:
            return

        popup = tk.Toplevel()
        popup.title("Saliency Map Viewer")
        popup.geometry("600x750")
        popup.configure(bg=bg_colour)

        sal_img_label = tk.Label(popup, bg=bg_colour)
        sal_img_label.pack()

        alpha1 = tk.DoubleVar(value=0.5)
        alpha2 = tk.DoubleVar(value=0.5)

        def update_saliency(*args):
            sal_img = generate_saliency_image(state=self.parent.state, 
                                              alpha1=alpha1.get(), 
                                              alpha2=alpha2.get())
            sal_img_label.config(image=sal_img)
            sal_img_label.image = sal_img

        tk.Label(popup, text="Alpha (Vgg16)", fg="white", bg=bg_colour).pack(pady=(10, 0))
        tk.Scale(popup, from_=0, to=1, resolution=0.05, orient="horizontal", variable=alpha1, command=update_saliency).pack()
        tk.Label(popup, text="Alpha (Xception)", fg="white", bg=bg_colour).pack(pady=(10, 0))
        tk.Scale(popup, from_=0, to=1, resolution=0.05, orient="horizontal", variable=alpha2, command=update_saliency).pack()
        update_saliency()

    '''
    Function which writes xml file for image using predictions of vgg
    '''
    def save_predicted_labels_vgg(self):
        base_name = os.path.splitext(os.path.basename(self.parent.image_list[self.parent.current_index]))[0]
        label_path = os.path.join(save_predictedlabloc, base_name + ".txt")
        os.makedirs(save_predictedlabloc, exist_ok=True)
        
        # Prevent overwriting
        #if os.path.exists(label_path):
        #    messagebox.showwarning("Exists", f"Label for {base_name} already exists. Skipping save.")
        #    return
        orig_img = keras.preprocessing.image.array_to_img(self.parent.state.imageu[0])
        with open(label_path, 'w') as f:
            for result, class_dict in [(self.parent.state.vgg_result, class_dictvgg16)]:
                if result:
                    x_min, y_min, x_max, y_max, class_index = result
                    cls_id = class_index[0]
                    xmin = x_min.item()
                    ymin = y_min.item()
                    xmax = x_max.item()
                    ymax = y_max.item()
                    f.write(f"{cls_id + 1} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")
        
                    ## Save image to existing class folder
                    class_name = class_dict[cls_id + 1]
                    class_folder = os.path.join(PARENT_FOLDER, class_name)
                    os.makedirs(class_folder, exist_ok=True)
                    save_image_path = os.path.join(class_folder, base_name + ".jpg")

                    if os.path.exists(save_image_path):
                        messagebox.showwarning("Exists", f"Image {base_name}.jpg already exists in {class_name}. Skipping save.")
                        return

                    orig_img.save(save_image_path)
        
        messagebox.showinfo("Saved", f"Predicted labels saved for {base_name}.")

    '''
    Function which writes xml file for image using predictions of vgg
    '''
    def save_predicted_labels_xception(self):
        base_name = os.path.splitext(os.path.basename(self.parent.image_list[self.parent.current_index]))[0]
        label_path = os.path.join(save_predictedlabloc, base_name + ".txt")
        os.makedirs(save_predictedlabloc, exist_ok=True)
        
        # Prevent overwriting
        #if os.path.exists(label_path):
        #    messagebox.showwarning("Exists", f"Label for {base_name} already exists. Skipping save.")
        #    return
        orig_img = keras.preprocessing.image.array_to_img(self.parent.state.imageu[0])
        with open(label_path, 'w') as f:
            for result, class_dict in [(self.parent.state.xcep_result, class_dictvgg16)]:
                if result:
                    x_min, y_min, x_max, y_max, class_index = result
                    cls_id = class_index[0]
                    xmin = x_min.item()
                    ymin = y_min.item()
                    xmax = x_max.item()
                    ymax = y_max.item()
                    f.write(f"{cls_id + 1} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")
        
                    ## Save image to existing class folder
                    class_name = class_dict[cls_id + 1]
                    class_folder = os.path.join(PARENT_FOLDER, class_name)
                    os.makedirs(class_folder, exist_ok=True)
                    save_image_path = os.path.join(class_folder, base_name + ".jpg")

                    if os.path.exists(save_image_path):
                        messagebox.showwarning("Exists", f"Image {base_name}.jpg already exists in {class_name}. Skipping save.")
                        return

                    orig_img.save(save_image_path)
        
        messagebox.showinfo("Saved", f"Predicted labels saved for {base_name}.")


    '''
    Function for relabeling image
      - opens additional frame (popup)
      - allows to drwa bounding box and select class
      - allows to store label (warns if it does not fit to label from model predictions)
    '''
    def relabel_image(self):
        # Temporarily disable zoom on vis_frame to avoid conflict with popup
        self.vis_frame.unbind_all("<MouseWheel>")
        if not self.parent.image_list:
            return

        orig_img = keras.preprocessing.image.array_to_img(self.parent.state.imageu[0])
        w, h = orig_img.size

        relabel_popup = tk.Toplevel()
        relabel_popup.title("Relabel Image")
        zoom_factor = 1.0

        img_display = resize_image(img=orig_img, 
                                   h = h, 
                                   w = w, 
                                   zoom_factor=zoom_factor)
        
        popup_width = img_display.width + 40
        popup_height = img_display.height + 200
        relabel_popup.geometry(f"{popup_width}x{popup_height}")
        relabel_popup.configure(bg=bg_colour)

        canvas_frame = tk.Frame(relabel_popup)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        vbar = tk.Scrollbar(canvas_frame, orient="vertical")
        vbar.pack(side="right", fill="y")
        hbar = tk.Scrollbar(canvas_frame, orient="horizontal")
        hbar.pack(side="bottom", fill="x")

        canvas = tk.Canvas(
            canvas_frame,
            width=min(img_display.width, popup_width - 60),
            height=min(img_display.height, popup_height - 150),
            yscrollcommand=vbar.set,
            xscrollcommand=hbar.set,
            cursor="cross",
            bg="black"
        )
        canvas.pack(side="left", fill="both", expand=True)

        vbar.config(command=canvas.yview)
        hbar.config(command=canvas.xview)

        tk.Label(relabel_popup, text="Left-click and drag to draw boxes", bg=bg_colour, fg="white").pack()
        tk.Label(relabel_popup, text="Select Class:", bg=bg_colour, fg="white").pack()

        class_names = list(class_dictvgg16.values())
        class_combo = ttk.Combobox(relabel_popup, values=class_names, state="readonly")
        class_combo.current(0)
        class_combo.pack(pady=5)

        boxes = []
        start_x = start_y = rect = None

        img_tk = ImageTk.PhotoImage(img_display)
        canvas.image = img_tk  # WICHTIG: Referenz halten!
        image_id = canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.config(scrollregion=(0, 0, img_display.width, img_display.height))
        canvas.config(scrollregion=(0, 0, img_display.width, img_display.height))
        
        def save_boxes(self):
            base_name = os.path.splitext(os.path.basename(self.parent.image_list[self.parent.current_index]))[0]
            label_path = os.path.join(save_predictedlabloc, base_name + ".txt")
            os.makedirs(save_predictedlabloc, exist_ok=True)
    
            if os.path.exists(label_path):
                messagebox.showwarning("Exists", f"Label for {base_name} already exists. Skipping save.")
                relabel_popup.destroy()
                return
        
            orig_img = keras.preprocessing.image.array_to_img(self.parent.state.imageu[0])
            w, h = orig_img.size
        
            # Get predicted class indices and names
            predicted_classes = set()
            pred_texts = []
        
            if self.parent.state.vgg_result:
                cls_idx_vgg = self.parent.state.vgg_result[-1][0]
                predicted_classes.add(cls_idx_vgg)
                pred_texts.append(f"VGG16: {class_dictvgg16.get(cls_idx_vgg + 1, 'N/A')}")
        
            if self.parent.state.xcep_result:
                cls_idx_xcep = self.parent.state.xcep_result[-1][0]
                predicted_classes.add(cls_idx_xcep)
                pred_texts.append(f"Xception: {class_dictvgg16.get(cls_idx_xcep + 1, 'N/A')}")
        
            # Only one box allowed
            if not boxes:
                messagebox.showwarning("No Box", "Please draw a box before saving.")
                return
        
            cls_id = boxes[0][4]
            selected_class_name = class_names[cls_id]
        
            # Warn if selected class not in predictions
            if cls_id not in predicted_classes:
                pred_display = "\n".join(pred_texts) if pred_texts else "No predictions available"
                proceed = messagebox.askyesno(
                    "Class Mismatch Warning",
                    f"Selected class: ✗ {selected_class_name}\n\n"
                    f"Model Predictions:\n{pred_display}\n\n"
                    f"The selected class does not match either model prediction.\n"
                    f"Do you still want to save the label?"
                )
                if not proceed:
                    return
        
            # Save label file and image
            with open(label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2, cls_id = box
                    x1_orig = x1 / zoom_factor
                    y1_orig = y1 / zoom_factor
                    x2_orig = x2 / zoom_factor
                    y2_orig = y2 / zoom_factor
                    xmin = x1_orig / w
                    ymin = y1_orig / h
                    xmax = x2_orig / w
                    ymax = y2_orig / h
                    f.write(f"{cls_id + 1} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")
        
                    class_name = class_names[cls_id]
                    class_folder = os.path.join(PARENT_FOLDER, class_name)
                    os.makedirs(class_folder, exist_ok=True)
                    save_image_path = os.path.join(class_folder, base_name + ".jpg")
        
                    if os.path.exists(save_image_path):
                        messagebox.showwarning("Exists", f"Image {base_name}.jpg already exists in {class_name}. Skipping save.")
                        relabel_popup.destroy()
                        return
        
                    orig_img.save(save_image_path)
        
            relabel_popup.destroy()
        

        self.mouse_handler = MouseHandler(canvas=canvas, 
                                          class_names=class_names, 
                                          boxes=boxes)
        
        mouse_up_fun = lambda event: self.mouse_handler.on_mouse_up(event=event, cls_name=class_combo.get())
        canvas.bind("<Button-1>", self.mouse_handler.on_mouse_down)
        canvas.bind("<B1-Motion>", self.mouse_handler.on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", mouse_up_fun)
        canvas.bind("<Button-3>", self.mouse_handler.undo_last_box)

        control_frame = tk.Frame(relabel_popup, bg=bg_colour)
        control_frame.pack(pady=10)
        tk.Button(control_frame, text="Zoom In", command=lambda: self.zoom_relabel(img_display=img_display, h = h, w = w, relabel_popup=relabel_popup, boxes=boxes, factor=1.25, canvas=canvas)).pack(side="left", padx=5)
        tk.Button(control_frame, text="Zoom Out", command=lambda: self.zoom_relabel(img_display=img_display, h = h, w = w, relabel_popup=relabel_popup, boxes=boxes, factor = 0.8, canvas=canvas)).pack(side="left", padx=5)
        tk.Button(control_frame, text="Save", command=lambda: save_boxes(self)).pack(side="left", padx=5)


        mouswheel_fun = lambda event: self.on_mousewheel(img_display=img_display, 
                                                    h = h, 
                                                    w = w, 
                                                    canvas = canvas, 
                                                    relabel_popup=relabel_popup, 
                                                    boxes=boxes, 
                                                    event = event)
        self.vis_frame.bind_all("<MouseWheel>", mouswheel_fun)

        def on_close():
            canvas.unbind("<MouseWheel>")
            self.vis_frame.bind_all("<MouseWheel>", lambda e: self.zoom_main(1.25) if e.delta > 0 else self.zoom_main(0.8))
            relabel_popup.destroy()

        relabel_popup.protocol("WM_DELETE_WINDOW", on_close)

    '''
    Function for zooming in relabel window 
    Only calls zoom function from ..image_utils and updates self.zoom_scale
    '''
    def zoom_relabel(self, 
                     img_display, 
                    relabel_popup, 
                    h: float, 
                    w: float,
                    canvas: tk.Canvas,
                    boxes: list, 
                    factor: float): 
        self.zoom_scale = zoom_relabel(img_display, 
                 relabel_popup, 
                 h, 
                 w,
                 canvas,
                 boxes, 
                 factor, 
                 self.zoom_scale)

    '''
    Function which allows to zoom in relabel frame using mouse wheel
    '''
    def on_mousewheel(self, 
                      img_display, 
                    relabel_popup, 
                    h: float, 
                    w: float,
                    canvas: tk.Canvas,
                    boxes: list, 
                    event: tk.Event):
        if event.delta > 0:
            self.zoom_relabel(img_display=img_display, 
                                            relabel_popup=relabel_popup,
                                            h =  h, 
                                            w = w,
                                            canvas=canvas,  
                                            boxes=boxes,
                                            factor=1.1)
        else:
            self.zoom_relabel(img_display=img_display, 
                                relabel_popup=relabel_popup,
                                h =  h, 
                                w = w,
                                canvas=canvas,  
                                boxes=boxes, 
                                factor=0.9)

    

