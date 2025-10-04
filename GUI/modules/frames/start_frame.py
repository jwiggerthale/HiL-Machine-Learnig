'''
This scrip implements the start frame of the user interface
'''
from ..settings import *
from ..utils import SharedState, clear_widgets
from ..image_utils import open_single_image, load_image
from ..model_utils import vggfun, xcepfun, predict_with_both_models
from .vis_frame import vis_frame
from ..plot_utils import plot_accuracy_popup

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # pyright: ignore[reportMissingModuleSource]
import os


'''
class which implements start frame of user interface 
From here you can: 
  1) Open image and make prediction
  2) Compare models
  3) Visualize training histories
Initialize with: 
  image_list: list --> usually empty list
  root --> tkinter root object
'''
class start_frame:
    def __init__ (self, 
                 image_list: list, 
                 root: tk.Tk):
        
        self.root = root
        self.state = SharedState()
        self.current_index = 0
        self.image_list = image_list
        self.main_frame = None
        self.vis_frame = None #vis_frame(self)
        self.selected_vgg_model = None
        self.selected_xcep_model = None
        


        self.load_frame()

    '''
    Function which loads visualization frame when initialized or when subsequent window is closed
    '''
    def load_frame(self):
        self.main_frame = tk.Frame(self.root, width=500, height=600, bg=bg_colour)
        self.main_frame.grid(row=0, column=0, sticky="nesw")
        self.main_frame.tkraise()
        self.main_frame.pack_propagate(False)

        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=2)
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Title
        tk.Label(
            self.main_frame, text="Upload the data",
            bg=bg_colour, fg="white",
            font=("Arial", 44, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(40, 20))

        # Subtitle
        tk.Label(
            self.main_frame, text="Select Trained Models and Image",
            bg=bg_colour, fg="white",
            font=("Arial", 24)
        ).grid(row=1, column=0, columnspan=2, pady=(10, 30))

        # Model selection section
        self.model_frame = tk.Frame(self.main_frame, bg=bg_colour)
        self.model_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Label(self.model_frame, text="Select VGG16 model (.h5):", bg=bg_colour, fg="white", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.vgg_model_combo = ttk.Combobox(self.model_frame, width=40, state="readonly")
        self.vgg_model_combo.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.model_frame, text="Select Xception model (.h5):", bg=bg_colour, fg="white", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.xcep_model_combo = ttk.Combobox(self.model_frame, width=40, state="readonly")
        self.xcep_model_combo.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.model_frame, text="Select Old Model History (.csv):", bg=bg_colour, fg="white", font=("Arial", 14)).grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.vgg_hist_combo = ttk.Combobox(self.model_frame, width=40, state="readonly")
        self.vgg_hist_combo.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self.model_frame, text="Select New Model History (.csv):", bg=bg_colour, fg="white", font=("Arial", 14)).grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.xcep_hist_combo = ttk.Combobox(self.model_frame, width=40, state="readonly")
        self.xcep_hist_combo.grid(row=3, column=1, padx=10, pady=10)
        

        self.compare_hist_btn = tk.Button(
                self.main_frame,
                text="Compare Training Histories",
                font=("Arial", 12, "bold"),
                bg="#1f3f49", fg="white",
                command=self.compare_histories
            )
        self.compare_hist_btn.grid(row=5, column=0, columnspan=2, pady=10)
            
        self.plot_button = tk.Button(
                self.main_frame,
                text="Model ImproveMent (Class Accuracy) (All Retrained Models)",
                font=("Arial", 12, "bold"),
                bg="#1f3f49", fg="white",
                command=plot_accuracy_popup
            )
        self.plot_button.grid(row=6, column=0, columnspan=2, pady=(10, 20))

        tk.Button(
                self.main_frame,
                text="Open Image & Predict (VGG16 + Xception)",
                font=("Arial", 16, "bold"),
                bg="#1f3f49", fg="white",
                cursor="hand2", activebackground="#4ca6a8", activeforeground="white",
                command=self.start_prediction,
                padx=20, pady=10
            ).grid(row=3, column=0, columnspan=2, pady=40)


        # Populate models
        vgg_files = sorted([f for f in os.listdir(vgg_dir) if  f.endswith(".h5")])
        xcep_files = sorted([f for f in os.listdir(xception_dir) if f.endswith(".h5")])
        self.vgg_model_combo["values"] = vgg_files
        self.xcep_model_combo["values"] = xcep_files

        if vgg_files:
            self.vgg_model_combo.current(len(vgg_files) - 1)
        if xcep_files:
            self.xcep_model_combo.current(len(xcep_files) - 1)
            
        # Populate history dropdowns from retrained/
        hist_files = sorted([f for f in os.listdir(vgg_dir) if f.endswith(".csv")])
        hist_files = [f'{vgg_dir}/{f}' for f in hist_files]
        hist_files.extend(sorted([f'{xception_dir}/{f}' for f in os.listdir(xception_dir) if f.endswith(".csv")]))
        self.vgg_hist_combo["values"] = hist_files
        self.xcep_hist_combo["values"] = hist_files
        if hist_files:
            self.vgg_hist_combo.current(0)
            self.xcep_hist_combo.current(len(hist_files) - 1)

    '''
    Function which predicts on given image 
    Gets selected models (VGG and Xception) from dropdown menus 
    Loads models
    Makes prediction using both models
    Adds prediction to frame's shared state
    '''
    def start_prediction(self):
        vgg_model = self.vgg_model_combo.get()
        xcep_model = self.xcep_model_combo.get()

        if not vgg_model or not xcep_model:
            messagebox.showwarning("Model Missing", "Please select both VGG and Xception models.")
            return

        self.selected_vgg_model = os.path.join(vgg_dir, vgg_model)
        self.selected_xcep_model = os.path.join(xception_dir, xcep_model)

        vgg16 =vggfun(self.selected_vgg_model)
        xception = xcepfun(self.selected_xcep_model)
        self.image_list = open_single_image()
        if not self.image_list:
            print("Prediction aborted. No image selected.")
            return
        img_tensor = load_image(self.image_list[0])
        print('Predicting')
        self.state = predict_with_both_models(vgg_model=vgg16, 
                                           xception_model=xception, 
                                           img_tensor=img_tensor, 
                                           shared_state=self.state, 
                                           )
        print(self.state)
        print('Predicted')
        self.load_vis_frame()

    '''
    Function which loads vis frame (subsequent frame)
    Destroys own start frame 
    initializes vis frame
    '''
    def load_vis_frame(self):
        if self.main_frame is not None:
            clear_widgets(self.main_frame)
            self.main_frame = None
        self.vis_frame = vis_frame(parent = self)

    '''
    Function which compares training histories for selected vgg and xception model
    Stats are visualized in popup window
    '''
    def compare_histories(self):
        vgg_hist_path = self.vgg_hist_combo.get() #os.path.join(history_dir, vgg_hist_combo.get())
        xcep_hist_path = self.xcep_hist_combo.get() #os.path.join(history_dir, xcep_hist_combo.get())
    
        try:
            vgg_hist = pd.read_csv(vgg_hist_path)
            xcep_hist = pd.read_csv(xcep_hist_path)
            metrics = [col for col in vgg_hist.columns if col in xcep_hist.columns]
    
            # Create scrollable popup window
            popup = tk.Toplevel(self.main_frame)
            popup.title("Training History Comparison")
            popup.geometry("1200x800")
    
            canvas = tk.Canvas(popup)
            scrollbar = tk.Scrollbar(popup, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas)
    
            scroll_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
    
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
    
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
    
        
            for metric in metrics:
                fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    
                # VGG plot
                ax[0].plot(vgg_hist[metric])
                ax[0].set_title(f"VGG16 - {metric}")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel(metric)
    
                # Xception plot
                ax[1].plot(xcep_hist[metric])
                ax[1].set_title(f"Xception - {metric}")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel(metric)
    
                fig.tight_layout()
    
                plot_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
                plot_canvas.draw()
                plot_canvas.get_tk_widget().pack(pady=10)
                plt.close(fig) 
    
            # Close button
            tk.Button(scroll_frame, text="Close", command=popup.destroy).pack(pady=10)
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare histories:\n{e}")

