'''
This script implements frame for retraining 
Allows to: 
  - retrain model
  - visualiize retraining results
  - compare models
'''
import tkinter as tk
import queue
from tkinter import ttk, messagebox, filedialog
import datetime
from tensorflow.keras.models import load_model # type: ignore # type: ignore
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # pyright: ignore[reportMissingModuleSource]
import time

from ..settings  import *
import sys
import os
import tensorflow as tf # pyright: ignore[reportMissingImports]
from ..utils import StreamRedirector, clear_widgets
from ..model_utils import vggfun, xcepfun, train_model, build_model
from ..image_utils import get_dataset

'''
Class which implements retraining frame
call with: 
  parent --> parent of the window calles (should be start frame, if it is not, automatically converted to start frame)
When called: 
  - dataset is autmatically created 
  - model is automatically retrained
  - operator is informed about retraining result  
'''
class retrain_frame:
    def __init__ (self, 
                  parent):
        if hasattr(parent, 'parent'):
            parent = parent.parent

        self.trained_model = None
        self.model_file = None
        self.parent = parent
        self.frame = tk.Frame(self.parent.root, bg=bg_colour)
        self.frame.grid(row=0, column=0)
        self.frame.tkraise()
        self.frame.pack_propagate(False)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        #self.parent.vis_frame.vis_frame.unbind_all("<MouseWheel>")

        self.train_dataset, self.test_dataset, self.Class_dict, self.train_count, self.test_count = get_dataset(im_path = im_path, 
                label_path = xml_path,
                batch_size = 32, 
                train_share = 0.8,
                verbose = True)

        self.log_text = tk.Text(self.frame, wrap="word", bg="black", fg="lime", font=("Courier", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, pady=(1, 2))
        self.progress.grid_remove()


        self.save_button = tk.Button(self.frame, text="Save Model", state="disabled", width=12)
        self.save_button.grid(row=1, column=0, sticky="w", padx=2, pady=1)
        self.save_button.config(command=self.save_model_to_disk)

        tk.Label(self.frame, text="Old Model:", font=("Arial", 10)).grid(row=2, column=0, sticky="w", padx=2, pady=(1, 0))
        self.old_model_var = tk.StringVar()
        self.old_model_menu = ttk.Combobox(self.frame, textvariable=self.old_model_var, width=50, state="disabled")
        self.old_model_menu.grid(row=3, column=0, padx=2, pady=1)
        self.old_model_name = self.old_model_var.get()

        tk.Label(self.frame, text="New Model:", font=("Arial", 10)).grid(row=4, column=0, sticky="w", padx=2, pady=(1, 0))
        self.new_model_var = tk.StringVar()
        self.new_model_menu = ttk.Combobox(self.frame, textvariable=self.new_model_var, width=50, state="disabled")
        self.new_model_menu.grid(row=5, column=0, padx=2, pady=1)
        self.new_model_name = self.new_model_var.get()

        self.compare_button = tk.Button(self.frame, text="Compare Models", command=self.compare_models, width=20, state="disabled")
        self.compare_button.grid(row=6, column=0, pady=10)
        
        tk.Label(self.frame, text="Old History File:", font=("Arial", 10)).grid(row=9, column=0, sticky="w", padx=2, pady=(1, 0))
        self.old_hist_var = tk.StringVar()
        self.old_hist_menu = ttk.Combobox(self.frame, textvariable=self.old_hist_var, width=50, state="disabled")
        self.old_hist_menu.grid(row=10, column=0, padx=2, pady=1)

        tk.Label(self.frame, text="New History File:", font=("Arial", 10)).grid(row=11, column=0, sticky="w", padx=2, pady=(1, 0))
        self.new_hist_var = tk.StringVar()
        self.new_hist_menu = ttk.Combobox(self.frame, textvariable=self.new_hist_var, width=50, state="disabled")
        self.new_hist_menu.grid(row=12, column=0, padx=2, pady=1)
        self.compare_hist_button = tk.Button(self.frame, text="Compare Training Histories", command=self.plot_selected_histories, width=25, state="disabled")
        self.compare_hist_button.grid(row=13, column=0, pady=1)

        tk.Button(self.frame, text="BACK", command=self.load_vis_frame, width=12).grid(row=8, column=0, pady=1)
        self.log_queue = queue.Queue()
        self.process_log_queue()

        threading.Thread(target=self.run_training, daemon=True).start()
        #self.run_training()


    '''
    Function which set model name variables from dropdown menus
    '''
    def get_model_names(self):
        self.old_model_name = self.old_model_var.get()
        self.new_model_name = self.new_model_var.get()
        

    '''
    Function which updates log screen in retrain frame
    '''
    def update_log(self, line):
        """Called from background thread - puts message in queue"""
        self.log_queue.put(line)

    '''
    Function which processes different calls to window which have to be run in main thread
    '''
    def process_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if callable(line):
                    line()    
                else:
                    self.log_text.insert("end", line + "\n")
                    self.log_text.see("end")
        except queue.Empty:
            pass
        
        # Schedule next check (every 100ms)
        if self.frame.winfo_exists():
            self.frame.after(100, self.process_log_queue)

    '''
    Function which clears own frame and loads visualization frame again
    '''
    def load_vis_frame(self):
        clear_widgets(self.frame)
        del self.frame
        self.frame = None
        self.parent.load_vis_frame()

    '''
    Function which compares models
    '''
    def compare_models(self):
        self.compare_button.config(state='disabled')
        self.progress.grid()
        self.progress.start(10)
        threading.Thread(target=self.run_comparison_task, daemon=True).start()

    '''
    Function which handles model comparison
      - loads models
      - compares models
      - informs operator about result
    '''
    def run_comparison_task(self):
        original_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.update_log)

        try:
            self.log_queue.put(lambda: self.get_model_names())
            time.sleep(1)
            self.log_queue.put(lambda: self.get_model_names())
            
            if not os.path.exists(self.old_model_name) or not os.path.exists(self.new_model_name):
                print("❌ Model paths not found.")
                messagebox.showerror("Error", "Please select valid model files.")
                return

            
            if 'vgg' in self.old_model_name.lower():
                print(f"🧪 Loading Old Model: {self.old_model_name}")
                old_model = vggfun(self.old_model_name)
                print(f"🧪 Loading New Model: {self.new_model_name}")
                new_model = vggfun(self.new_model_name)
            else:            
                print(f"🧪 Loading Old Model: {self.old_model_name}")
                old_model = xcepfun(self.old_model_name)
                print(f"🧪 Loading New Model: {self.new_model_name}")
                new_model = xcepfun(self.new_model_name)
    

            old_model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                              loss={"xmin": "mse", "ymin": "mse", "xmax": "mse", "ymax": "mse", "class": "categorical_crossentropy"},
                              metrics={"xmin": "mae", "ymin": "mae", "xmax": "mae", "ymax": "mae", "class": "acc"})
            print("📊 Evaluating Old Model...")
            old_results = old_model.evaluate(self.test_dataset, verbose=1)
            old_acc = old_results[0]  # Adjust if needed

            
            new_model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                              loss={"xmin": "mse", "ymin": "mse", "xmax": "mse", "ymax": "mse", "class": "categorical_crossentropy"},
                              metrics={"xmin": "mae", "ymin": "mae", "xmax": "mae", "ymax": "mae", "class": "acc"})
            print("📊 Evaluating New Model...")
            new_results = new_model.evaluate(self.test_dataset, verbose=1)
            new_acc = new_results[0]  # Adjust if needed

            print(f"📊 Old Model Accuracy: {old_acc:.4f}")
            print(f"📊 New Model Accuracy: {new_acc:.4f}")

            if new_acc > old_acc:
                msg = f"✅ New model outperforms the old model!\n\nNew Accuracy: {new_acc:.4f}\nOld Accuracy: {old_acc:.4f}"
            elif new_acc < old_acc:
                msg = f"❌ New model performs worse than the old model.\n\nNew Accuracy: {new_acc:.4f}\nOld Accuracy: {old_acc:.4f}"
            else:
                msg = f"⚖️ Both models perform equally.\n\nAccuracy: {new_acc:.4f}"

            self.log_queue.put(lambda:messagebox.showinfo("Model Comparison Result", msg))

        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            #self.log_queue.put(lambda:messagebox.showerror("Error", str(e)))

        finally:
            sys.stdout = original_stdout
            self.log_queue.put(lambda: self.compare_button.config(state='normal'))
            self.log_queue.put(lambda: self.progress.stop())
            self.log_queue.put(lambda: self.progress.grid_remove())


    '''
    Function which loads models from file names in dropdown menu
    '''
    def load_model_filenames(self):
        if not os.path.exists(vgg_dir) and not os.path.exists(xception_dir):
            self.log_queue.put("⚠️ 'retrained_models' folder not found.")
            return

        files = [f'{vgg_dir}/{f}' for f in os.listdir(vgg_dir) if f.endswith(".h5")]
        files.extend([f'{xception_dir}/{f}' for f in os.listdir(xception_dir) if f.endswith(".h5")])
        if not files:
            self.log_queue.put("⚠️ No .h5 models found.")
            return

        self.old_model_menu["values"] = files
        self.new_model_menu["values"] = files

        self.log_queue.put(lambda: self.old_model_menu.config(state="readonly"))
        self.log_queue.put(lambda: self.new_model_menu.config(state="readonly"))
        self.log_queue.put(lambda: self.compare_button.config(state="normal"))

        try:
            selected_vgg_filename = os.path.basename(self.parent.selected_vgg_model)
            self.log_queue.put(lambda: self.old_model_var.set(selected_vgg_filename if selected_vgg_filename in files else files[0]))
        except:
            self.log_queue.put(lambda: self.old_model_var.set(files[0]))

        self.log_queue.put(lambda: self.new_model_var.set(files[-1]))
        hist_files = [f'{vgg_dir}/{f}' for f in os.listdir(vgg_dir) if f.endswith(".csv")]
        hist_files.extend([f'{xception_dir}/{f}' for f in os.listdir(xception_dir) if f.endswith(".csv")])

        if hist_files:
            self.old_hist_menu["values"] = hist_files
            self.new_hist_menu["values"] = hist_files
            self.log_queue.put(lambda: self.old_hist_menu.config(state="readonly"))
            self.log_queue.put(lambda: self.new_hist_menu.config(state="readonly"))
            self.log_queue.put(lambda: self.compare_hist_button.config(state="normal"))

            self.log_queue.put(lambda: self.old_hist_var.set(hist_files[0]))
            self.log_queue.put(lambda: self.new_hist_var.set(hist_files[-1]))

    '''
    Function which runs retraining
      - runs trainig updates self.model and self.model_name
    '''
    def run_training(self, 
                     mode: str = 'Vgg'):
        
        original_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.update_log)

        try:   
            model = build_model(num_classes=10, mode=mode)
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            self.model_file = train_model(model=model, callback_update=None, 
                                        train_dataset=self.train_dataset, 
                                        test_dataset=self.test_dataset,
                                        train_count=self.train_count, 
                                        test_count=self.test_count, 
                                        save_path=self.model_file,
                                        mode = mode )
            
            self.log_queue.put("✅ Retraining Complete!")
            self.log_queue.put(f"📦 Model saved as: {self.model_file}")
            
            # Schedule GUI updates in main thread
            self.log_queue.put(lambda: self.save_button.config(state="normal"))
            self.log_queue.put(lambda: self.load_model_filenames())
            self.trained_model = load_model(self.model_file, compile = False)

        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            sys.stdout = original_stdout

    '''
    Function which allows to save new model to disk
      - opens dialog where user can select model name
    '''
    def save_model_to_disk(self):
        if self.trained_model is not None:
            if self.model_file == None:
                self.model_file = 'DefaultModel'
            path = filedialog.asksaveasfilename(defaultextension=".h5",
                                                filetypes=[("HDF5 Model", "*.h5")],
                                                initialfile=self.model_file)
            if path:
                self.trained_model.save(path)
                self.update_log(f"💾 Model saved to: {path}")

    '''
    Fucntion which plots old and new training histories
    '''
    def plot_selected_histories(self):
        try:
            old_hist_path = self.old_hist_var.get()# os.path.join(model_dir, old_hist_var.get())
            new_hist_path = self.new_hist_var.get() #os.path.join(model_dir, new_hist_var.get())
    
            old_hist = pd.read_csv(old_hist_path)
            new_hist = pd.read_csv(new_hist_path)
            metrics = [col for col in old_hist.columns if col in new_hist.columns]
    

            popup = tk.Toplevel(self.frame)
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
    
            # One subplot per metric (VGG vs Xception side by side)
            for metric in metrics:
                fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    
                # Old model plot (VGG)
                ax[0].plot(old_hist[metric])
                ax[0].set_title(f"Old - {metric}")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel(metric)
    
                # New model plot (Xception)
                ax[1].plot(new_hist[metric])
                ax[1].set_title(f"New - {metric}")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel(metric)
    
                fig.tight_layout()
    
                plot_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
                plot_canvas.draw()
                plot_canvas.get_tk_widget().pack(pady=10)
    
                plt.close(fig)  # ✅ Prevent memory leak
    
            tk.Button(scroll_frame, text="Close", command=popup.destroy).pack(pady=10)
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare histories:\n{e}")

