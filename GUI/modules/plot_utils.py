'''
This script implements different functions for plotting (images, results, ...)
'''

from .settings import *
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # pyright: ignore[reportMissingModuleSource]


'''
Function which plots uncertainty bar in vis_frame
call with: 
   canvas --> cannvas 
   bar_width --> width of bar
   bar_height --> height  of bar
   percent --> percent value for uncertainty
   color --> color to use (depending on uncertainty)
'''
def draw_uncertainty_bar(canvas: tk.Canvas, 
                         bar_width: float, 
                         bar_height: float, 
                         percent: float, 
                         color: str):
    canvas.delete("all")
    filled = int(bar_width * (percent / 100))
    create_rounded_rect(canvas, 0, 0, bar_width, bar_height, radius=10, fill="gray", outline="")
    create_rounded_rect(canvas, 0, 0, filled, bar_height, radius=10, fill=color, outline="")
    canvas.create_text(bar_width//2, bar_height//2, text=f"{percent:.1f}%", fill="white", font=("Arial", 10, "bold"))


'''
Function which plots accuracy for different models (start frame)
Gets folder to use and uses all csv-files from that folder
'''
def plot_accuracy_popup():
    folder = filedialog.askdirectory(title="Select 'retrained model' Folder")
    if not folder:
        return

    csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not csv_files:
        messagebox.showerror("Error", "No .csv files found in the selected folder.")
        return

    model_names = []
    accuracies = []

    for csv_file in csv_files:
        file_path = os.path.join(folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            if accuracy_column not in df.columns:
                continue
            avg_acc = df[accuracy_column].tail(last_n_epochs).mean()
            model_names.append(os.path.splitext(csv_file)[0])  # Remove .csv
            accuracies.append(avg_acc)
        except Exception:
            continue

    if not model_names:
        messagebox.showwarning("No Valid Files", f"No valid .csv files with '{accuracy_column}' found.")
        return

    # Create new popup window
    popup = tk.Toplevel()
    popup.title("Class Accuracy Plot")
    popup.geometry("1000x500")

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(model_names, accuracies, marker='o', linestyle='-', linewidth=2)
    ax.set_title(f"Retrained Models - Avg Class Accuracy (Last {last_n_epochs} Epochs)")
    ax.set_xlabel("Model Name (CSV Filename)")
    ax.set_ylabel("Class Accuracy")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()

    # Embed the figure in the popup window
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


'''
Function which creates rectangle (uncertainty bar)
call with: 
  canvas --> canvas where to plot 
  x1: float --> position x1
  y1: float --> position y1 
  x2: float --> position x2
  y2: float --> position y2
  radius: int = 10 --> radius of corners
'''
def create_rounded_rect(canvas, 
                        x1: float, 
                        y1: float, 
                        x2: float, 
                        y2: float, 
                        radius: int = 10, 
                        **kwargs):
    points = [
        x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
        x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
        x1, y2, x1, y2-radius, x1, y1+radius, x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

