'''
This script starts the GUI workflow
'''
from modules.frames.start_frame import start_frame
import tkinter as tk


root = tk.Tk()
root.title("XAI4HiL Machine Learning")
root.eval("tk::PlaceWindow . center")

frame = start_frame(image_list=[],
                      root = root)

root.mainloop()
