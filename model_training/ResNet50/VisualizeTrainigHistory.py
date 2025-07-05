import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
hist = pd.read_csv('History.csv')
fig, axes = plt.subplots(5, 5, figsize = (25, 25))
row_counter = 0
col_counter = 0
for col in hist.columns.values.tolist():
    vals = hist.loc[:, col]
    axes[row_counter, col_counter].plot(np.arange(len(vals)), vals)
    axes[row_counter, col_counter].set_title(f'Development of {col} over epochs')
    axes[row_counter, col_counter].set_xlabel('Epoch')
    axes[row_counter, col_counter].set_ylabel(col)
    col_counter += 1
    if col_counter > 4:
        col_counter = 0
        row_counter += 1
fig.suptitle('Development of Performance Metrics During Model Training', y = 0.92, fontsize = 20)
plt.savefig('TrainingHistory.png')
