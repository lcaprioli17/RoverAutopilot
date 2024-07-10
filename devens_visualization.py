import os
import re
import pandas as pd

from src import data_visualization

devens_data_root = 'D:/Dataset/devens_snowy_fixed'
devens_img_path = 'D:/Dataset/devens_snowy_fixed/dataset/image'
devens_sensor_path = 'D:/Dataset/devens_snowy_fixed/dataset/sensor'
devens_label_path = 'D:/Dataset/devens_snowy_fixed/dataset/label'
losses = 'logs/epoch_loss.csv'

# data_visualization.aggregate_and_print_stats(devens_data_root, 'data_out.csv')

# data_visualization.line_plot(losses)

min_seq = 1000
min_path = None
for folder in os.listdir(devens_data_root):
    path = os.path.join(devens_data_root, folder)
    if len(os.listdir(path)) < min_seq:
        min_seq = len(os.listdir(path))
        min_path = path

print(min_seq, min_path)

