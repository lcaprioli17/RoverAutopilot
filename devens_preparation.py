import os
import re
import pandas as pd

from src import data_preparation

devens_data_root = 'D:/Dataset/devens_snowy_fixed'
devens_img_path = 'D:/Dataset/devens_snowy_fixed/dataset/image'
devens_sensor_path = 'D:/Dataset/devens_snowy_fixed/dataset/sensor'
devens_label_path = 'D:/Dataset/devens_snowy_fixed/dataset/label'
cmd_bound = [[-2.355191, -2.958401, -1.673000, -1.280000], [8.078414, 3.405209, 3.185000, 1.304000]]
cmd_dev = [[0.763999, 0.039917, -0.043323, -0.012103], [1.087240, 0.376899, 0.180645, 0.116507]]

# Move images to the dataset folder
# data_preparation.traverse_to_csv(devens_data_root)

# Merge and create the csv file of the sensor slices

# Convert the time in microseconds to timestamp

# Interpolate the sensor measurements to get a label for each image

# Take the sensor path and select the labels that will be used to train the model
data_preparation.repeat_get_labels(devens_data_root, 'data_out.csv', devens_data_root, ['vx', 'vy', 'vz', 'omega_z'], cmd_dev[0], cmd_dev[1], 'z-score')

# Split the image dataset and labels in train, validation and test sets and saves them in /train, /val and /test
# data_preparation.train_val_test_split(devens_img_path, devens_label_path)
