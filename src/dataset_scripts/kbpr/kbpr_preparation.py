import os

import pandas as pd

from src import data_preparation


kbpr_img_path = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
kbpr_sensor_path = 'D:/Dataset/Rover/KBPR/dataset/sensor/imu'
kbpr_label_path = 'D:/Dataset/Rover/KBPR/dataset/label'

# Take the sensor path and select the labels that will be used to train the model
data_preparation.get_labels(kbpr_sensor_path, 'frame_sensor_imu_standard_timestamp.csv', kbpr_label_path, ['acc_x', 'gyro_z'])

# Split the image dataset and labels in train, validation and test sets and saves them in /train, /val and /test
data_preparation.train_val_test_split(kbpr_img_path, kbpr_label_path)
