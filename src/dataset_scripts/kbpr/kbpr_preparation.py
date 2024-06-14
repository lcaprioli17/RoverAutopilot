import os

import pandas as pd

from src import data_preparation

kbpr_img_path = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
kbpr_sensor_path = 'D:/Dataset/Rover/KBPR/dataset/sensor/imu'
kbpr_label_path = 'D:/Dataset/Rover/KBPR/dataset/label'

# Integrate acceleration to get velocity as a label
data_preparation.get_velocity(kbpr_sensor_path, 'imu_standard_timestamp.csv', time_threshold=10)

# Interpolate the sensor measurements to get a label for each image
# data_preparation.interpolate_frame_sensor(os.path.join(kbpr_img_path, 'full'), kbpr_sensor_path, 'wVelocity-imu_standard_timestamp.csv', sub_fixes=['LocCam_', '_0.png'])

# # Take the sensor path and select the labels that will be used to train the model
# data_preparation.get_labels(kbpr_sensor_path, 'frame_sensor_imu_standard_timestamp.csv', kbpr_label_path, ['acc_x', 'gyro_z'])
#
# # Split the image dataset and labels in train, validation and test sets and saves them in /train, /val and /test
# data_preparation.train_val_test_split(kbpr_img_path, kbpr_label_path)
