import os
import re
import pandas as pd

from src import data_preparation

ameds_data_root = 'D:/Dataset/Rover/AMEDS'
ameds_img_path = 'D:/Dataset/Rover/AMEDS/dataset/image'
ameds_sensor_path = 'D:/Dataset/Rover/AMEDS/dataset/sensor'
ameds_label_path = 'D:/Dataset/Rover/AMEDS/dataset/label'

# Move images to the dataset folder
data_preparation.images_to_dataset(ameds_data_root, 'MainCamImages', prefix=['MainCam_'], suffix=['.jpg'], pivot='_', crop_ratio=0.1)

# Merge and create the csv file of the sensor slices
# data_preparation.sensor_to_dataset(ameds_data_root, 'GnssBodyVelocity.csv', 'GnssBodyVelocity.csv')
# data_preparation.sensor_to_dataset(ameds_data_root, 'GnssBodyAngularVelocity.csv', 'GnssBodyAngularVelocity.csv')

# Convert the time in microseconds to timestamp
# data_preparation.sec_to_timestamp(cpet_sensor_path, 'cmd-velocities.csv', time_feature='# time [s]', formatting=True)

# Interpolate the sensor measurements to get a label for each image
# data_preparation.interpolate_frame_sensor(ameds_img_path, ameds_sensor_path, 'GnssBodyVelocity.csv', timestamp='Source timestamp [unix_s]', sub_fixes=['.jpg'], drop_column=['Rover ID [-]'])
# data_preparation.interpolate_frame_sensor(ameds_img_path, ameds_sensor_path, 'GnssBodyAngularVelocity.csv', timestamp='Source timestamp [unix_s]', sub_fixes=['.jpg'], drop_column=['Rover ID [-]'])

# Take the sensor path and select the labels that will be used to train the model
# data_preparation.get_labels(cpet_sensor_path, 'frame_sensor_timestamp-cmd-velocities.csv', cpet_label_path, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])

# Split the image dataset and labels in train, validation and test sets and saves them in /train, /val and /test
# data_preparation.train_val_test_split(cpet_img_path, cpet_label_path)