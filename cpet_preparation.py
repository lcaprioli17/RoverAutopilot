import os
import re
import pandas as pd

from src import data_preparation

cpet_data_root = 'D:/Dataset/Rover/CPET'
cpet_img_path = 'D:/Dataset/Rover/CPET/dataset/image/full'
cpet_sensor_path = 'D:/Dataset/Rover/CPET/dataset/sensor'
cpet_label_path = 'D:/Dataset/Rover/CPET/dataset/label'
cmd_bound = [[-1.5, -0.4], [1.5, 0.4]]
estimate_bound = [[-0.5626250647999138, -0.9563750857151744], [0.7628958745885401, 0.9774890442063396]]

# Move images to the dataset folder
# data_preparation.images_to_dataset(cpet_data_root, 'mono_image', prefix=['frame'], suffix=[''], pivot='_')
# data_preparation.move_n_to_n_image(cpet_data_root, 'mono_image', pivot='_')
# data_preparation.traverse_to_csv(cpet_data_root)

# Merge and create the csv file of the sensor slices
# data_preparation.sensor_to_dataset(cpet_data_root, 'cmd-velocities.txt', 'cmd-velocities.csv')
# data_preparation.move_n_to_n_sensor(cpet_data_root, 'cmd-velocities.txt', 'cmd-velocities.csv')
# data_preparation.move_n_to_n_sensor(cpet_data_root, 'velocity-estimates.txt', 'velocity-estimates.csv')

# Convert the time in microseconds to timestamp
# data_preparation.sec_to_timestamp(cpet_sensor_path, 'cmd-velocities.csv', time_feature='# time [s]', formatting=True)
# data_preparation.repeat_sec_to_timestamp(cpet_sensor_path, 'velocity-estimates.csv', time_feature='# time [s]', formatting=True)

# Interpolate the sensor measurements to get a label for each image
# data_preparation.interpolate_frame_sensor(cpet_img_path, cpet_sensor_path, 'timestamp_cmd-velocities.csv', sub_fixes=['.png'])
# data_preparation.repeat_interpolation(cpet_img_path, cpet_sensor_path, 'timestamp_velocity-estimates.csv', sub_fixes=['.png'])

# data_preparation.concatenate_csv('D:/Dataset/Rover/CPET/dataset/sensor', 'frame_sensor_timestamp_velocity-estimates.csv')

# Take the sensor path and select the labels that will be used to train the model
# data_preparation.get_labels(cpet_sensor_path, 'frame_sensor_timestamp-cmd-velocities.csv', cpet_label_path, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
data_preparation.repeat_get_labels(cpet_sensor_path, 'frame_sensor_timestamp_cmd-velocities.csv', cpet_label_path, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'], cmd_bound[0], cmd_bound[1], 'min-max')
# data_preparation.repeat_get_labels(cpet_sensor_path, 'frame_sensor_timestamp_velocity-estimates.csv', cpet_label_path, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])

# Split the image dataset and labels in train, validation and test sets and saves them in /train, /val and /test
# data_preparation.train_val_test_split(cpet_img_path, cpet_label_path)
# data_preparation.repeat_train_val_test_split(cpet_img_path, cpet_label_path)
