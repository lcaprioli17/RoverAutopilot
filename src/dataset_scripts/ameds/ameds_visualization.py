import os
import re
import pandas as pd

from src import data_visualization

ameds_groundtruth_normalized = 'D:/Dataset/Rover/CPET/dataset/label/labels.csv'
ameds_groundtruth = 'D:/Dataset/Rover/CPET/dataset/sensor/frame_sensor_timestamp-cmd-velocities.csv'
ameds_split_path = 'D:/Dataset/Rover/CPET/dataset'
losses = '../../logs/epoch_loss.csv'

# data_visualization.multi_line_plot(cpet_groundtruth, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
# data_visualization.line_plot(losses)
# data_visualization.multi_line_plot(os.path.join(cpet_split_path, 'label/train/labels.csv'), ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
# data_visualization.multi_line_plot(os.path.join(cpet_split_path, 'label/val/labels.csv'), ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])

data_visualization.generate_video(os.path.join(ameds_split_path, 'image/full'), os.path.join(ameds_split_path, 'label/labels.csv'), ameds_split_path + '/train_video.mp4')