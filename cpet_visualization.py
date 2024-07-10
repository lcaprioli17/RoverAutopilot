import os
import re
import pandas as pd

from src import data_visualization

cpet_groundtruth_normalized = 'D:/Dataset/Rover/CPET/dataset/label/labels.csv'
cpet_groundtruth = 'D:/Dataset/Rover/CPET/dataset/sensor/frame_sensor_timestamp_velocity-estimates.csv'
cpet_split_path = 'D:/Dataset/Rover/CPET/dataset'
losses = '../../logs/epoch_loss.csv'
traverse = 6

# data_visualization.multi_line_plot(cpet_groundtruth, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
# data_visualization.line_plot(losses)
# data_visualization.multi_line_plot(os.path.join(cpet_split_path, 'label/train/labels.csv'), ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
# data_visualization.multi_line_plot(os.path.join(cpet_split_path, 'label/val/labels.csv'), ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])

for i in range(1, traverse + 1):
    data_visualization.generate_video(os.path.join(cpet_split_path, f'image/full/{str(i)}'), os.path.join(cpet_split_path, f'sensor/{str(i)}/frame_sensor_timestamp_cmd-velocities.csv'), f'C:/Users/VR Admin/Desktop/RoverAutopilot/5_Pictures/cpet/traverse_{str(i)}.mp4')
