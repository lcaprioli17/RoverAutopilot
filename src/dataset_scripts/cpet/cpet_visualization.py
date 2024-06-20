import os
import re
import pandas as pd

from src import data_visualization

cpet_groundtruth_normalized = 'D:/Dataset/Rover/CPET/dataset/label/labels.csv'
cpet_groundtruth = 'D:/Dataset/Rover/CPET/dataset/sensor/frame_sensor_timestamp-cmd-velocities.csv'
losses = '../../logs/epoch_loss.csv'

# data_visualization.multi_line_plot(cpet_groundtruth, ['x_linear_velocity [m/s]', 'z_angular_velocity [rad/s]'])
data_visualization.line_plot(losses)
