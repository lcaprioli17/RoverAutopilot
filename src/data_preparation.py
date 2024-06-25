import glob
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import os
import shutil
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.integrate import cumtrapz
import pytz


def move_images_with_specific_name(
        src_dir: str,
        dest_dirs: list[str],
        prefixes: list[str] = None,
        suffixes: list[str] = None,
        pivot: str = None
):
    """
    Organize images by prefixes and suffixes from a common directory src_dir to the directories specified in the dest_dirs list.

    :param src_dir: Source directory where all the images are found. The full path of the directory is needed.
    :param dest_dirs: List of destination directories to organize the images by prefixes and suffixes. The full path of the directories is needed.
    :param prefixes: List of prefixes used to organize the images in the dest_dirs. Defaults to an empty list.
    :param suffixes: List of suffixes used to organize the images in the dest_dirs. Defaults to an empty list.
    :param pivot: Character to split the file name for renaming (optional).
    """
    if suffixes is None:
        suffixes = ['']
    if prefixes is None:
        prefixes = ['']

    if len(dest_dirs) != len(suffixes) or len(dest_dirs) != len(prefixes):
        raise ValueError('ERROR: Number of destination directories and number of file suffixes/prefixes are different!')

    for i in range(len(dest_dirs)):
        # Ensure the destination directory exists
        if not os.path.exists(dest_dirs[i]):
            os.makedirs(dest_dirs[i])

    # Iterate over all files in the source directory
    for file_name in os.listdir(src_dir):
        for i in range(len(dest_dirs)):
            # Check if the file name contains the specific name
            if file_name.endswith(suffixes[i]) or file_name.startswith(prefixes[i]):
                # Construct full file paths
                src_file = os.path.join(src_dir, file_name)
                new_name = file_name
                if pivot is not None:
                    # Find the position of the first occurrence of the character
                    pos = new_name.find(pivot)
                    new_name = new_name[pos + 1:]
                dst_file = os.path.join(dest_dirs[i], new_name)

                # Move the file
                shutil.copy(src_file, dst_file)


def concatenate_files(
        input_file: str,
        output_file: str
):
    """
    Concatenate content from the input file to the output file.

    :param input_file: File to be concatenated.
    :param output_file: File to concatenate.
    """
    with open(output_file, 'a') as outfile:
        with open(input_file, 'r') as infile:
            outfile.write('\n')
            outfile.write(infile.read())


def images_from_root(
        root_dir: str,
        src_dir: str,
        dst_dirs: list[str],
        suffices: list[str]
):
    """
    Search in a root directory for the images in src_dir to organize in dst_dirs by suffices.

    :param root_dir: Root directory where the subdirectories are located.
    :param src_dir: Name of the subdirectories to search from the root.
    :param dst_dirs: List of destination directories to organize the images by suffices.
    :param suffices: List of suffices used to organize the images in the dst_dirs.
    """
    data_root = root_dir + '/data'

    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    dst_dirs.sort()
    suffices.sort()
    i = 0

    # List all entries in the directory
    print('\nOrganizing images...')
    for path, dirs, files in tqdm(os.walk(data_root)):
        for subdir in dirs:
            if subdir == src_dir:
                src = os.path.join(path, src_dir)
                dst = os.path.join(path, dst_dirs[i])
                sfx = suffices[i]
                move_images_with_specific_name(src, [dst], [sfx])
                i += i


def images_to_dataset(
        root_dir: str,
        src_dir: str,
        prefix: list[str] = None,
        suffix: list[str] = None,
        pivot: str = None
):
    """
    Move images from subdirectories with the same name to a common directory.

    :param root_dir: Root directory where the subdirectories are located.
    :param src_dir: Name of the subdirectories to search from the root.
    :param prefix: Prefix of the images to move.
    :param suffix: Suffix of the images to move.
    :param pivot: Character to split the file name for renaming (optional).
    """
    data_root = root_dir + '/data'
    dataset_dir = root_dir + '/dataset/image/full'

    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    # Create the destination directory if it does not exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print('\nMoving images to dataset folder...')
    for dir_path, dir_names, file_names in tqdm(os.walk(data_root)):
        for dir_name in dir_names:
            if src_dir in dir_name:
                subdirectory_src = os.path.join(dir_path, dir_name)
                move_images_with_specific_name(subdirectory_src, [dataset_dir], prefix, suffix, pivot)


def sensor_to_dataset(
        root_dir: str,
        sensor_file: str,
        csv_file: str
):
    """
    Search the root directory for files with the same name containing the data of the sensors and merge them.

    :param root_dir: Root directory where the files are located.
    :param sensor_file: Full name of the file to search.
    :param csv_file: CSV file to save the merged sensor data.
    """
    data_root = root_dir + '/data'
    dataset_dir = root_dir + '/dataset/sensor'
    output_sensor = os.path.join(dataset_dir, sensor_file)
    csv_sensor = os.path.join(dataset_dir, csv_file)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    # List all entries in the directory
    print('\nMerging sensor files...')
    for path, dirs, files in tqdm(os.walk(data_root)):
        for file in files:
            if file == sensor_file:
                input_sensor = os.path.join(path, file)
                concatenate_files(input_sensor, output_sensor)
                convert_txt_to_csv(output_sensor, csv_sensor)


def convert_txt_to_csv(
        input_file: str,
        output_file: str
):
    """
    Convert a text file to a CSV file.

    :param input_file: File to convert.
    :param output_file: Converted CSV file.
    """
    with open(input_file, 'r') as infile:
        # Read lines from the text file
        lines = infile.readlines()

    with open(output_file, "w", newline='') as csvfile:
        for line in lines:
            csvfile.write(line)


def add_char_before_position(
        string: str,
        char: str,
        position: int
) -> str:
    """
    Add characters at a given position in a string.

    :param string: String to modify.
    :param char: Characters to add to the string.
    :param position: Position where to add the characters.
    :return: Modified string.
    """
    if position < 0 or position > len(string):
        raise ValueError("Position is out of range")
    return string[:position] + char + string[position:]


def standardize_timestamp(
        csv_file: str,
        standard_len: int,
        csv_dst: str,
        csv_delimiter: str = ',',
        remove: bool = False,
        to_remove: str = ''
):
    """
    Converts the timestamps to a standardized format, filling with zeroes if necessary.

    :param csv_file: Path to the file to standardize.
    :param standard_len: Length of the standard format.
    :param csv_dst: Path where to save the standardized CSV.
    :param csv_delimiter: Delimiter of the columns in the CSV to read.
    :param remove: Boolean indicating if characters should be removed from the timestamp.
    :param to_remove: Character to be removed from the timestamp.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, delimiter=csv_delimiter)

    print('\nStandardizing timestamps...')
    for index, value in tqdm(df['timestamp'].items()):
        if len(value) == (standard_len - 2):
            new_timestamp = add_char_before_position(value, '00', len(value) - 1)
            df.at[index, 'timestamp'] = new_timestamp

        elif len(value) == (standard_len - 1):
            new_timestamp = add_char_before_position(value, '0', len(value) - 2)
            df.at[index, 'timestamp'] = new_timestamp

        if remove is True and to_remove != '':
            modified_timestamp = df.at[index, 'timestamp'].replace(to_remove, '')
            df.at[index, 'timestamp'] = modified_timestamp

    df.to_csv(csv_dst, index=False)


def sec_to_timestamp(
        sensors_dir: str,
        sensor_file: str,
        time_feature: str = None
):
    """
    Convert Unix timestamps in a sensor file to a formatted date string.

    :param sensors_dir: Path to the directory containing the sensor measurements.
    :param sensor_file: Name of the sensor file.
    :param time_feature: Feature containing the Unix timestamps.
    """
    if time_feature is None:
        return
    # Read the CSV file
    sensor_df = pd.read_csv(str(os.path.join(sensors_dir, sensor_file)))

    # Apply the conversion function to the 'timestamp' column
    sensor_df['timestamp'] = sensor_df[time_feature].apply(convert_to_formatted_date)

    if time_feature != 'timestamp':
        # Remove column 'B'
        sensor_df.drop(columns=[time_feature], inplace=True)
        # Move column 'timestamp' to the first position (index 0)
        col_to_move = sensor_df.pop('timestamp')
        sensor_df.insert(0, 'timestamp', col_to_move)

    # Save the updated DataFrame back to a CSV file
    sensor_df.to_csv(sensors_dir + '/timestamp-' + sensor_file, index=False)


# Function to convert Unix timestamp to formatted date string
def convert_to_formatted_date(
        unix_microseconds: float
) -> str:
    """
    Convert a Unix timestamp to a formatted date string.

    :param unix_microseconds: Unix timestamp to convert.
    :return: Formatted date string.
    """
    # Convert Unix timestamp to UTC datetime object
    dt_utc = datetime.utcfromtimestamp(unix_microseconds)

    # Define UTC-4 timezone
    utc_minus_4 = pytz.timezone('America/New_York')

    # Convert UTC datetime object to UTC-4 datetime object
    dt_utc_minus_4 = pytz.utc.localize(dt_utc).astimezone(utc_minus_4)

    # Format with exactly six microseconds
    formatted_dt = dt_utc_minus_4.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-6] + '{:06d}'.format(dt_utc_minus_4.microsecond)
    return formatted_dt


# def get_velocity(
#         sensors_dir,
#         sensor_file,
#         time_threshold
# ):
#     sensor_df = pd.read_csv(str(os.path.join(sensors_dir, sensor_file)))
#
#     # Convert timestamp string to datetime
#     sensor_df['formatted_timestamp'] = pd.to_datetime(sensor_df['timestamp'], format='%Y_%m_%d_%H_%M_%S_%f')
#     # Step 1: Convert timestamps to time intervals (in seconds)
#     sensor_df['time_interval'] = sensor_df['formatted_timestamp'].diff().dt.total_seconds()
#     # Fill the first time_interval with 0
#     sensor_df.loc[0, 'time_interval'] = 0
#
#     # Identify switching points where the time interval exceeds the threshold
#     switch_points = sensor_df.index[sensor_df['time_interval'] > time_threshold].tolist()
#     switch_points = [0] + switch_points + [len(sensor_df)]  # Add the start and end indices
#
#     for point in switch_points:
#         sensor_df.loc[point, 'time_interval'] = 0
#
#     # Initialize velocity columns
#     sensor_df['vel_x'] = 0.0
#     sensor_df['vel_y'] = 0.0
#     sensor_df['vel_z'] = 0.0
#
#     # Iterate through the segments defined by the switch points
#     for i in range(len(switch_points) - 1):
#         start_idx = switch_points[i]
#         end_idx = switch_points[i + 1]
#         segment = sensor_df.iloc[start_idx:end_idx]
#
#         # Integrate acceleration to obtain velocity for the current segment
#         segment_vel_x = np.cumsum(segment['acc_x'] * segment['time_interval'])
#         segment_vel_y = np.cumsum(segment['acc_y'] * segment['time_interval'])
#         segment_vel_z = np.cumsum(segment['acc_z'] * segment['time_interval'])
#
#         sensor_df.loc[start_idx:end_idx - 1, 'vel_x'] = segment_vel_x
#         sensor_df.loc[start_idx:end_idx - 1, 'vel_y'] = segment_vel_y
#         sensor_df.loc[start_idx:end_idx - 1, 'vel_z'] = segment_vel_z
#
#     # Drop the 'time_interval' column if no longer needed
#     sensor_df.drop(columns=['formatted_timestamp', 'time_interval'], inplace=True)
#     sensor_df.to_csv(sensors_dir + '/wVelocity-' + sensor_file, mode='w', index=False)
#
#
# def get_velocity_std(
#         sensors_dir,
#         sensor_file,
# ):
#     sensor_df = pd.read_csv(str(os.path.join(sensors_dir, sensor_file)))
#
#     time_interval = 0.008
#     # Fill the first time_interval with 0
#     sensor_df.loc[0, 'time_interval'] = 0
#     # Initialize velocity columns
#     sensor_df['vel_x'] = 0.0
#     sensor_df['vel_y'] = 0.0
#     sensor_df['vel_z'] = 0.0
#
#     # Integrate acceleration to obtain velocity for the current segment
#     sensor_df['vel_x'] = np.cumsum(sensor_df['acc_x'] * time_interval)
#     sensor_df['vel_y'] = np.cumsum(sensor_df['acc_y'] * time_interval)
#     sensor_df['vel_z'] = np.cumsum(sensor_df['acc_z'] * time_interval)
#
#     sensor_df.to_csv(sensors_dir + '/wVelocity-' + sensor_file, mode='w', index=False)


def interpolate_frame_sensor(
        img_dir: str,
        sensors_dir: str,
        sensor_file: str,
        n: int = 1,
        interpolation: int = 1,
        sub_fixes: list = None
):
    """
    For each image in img_dir, interpolates measurements from sensor_file and saves them in a file.
    At the end, there will be a file containing a measurement of the specified sensor for each image.

    :param img_dir: Path to the directory containing the images.
    :param sensors_dir: Path to the directory containing the sensor measurements, used both to get the source sensor and to save interpolated values to the destination sensor.
    :param sensor_file: Name of the sensor to use for interpolating image timestamps.
    :param n: Number of neighboring measurements to interpolate.
    :param interpolation: Direction for interpolation: 1 forward, -1 backward, 0 both.
    :param sub_fixes: Characters that need to be removed from the timestamp.
    :raises ValueError: If the sensor file is not in CSV format or if interpolation direction is invalid.
    """
    if sub_fixes is None:
        sub_fixes = ['']

    if not sensor_file.endswith('.csv'):
        raise ValueError("Convert the sensor file to CSV format!")

    img_dir = os.path.join(img_dir, 'full')
    sensor_path = sensors_dir + '/' + sensor_file
    sensor_df = pd.read_csv(sensor_path)
    frame_sensor_df = pd.DataFrame(columns=sensor_df.columns.str.strip())
    frame_sensor_df.to_csv(sensors_dir + '/frame_sensor_' + sensor_file, mode='w', index=False)
    frame_sensor_row = None
    checkpoint = 0
    prev_row = None
    img_num = 0

    print('\nInterpolating measurements for the images...')
    for filename in tqdm(os.listdir(img_dir)):
        img_timestamp = filename

        for substring in sub_fixes:
            img_timestamp = img_timestamp.replace(substring, '')

        for index, row in sensor_df.iloc[checkpoint:].iterrows():
            if prev_row is None and img_timestamp < row['timestamp']:
                # print(f"Image number {img_num} timestamp: {img_timestamp} is before the first row timestamp at index {checkpoint}: {row['timestamp']}")
                frame_sensor_row = interpolate_row(sensor_df, img_timestamp, index, interpolation, n)
                img_num += 1
                break

            elif prev_row is not None and prev_row['timestamp'] <= img_timestamp < row['timestamp']:
                # print(f"Nearest timestamp for image number {img_num} with timestamp {img_timestamp} is {prev_row['timestamp']} at index {checkpoint}")
                frame_sensor_row = interpolate_row(sensor_df, img_timestamp, index - 1, interpolation, n)
                img_num += 1
                break

            prev_row = row
            checkpoint = index

        # frame_sensor_df = frame_sensor_row
        frame_sensor_row.to_csv(sensors_dir + '/frame_sensor_' + sensor_file, mode='a', index=False, header=False)


def interpolate_row(
        src_df: pd.DataFrame,
        timestamp: str,
        index: int,
        interpolation: int,
        n: int
) -> pd.DataFrame:
    """
    Interpolate the measurements for the image with the specific timestamp using values from src_df.
    Starting from index, it computes the measurements for the image following the interpolation method and considering n neighbors.

    :param src_df: Source dataframe for the sensor measurements.
    :param timestamp: Timestamp of the image for which to interpolate new measurements.
    :param index: Index from where to start the interpolation.
    :param interpolation: Direction for interpolation: 1 forward, -1 backward, 0 both.
    :param n: Number of neighboring measurements to interpolate.
    :return: DataFrame with interpolated measurements.
    :raises ValueError: If interpolation direction is invalid.
    """
    if interpolation == -1:
        if index - n < 0:
            interpolated_row = src_df.iloc[0:index, 1:].mean()
        else:
            interpolated_row = src_df.iloc[index - n:index, 1:].mean()

    elif interpolation == 1:
        if index + n > len(src_df):
            interpolated_row = src_df.iloc[index:len(src_df), 1:].mean()
        else:
            interpolated_row = src_df.iloc[index:index + n, 1:].mean()

    elif interpolation == 0:
        if index - n < 0:
            interpolated_row = src_df.iloc[0:index + n, 1:].mean()
        elif index + n > len(src_df):
            interpolated_row = src_df.iloc[index - n:len(src_df), 1:].mean()
        else:
            interpolated_row = src_df.iloc[index - n:index + n, 1:].mean()

    else:
        raise ValueError("Interpolation should be 1: forward, -1: backward, 0: both.")

    interpolated_row = interpolated_row.to_frame().T
    interpolated_row.insert(0, 'timestamp', timestamp)
    # print(f'The nearest row added for image {timestamp} had index {index}. Now it has timestamp {interpolated_row.iloc[-1]}.\n')
    return interpolated_row


def get_labels(
        sensors_dir: str,
        sensor_file: str,
        labels_dir: str,
        features: list,
        normalization_method: str = 'min-max'
):
    """
    Extract the labels from the sensor file and normalize them.

    :param sensors_dir: Path to the directory where all the sensors are saved.
    :param sensor_file: File containing the specific sensor measurements.
    :param labels_dir: Path where to save the labels file.
    :param features: Features to extract and use as labels.
    :param normalization_method: Method to normalize the features ('min-max' or 'z-score').
    :raises ValueError: If the normalization method is invalid.
    """
    sensor_path = sensors_dir + '/' + sensor_file
    sensor_df = pd.read_csv(sensor_path)

    labels = pd.DataFrame(columns=features, data=sensor_df[features])

    # Normalize the features
    if normalization_method == 'min-max':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif normalization_method == 'z-score':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'min-max' or 'z-score'.")

    normalized_labels = scaler.fit_transform(labels)
    normalized_labels_df = pd.DataFrame(columns=features, data=normalized_labels)

    normalized_labels_df.to_csv(labels_dir + '/labels.csv', mode='w', index=False, header=True)


def train_val_test_split(
        image_dir: str,
        label_dir: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
):
    """
    Split the dataset into training, validation, and test sets.

    :param image_dir: Path where all the images are located.
    :param label_dir: Path where the labels are located.
    :param val_ratio: Ratio of the validation set with respect to the training set.
    :param test_ratio: Ratio of the test set with respect to the training set.
    :raises ValueError: If there is a mismatch between image and label counts.
    """

    image_full = os.listdir(image_dir + '/full')
    train_dst = image_dir + '/train'
    val_dst = image_dir + '/val'
    test_dst = image_dir + '/test'
    label_file = label_dir + '/labels.csv'
    sensor_df = pd.read_csv(label_file)

    if os.path.exists(train_dst):
        files = os.listdir(train_dst)
        for f in files:
            os.remove(os.path.join(train_dst, f))
    if os.path.exists(val_dst):
        files = os.listdir(val_dst)
        for f in files:
            os.remove(os.path.join(val_dst, f))
    if os.path.exists(test_dst):
        files = os.listdir(test_dst)
        for f in files:
            os.remove(os.path.join(test_dst, f))

    if len(image_full) != len(sensor_df):
        raise ValueError("Mismatch in dimensions between images and labels in the dataset.")

    print(f'\nDataset and label dimensions: {len(image_full)}')

    x_train, x_val, y_train, y_val = train_test_split(image_full, sensor_df, test_size=val_ratio, shuffle=False)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_ratio, shuffle=False)

    if len(x_train) != len(y_train):
        raise ValueError("Mismatch in dimensions between training images and labels.")
    if len(x_val) != len(y_val):
        raise ValueError("Mismatch in dimensions between validation images and labels.")
    if len(x_test) != len(y_test):
        raise ValueError("Mismatch in dimensions between test images and labels.")

    print(f'\nAfter the training-validation-test split with ratio {val_ratio} and {test_ratio} the dataset and label dimensions are:')
    print(len(x_train))
    print(len(x_val))
    print(len(x_test))

    print('\nSaving train, validation and test labels to file...')
    y_train.to_csv(label_dir + '/train/labels.csv', mode='w', index=False)
    y_val.to_csv(label_dir + '/val/labels.csv', mode='w', index=False)
    y_test.to_csv(label_dir + '/test/labels.csv', mode='w', index=False)

    print('\nMoving training set images...')
    for image in tqdm(x_train):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(train_dst, image)))

    print('\nMoving validation set images...')
    for image in tqdm(x_val):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(val_dst, image)))

    print('\nMoving test set images...')
    for image in tqdm(x_test):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(test_dst, image)))

    if len(os.listdir(train_dst)) != len(y_train):
        raise ValueError("Mismatch in dimensions between training images and labels after move.")
    if len(os.listdir(val_dst)) != len(y_val):
        raise ValueError("Mismatch in dimensions between validation images and labels after move.")
    if len(os.listdir(test_dst)) != len(y_test):
        raise ValueError("Mismatch in dimensions between test images and labels after move.")