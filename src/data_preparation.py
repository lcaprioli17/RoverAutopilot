import csv
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import pytz


def move_images_with_specific_name(
        src_dir: str,
        dest_dirs: list[str],
        prefixes: list[str] = None,
        suffixes: list[str] = None,
        pivot: str = None,
        crop_ratio: float = 0
):
    """
    :param src_dir: Source directory where all the images are found. The full path of the directory is needed.
    :param dest_dirs: List of destination directories to organize the images by prefixes and suffixes. The full path of the directories is needed.
    :param prefixes: List of prefixes used to organize the images in the dest_dirs. Defaults to an empty list.
    :param suffixes: List of suffixes used to organize the images in the dest_dirs. Defaults to an empty list.
    :param pivot: Character to split the file name for renaming (optional).
    :param crop_ratio: Ratio for cropping the image. Should be between 0 and 1.

    :raises ValueError: If the number of destination directories does not match the number of prefixes or suffixes.

    This function iterates through all files in src_dir. For each file, it checks if the file name matches any of the specified prefixes or suffixes. If a match is found, the function moves the image file to the corresponding destination directory specified in dest_dirs.

    If pivot is provided, the function will split the file name at the pivot character and use the latter part as the new file name.

    If crop_ratio is provided and within valid range (0 < crop_ratio <= 1), the function crops the image by the specified ratio from each side (left, upper, right, lower).

    Example usage:
    src_dir = '/path/to/source/directory'
    dest_dirs = ['/path/to/destination1', '/path/to/destination2']
    prefixes = ['prefix1_', 'prefix2_']
    suffixes = ['_suffix1.jpg', '_suffix2.jpg']
    pivot = '_'
    crop_ratio = 0.1

    move_images_with_specific_name(src_dir, dest_dirs, prefixes, suffixes, pivot, crop_ratio)
    """
    if suffixes is None:
        suffixes = ['']
    if prefixes is None:
        prefixes = ['']

    # Check if the number of dest_dirs matches prefixes and suffixes
    if len(dest_dirs) != len(suffixes) or len(dest_dirs) != len(prefixes):
        raise ValueError('ERROR: Number of destination directories and number of file suffixes/prefixes are different!')

    # Ensure destination directories exist; create if not
    for i in range(len(dest_dirs)):
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

                # Open the image
                with Image.open(src_file) as img:
                    # Crop the image if crop_ratio is provided
                    if 0 < crop_ratio <= 1:
                        width, height = img.size
                        left = int(width * crop_ratio)
                        upper = int(height * crop_ratio)
                        right = int(width * (1 - crop_ratio))
                        lower = int(height * (1 - crop_ratio))
                        img = img.crop((left, upper, right, lower))

                    # Save the cropped image to the destination
                    img.save(dst_file)

                # Print progress message
                print(f'\rWrote image {new_name}', end='')

    # Optionally, return a message indicating success
    print("\nImage organization complete.")


def concatenate_files(
        input_file: str,
        output_file: str
):
    """
    :param input_file: File path of the file whose content will be concatenated.
    :param output_file: File path of the file to which content will be concatenated.

    This function reads the content of `input_file` and appends it to `output_file`.
    It does so by opening both files in read ('r') and append ('a') modes respectively,
    ensuring that the existing content in `output_file` remains intact while new content
    is added from `input_file`.

    Example usage:
    input_file = 'input.txt'
    output_file = 'output.txt'
    concatenate_files(input_file, output_file)
    """
    with open(output_file, 'a') as outfile:
        with open(input_file, 'r') as infile:
            outfile.write(infile.read())


def concatenate_csv(file_path: str, input_file: str):
    """
    Parameters:
    file_path (str): Directory path containing the CSV files to be concatenated.
    input_file (str): Name of the CSV file to concatenate.

    This function searches for all occurrences of `input_file` within `file_path` and its subdirectories.
    It reads each CSV file found, stores them as pandas DataFrames in a list, and then concatenates them into
    a single DataFrame (`concatenated_df`). Finally, it saves `concatenated_df` as a new CSV file named `input_file`
    in the same `file_path`.

    Example usage:
    file_path = '/path/to/csv_files/'
    input_file = 'concatenated.csv'
    concatenate_csv(file_path, input_file)
    """
    # Initialize an empty list to store dataframes
    dfs = []

    # Iterate through all files in file_path and its subdirectories
    for path, dirs, files in tqdm(os.walk(file_path)):
        for file in files:
            if file == input_file:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(os.path.join(path, file))
                dfs.append(df)

    # Concatenate all the dataframes in the list
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Construct the output file path
    output_file = os.path.join(file_path, input_file)

    # Save the concatenated dataframe to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as {output_file}")


def replace_small_numbers_in_df(
        df,
        threshold=10e-3
) -> pd.DataFrame:
    """
    Parameters:
    df (pandas.DataFrame): Input DataFrame where small numbers will be replaced.
    threshold (float): Threshold value below which numbers are considered small and replaced with zero. Default is 10e-3.

    Returns:
    pandas.DataFrame: DataFrame with small numbers replaced by zero.

    This function applies a lambda function to each element of the DataFrame `df`.
    If the absolute value of an element is less than `threshold`, it is replaced with zero.
    Otherwise, the element remains unchanged.

    Example usage:
    df = pd.DataFrame([[0.001, 0.002], [0.01, 0.02]])
    threshold = 0.01
    replaced_df = replace_small_numbers_in_df(df, threshold)
    """
    return df.applymap(lambda x: 0 if abs(x) < threshold else x)


def images_from_root(
        root_dir: str,
        src_dir: str,
        dst_dirs: list[str],
        suffices: list[str]
):
    """
    Parameters:
    root_dir (str): Root directory where subdirectories are located.
    src_dir (str): Name of the subdirectory containing images to be organized.
    dst_dirs (list of str): List of destination directories where images will be organized based on suffices.
    suffices (list of str): List of suffices used to organize the images in dst_dirs.

    This function searches through all subdirectories of `root_dir` for a specific subdirectory `src_dir`.
    Once found, it moves images from `src_dir` into corresponding directories in `dst_dirs` based on specified suffices.
    It uses the `move_images_with_specific_name` function to perform the organization.

    Raises:
    FileNotFoundError: If the "/data" folder under `root_dir` does not exist.

    Example:
    root_dir = '/path/to/root/'
    src_dir = 'source_images'
    dst_dirs = ['/path/to/destination1', '/path/to/destination2']
    suffices = ['_suffix1', '_suffix2']

    images_from_root(root_dir, src_dir, dst_dirs, suffices)
    """
    # Define the path to the data folder under root_dir
    data_root = os.path.join(root_dir, 'data')

    # Check if the data_root exists; raise error if not
    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    # Sort destination directories and suffices for organized processing
    dst_dirs.sort()
    suffices.sort()

    i = 0  # Initialize index for dst_dirs and suffices lists

    # Search for the src_dir in all subdirectories under data_root
    print('\nOrganizing images...')
    for path, dirs, files in tqdm(os.walk(data_root)):
        for subdir in dirs:
            if subdir == src_dir:
                # Construct source and destination paths
                src = os.path.join(path, src_dir)
                dst = os.path.join(path, dst_dirs[i])
                sfx = suffices[i]

                # Move images from src to dst based on suffix organization
                move_images_with_specific_name(src, [dst], suffixes=[sfx])

                # Increment index for next destination directory and suffix
                i += 1


def images_to_dataset(
        root_dir: str,
        src_dir: str = None,
        prefix: list[str] = None,
        suffix: list[str] = None,
        pivot: str = None,
        crop_ratio: float = 0
):
    """
    Parameters:
    root_dir (str): Root directory where the subdirectories are located.
    src_dir (str): Name of the subdirectories containing images to be moved. If None, all subdirectories are considered.
    prefix (list of str): List of prefixes of image names to be moved. Default is None (all images).
    suffix (list of str): List of suffixes of image names to be moved. Default is None (all images).
    pivot (str): Character to split the file name for renaming (optional).
    crop_ratio (float): Ratio to crop the images. Default is 0 (no cropping).

    This function searches through subdirectories (`src_dir`) under `root_dir` for images.
    It moves these images to a common dataset directory (`dataset_dir`) based on specified prefixes, suffixes, pivot,
    and crop ratio using the `move_images_with_specific_name` function.

    Raises:
    FileNotFoundError: If the "/data" folder under `root_dir` does not exist.

    Example:
    root_dir = '/path/to/root/'
    src_dir = 'source_images'
    prefix = ['img_', 'photo_']
    suffix = ['_a', '_b']
    pivot = '_'
    crop_ratio = 0.1

    images_to_dataset(root_dir, src_dir, prefix, suffix, pivot, crop_ratio)
    """
    # Define paths for data and dataset directories
    data_root = os.path.join(root_dir, 'data')
    dataset_dir = os.path.join(root_dir, 'dataset/image/full')

    # Check if the data_root exists; raise error if not
    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    # Create the dataset directory if it does not exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print('\nMoving images to dataset folder...')
    for dir_path, dir_names, file_names in tqdm(os.walk(data_root)):
        for dir_name in dir_names:
            # Process subdirectories matching src_dir or all if src_dir is None
            if src_dir is None or src_dir in dir_name:
                subdirectory_src = os.path.join(dir_path, dir_name)
                move_images_with_specific_name(subdirectory_src, [dataset_dir], prefix, suffix, pivot, crop_ratio)


def sensor_to_dataset(
        root_dir: str,
        sensor_file: str,
        csv_file: str
):
    """
    Parameters:
    root_dir (str): Root directory where the sensor data files are located.
    sensor_file (str): Full name of the sensor data file to search and merge.
    csv_file (str): Name of the CSV file to save the merged sensor data.

    This function searches through `root_dir` for files named `sensor_file` containing sensor data.
    It merges these files into a single CSV dataset saved as `csv_file` in the '/dataset/sensor' directory under `root_dir`.

    Raises:
    FileNotFoundError: If the "/data" folder under `root_dir` does not exist.

    Example:
    root_dir = '/path/to/root/'
    sensor_file = 'sensor_data.txt'
    csv_file = 'merged_sensor_data.csv'

    sensor_to_dataset(root_dir, sensor_file, csv_file)
    """
    data_root = os.path.join(root_dir, 'data')
    dataset_dir = os.path.join(root_dir, 'dataset/sensor')
    output_sensor = os.path.join(dataset_dir, sensor_file)
    csv_sensor = os.path.join(dataset_dir, csv_file)

    # Create the dataset directory if it does not exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Check if the data_root exists; raise error if not
    if not os.path.exists(data_root):
        raise FileNotFoundError('ERROR: Put the raw data in the "/data" folder under the root directory!')

    # List all entries in the directory
    print('\nMerging sensor files...')
    for path, dirs, files in tqdm(os.walk(data_root)):
        for file in files:
            if file == sensor_file:
                input_sensor = os.path.join(path, file)
                concatenate_files(input_sensor, output_sensor)
                if sensor_file.endswith('.txt'):
                    convert_txt_to_csv(output_sensor, csv_sensor)

    # Read the merged CSV file into a DataFrame
    df = pd.read_csv(csv_sensor)

    # Replace small numbers in the DataFrame with zero
    df = replace_small_numbers_in_df(df)

    # Remove rows where any cell value is equal to the column name
    column_names = df.columns
    for column in column_names:
        df = df[df[column] != column]

    # Save the updated DataFrame to the CSV file
    df.to_csv(csv_sensor, index=False)


def convert_txt_to_csv(
        input_file: str,
        output_file: str,
):
    """
    Parameters:
    input_file (str): Path to the input text file to convert.
    output_file (str): Path to save the converted CSV file.

    This function reads the contents of `input_file`, which is assumed to be in text format,
    and writes each line into `output_file` as CSV format. The function ensures that each line
    from the text file is written directly into the CSV file without any additional processing.

    Example:
    input_file = '/path/to/input.txt'
    output_file = '/path/to/output.csv'

    convert_txt_to_csv(input_file, output_file)
    """
    with open(input_file, 'r') as infile:
        # Read lines from the text file
        lines = infile.readlines()

    with open(output_file, "w", newline='') as csvfile:
        for line in lines:
            # Write each line from the text file to the CSV file
            csvfile.write(line)


def add_char_before_position(
        string: str,
        char: str,
        position: int
) -> str:
    """
    Parameters:
    string (str): The original string where characters will be added.
    char (str): The characters to add into the string.
    position (int): The position in the string where characters will be inserted.

    Returns:
    str: The modified string with characters inserted at the specified position.

    Raises:
    ValueError: If the position is out of the valid range (negative or beyond the string length).

    Example:
    string = "hello world"
    char = "_"
    position = 5

    modified_string = add_char_before_position(string, char, position)
    print(modified_string)  # Output: "hello_world world"
    """
    if position < 0 or position > len(string):
        raise ValueError("Position is out of range")

    # Insert char at the specified position in the string
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
    Parameters:
    csv_file (str): Path to the CSV file containing timestamps to standardize.
    standard_len (int): Length of the standard timestamp format.
    csv_dst (str): Path where the standardized CSV will be saved.
    csv_delimiter (str, optional): Delimiter used in the CSV file. Default is ','.
    remove (bool, optional): If True, removes specified characters from the timestamp. Default is False.
    to_remove (str, optional): Character to remove from the timestamp when `remove` is True.

    This function reads a CSV file (`csv_file`) containing timestamps, converts each timestamp to a standardized format
    of length `standard_len`, and optionally removes specified characters (`to_remove`) from the timestamps.
    The standardized CSV is then saved to `csv_dst`.

    Example:
    csv_file = '/path/to/input.csv'
    standard_len = 14
    csv_dst = '/path/to/output.csv'
    csv_delimiter = ','
    remove = True
    to_remove = '-'

    standardize_timestamp(csv_file, standard_len, csv_dst, csv_delimiter, remove, to_remove)
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

    # Save the modified DataFrame to a new CSV file
    df.to_csv(csv_dst, index=False)


def sec_to_timestamp(
        sensors_dir: str,
        sensor_file: str,
        time_feature: str = None,
        formatting: bool = False
):
    """
    Parameters:
    sensors_dir (str): Path to the directory containing the sensor measurements.
    sensor_file (str): Name of the sensor file (CSV format).
    time_feature (str, optional): Name of the feature containing Unix timestamps in the sensor file.
    formatting (bool, optional): If True, formats the Unix timestamps into a readable date string.

    If `formatting` is True, the Unix timestamps in the specified `time_feature` column are converted
    to a formatted date string using `convert_to_formatted_date` function. The resulting DataFrame
    is then saved back to a CSV file prefixed with 'timestamp_'.

    Example:
    sensors_dir = '/path/to/sensors'
    sensor_file = 'sensor_data.csv'
    time_feature = 'timestamp'
    formatting = True

    sec_to_timestamp(sensors_dir, sensor_file, time_feature, formatting)
    """
    if time_feature is None:
        return

    # Read the CSV file into a DataFrame
    sensor_df = pd.read_csv(os.path.join(sensors_dir, sensor_file))

    if formatting:
        # Apply the conversion function to the 'timestamp' column
        sensor_df['timestamp'] = pd.to_datetime(sensor_df[time_feature], unit='s')

    if time_feature != 'timestamp':
        # Remove the original timestamp column
        sensor_df.drop(columns=[time_feature], inplace=True)
        # Move the 'timestamp' column to the first position
        col_to_move = sensor_df.pop('timestamp')
        sensor_df.insert(0, 'timestamp', col_to_move)

    # Save the updated DataFrame to a new CSV file prefixed with 'timestamp_'
    output_csv = os.path.join(sensors_dir, 'timestamp_' + sensor_file)
    sensor_df.to_csv(output_csv, index=False)

    print(f"Converted timestamps saved as {output_csv}")


# Function to convert Unix timestamp to formatted date string
def convert_to_formatted_date(
        unix_microseconds: float
) -> str:
    """
    Parameters:
    unix_microseconds (float): Unix timestamp (in microseconds) to convert.

    Returns:
    str: Formatted date string in the format 'YYYY_MM_DD_HH_MM_SS_uuuuuu' (microseconds).

    Example:
    unix_time = 1625846523.123456
    formatted_date = convert_to_formatted_date(unix_time)
    print(formatted_date)  # Output: '2021_07_09_10_15_23_123456'

    This function converts a Unix timestamp (in microseconds) to a formatted date string
    representing the UTC-4 time zone.

    Note:
    - The function uses UTC as the initial timezone and then converts it to UTC-4 (America/New_York).
    - The formatted date string includes microseconds with exactly six decimal places.
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


def interpolate_frame_sensor(
        img_dir: str,
        sensors_dir: str,
        sensor_file: str,
        n: int = 1,
        interpolation: int = 1,
        timestamp: str = 'timestamp',
        sub_fixes: list = None,
        drop_column: list = None
):
    """
    Parameters:
    img_dir (str): Path to the directory containing the images.
    sensors_dir (str): Path to the directory containing the sensor measurements.
    sensor_file (str): Name of the sensor file (in CSV format) used for interpolating image timestamps.
    n (int, optional): Number of neighboring measurements to use for interpolation. Default is 1.
    interpolation (int, optional): Direction for interpolation: 1 (forward), -1 (backward), 0 (both). Default is 1.
    timestamp (str, optional): Name of the timestamp column in the sensor file. Default is 'timestamp'.
    sub_fixes (list, optional): List of characters to remove from image timestamps before comparison. Default is None.
    drop_column (list, optional): List of columns to drop from the sensor DataFrame before interpolation. Default is None.

    Raises:
    ValueError: If the sensor file is not in CSV format or if the interpolation direction is invalid.

    Example:
    img_dir = '/path/to/images'
    sensors_dir = '/path/to/sensors'
    sensor_file = 'sensor_data.csv'
    interpolate_frame_sensor(img_dir, sensors_dir, sensor_file, n=2, interpolation=1, timestamp='timestamp')

    This function iterates over images in `img_dir`, reads sensor measurements from `sensor_file` located in `sensors_dir`,
    and interpolates sensor values based on image timestamps. Interpolated values are saved to a new CSV file named
    'frame_sensor_sensor_data.csv' in `sensors_dir`.

    Notes:
    - Interpolation direction determines whether to interpolate forward, backward, or both directions.
    - Substrings specified in `sub_fixes` are removed from image timestamps before comparison.
    - Columns specified in `drop_column` are dropped from the sensor DataFrame before interpolation.
    """
    if sub_fixes is None:
        sub_fixes = ['']

    if not sensor_file.endswith('.csv'):
        raise ValueError("Convert the sensor file to CSV format!")
    if os.path.exists(sensors_dir + '/frame_sensor_' + sensor_file):
        os.remove(sensors_dir + '/frame_sensor_' + sensor_file)

    sensor_path = os.path.join(sensors_dir, sensor_file)
    sensor_df = pd.read_csv(sensor_path)
    if drop_column is not None:
        # Drop the columns
        sensor_df = sensor_df.drop(columns=drop_column)
    frame_sensor_df = pd.DataFrame(columns=sensor_df.columns.str.strip())
    frame_sensor_df.to_csv(sensors_dir + '/frame_sensor_' + sensor_file, mode='w', index=False)
    frame_sensor_row = None
    checkpoint = 0
    prev_row = None
    img_num = 0

    print('\nInterpolating measurements for the images...')
    for filename in tqdm(os.listdir(img_dir)):
        img_timestamp = filename

        if not (filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.jpg')):
            continue

        for substring in sub_fixes:
            img_timestamp = img_timestamp.replace(substring, '')

        for index, row in sensor_df.iloc[checkpoint:].iterrows():
            if prev_row is None and float(img_timestamp) < float(row[timestamp]):
                # print(f"Image number {img_num} timestamp: {img_timestamp} is before the first row timestamp at index {checkpoint}: {row['timestamp']}")
                frame_sensor_row = interpolate_row(sensor_df, img_timestamp, index, interpolation, n)
                img_num += 1
                break

            elif prev_row is not None and float(prev_row[timestamp]) <= float(img_timestamp) < float(row[timestamp]):
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
    Parameters:
    src_df (pd.DataFrame): Source dataframe containing sensor measurements.
    timestamp (str): Timestamp of the image for which measurements are interpolated.
    index (int): Index from where to start the interpolation in src_df.
    interpolation (int): Direction for interpolation: 1 (forward), -1 (backward), 0 (both).
    n (int): Number of neighboring measurements to interpolate.

    Returns:
    pd.DataFrame: DataFrame with interpolated measurements.

    Raises:
    ValueError: If the interpolation direction is invalid.

    Example:
    src_df = pd.DataFrame({'timestamp': ['2024-01-01 12:00:00', '2024-01-01 12:05:00', '2024-01-01 12:10:00'],
                           'measurement': [10, 15, 20]})
    timestamp = '2024-01-01 12:03:00'
    index = 1
    interpolation = 1
    n = 1
    interpolate_row(src_df, timestamp, index, interpolation, n)

    This function interpolates sensor measurements for an image with the specified timestamp using values from src_df.
    The interpolation is performed based on the direction specified (forward, backward, or both) and considering n neighbors.
    The resulting interpolated measurements are returned as a DataFrame.

    Notes:
    - For interpolation direction:
        - 1: Interpolate forward from the given index.
        - -1: Interpolate backward from the given index.
        - 0: Interpolate both forward and backward from the given index.
    - The interpolated measurements are calculated as the mean of neighboring measurements.

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
        min_mean: list[float] = None,
        max_std: list[float] = None,
        normalization_method: str = None
):
    """
    Parameters:
    sensors_dir (str): Path to the directory containing sensor files.
    sensor_file (str): Name of the sensor file containing measurements.
    labels_dir (str): Directory where the labels file will be saved.
    features (list): List of features to extract and use as labels from the sensor file.
    min_mean (list[float], optional): List of minimum values (for 'min-max' normalization) or mean values (for 'z-score' normalization) of the features. Defaults to None.
    max_std (list[float], optional): List of maximum values (for 'min-max' normalization) or standard deviations (for 'z-score' normalization) of the features. Defaults to None.
    normalization_method (str, optional): Method to normalize the features ('min-max' or 'z-score'). Defaults to None.

    Raises:
    ValueError: If the normalization method is invalid or if necessary values (min_mean, max_std) are not provided for 'z-score' normalization.

    """
    # Construct the path to the sensor file
    sensor_path = str(os.path.join(sensors_dir, sensor_file))

    # Read the sensor data into a DataFrame
    sensor_df = pd.read_csv(sensor_path)

    # Ensure the labels directory exists; create if not
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Extract the specified features as labels
    labels = sensor_df[features]

    # Normalize the features based on the chosen method
    if normalization_method is None:
        normalized_labels = labels
    elif normalization_method == 'min-max':
        if min_mean is None or max_std is None:
            raise ValueError("min_mean and max_std must be provided for 'min-max' normalization.")
        min_vals = np.array(min_mean)
        max_vals = np.array(max_std)
        normalized_labels = 2 * (labels - min_vals) / (max_vals - min_vals) - 1
    elif normalization_method == 'z-score':
        if min_mean is None or max_std is None:
            raise ValueError("min_mean and max_std must be provided for 'z-score' normalization.")
        mean_vals = np.array(min_mean)
        std_vals = np.array(max_std)
        normalized_labels = (labels - mean_vals) / std_vals
    else:
        raise ValueError("Invalid normalization method. Use 'min-max' or 'z-score'.")

    # Create a DataFrame for the normalized labels
    normalized_labels_df = pd.DataFrame(columns=features, data=normalized_labels)

    # Save the normalized labels to a CSV file in the labels directory
    normalized_labels_df.to_csv(os.path.join(labels_dir, 'labels.csv'), mode='w', index=False, header=True)


# Function to write image paths to CSV
def write_image_paths_to_csv(images, src_dir, csv_path):
    """
    Write the paths of images located in src_dir to a CSV file.

    Parameters:
    images (list): List of image filenames.
    src_dir (str): Directory path where the images are located.
    csv_path (str): Path to the CSV file where the image paths will be written.

    """
    # Open the CSV file in write mode and create a CSV writer
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row in the CSV file
        csv_writer.writerow(['image_path'])

        # Inform user about the process
        print(f'\nWriting image paths to {csv_path}...')

        # Iterate through each image filename and write its full path to the CSV file
        for image in tqdm(images):
            src_path = os.path.join(src_dir, image)
            csv_writer.writerow([src_path])


def train_val_test_split(
        image_dir: str,
        label_dir: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.05
):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    ----------
    image_dir : str
        Path where all the images are located.
    label_dir : str
        Path where the labels are located.
    val_ratio : float, optional
        Ratio of the validation set with respect to the training set (default is 0.2).
    test_ratio : float, optional
        Ratio of the test set with respect to the training set (default is 0.05).

    Raises:
    ------
    ValueError
        If there is a mismatch between image and label counts.

    """
    # Get the list of all images in the image directory
    image_full = os.listdir(image_dir)

    # Define paths for train, validation, and test directories
    train_dst = os.path.join(image_dir, 'train')
    val_dst = os.path.join(image_dir, 'val')
    test_dst = os.path.join(image_dir, 'test')

    # Define path for the labels CSV file
    label_file = os.path.join(label_dir, 'labels.csv')

    # Read the sensor data (labels) from the CSV file into a DataFrame
    sensor_df = pd.read_csv(label_file)

    # Create directories if they do not exist
    if not os.path.exists(train_dst):
        os.makedirs(train_dst)
    if not os.path.exists(val_dst):
        os.makedirs(val_dst)
    if not os.path.exists(test_dst):
        os.makedirs(test_dst)

    # Clear existing files in train, val, test directories
    clear_directory(train_dst)
    clear_directory(val_dst)
    clear_directory(test_dst)

    # Check for dimensions mismatch between images and labels
    if len(image_full) != len(sensor_df):
        raise ValueError("Mismatch in dimensions between images and labels in the dataset.")

    print(f'\nDataset and label dimensions: {len(image_full)}')

    # Splitting dataset into train, validation, test sets
    x_train, x_val, y_train, y_val = train_test_split(image_full, sensor_df, test_size=val_ratio, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_ratio, shuffle=False)

    # Check for dimensions mismatch between split sets and labels
    if len(x_train) != len(y_train):
        raise ValueError("Mismatch in dimensions between training images and labels.")
    if len(x_val) != len(y_val):
        raise ValueError("Mismatch in dimensions between validation images and labels.")
    if len(x_test) != len(y_test):
        raise ValueError("Mismatch in dimensions between test images and labels.")

    print(f'\nAfter the training-validation-test split with ratio {val_ratio} and {test_ratio} the dataset and label dimensions are:')
    print(f'Training set: {len(x_train)} images, Validation set: {len(x_val)} images, Test set: {len(x_test)} images')

    # CSV file paths for saving image paths
    train_csv_path = os.path.join(train_dst, 'image_path.csv')
    val_csv_path = os.path.join(val_dst, 'image_path.csv')
    test_csv_path = os.path.join(test_dst, 'image_path.csv')

    print('\nSaving train, validation, and test image paths to files...')
    # Write training set image paths to CSV
    write_image_paths_to_csv(x_train, image_dir, train_csv_path)

    # Write validation set image paths to CSV
    write_image_paths_to_csv(x_val, image_dir, val_csv_path)

    # Write test set image paths to CSV
    write_image_paths_to_csv(x_test, image_dir, test_csv_path)

    print('\nSaving train, validation, and test labels to files...')
    # Save training set labels to CSV
    y_train.to_csv(os.path.join(label_dir, 'train', 'labels.csv'), mode='w', index=False)

    # Save validation set labels to CSV
    y_val.to_csv(os.path.join(label_dir, 'val', 'labels.csv'), mode='w', index=False)

    # Save test set labels to CSV
    y_test.to_csv(os.path.join(label_dir, 'test', 'labels.csv'), mode='w', index=False)


def clear_directory(directory):
    if os.path.exists(directory):
        files = os.listdir(directory)
        for f in files:
            os.remove(os.path.join(directory, f))


def move_n_to_n_image(
        src_dir: str,
        target_dir: str,
        pivot: str = None,
        crop_ratio: list[float] = None,
):
    """
    Move images from subdirectories in src_dir to target_dir, optionally renaming them and cropping them.

    Parameters:
    ----------
    src_dir : str
        Root directory containing subdirectories with images.
    target_dir : str
        Name of the subdirectory to move images from.
    pivot : str, optional
        Optional pivot character to split and rename images based on its position.
    crop_ratio : list[float], optional
        Optional list of four floats representing crop ratios (left, top, right, bottom) to apply to each image.

    """
    # Define source and dataset directories
    data_root = os.path.join(src_dir, 'data')
    dataset_dir = os.path.join(src_dir, 'dataset', 'image', 'full')

    i = 1
    # Traverse all directories and subdirectories in data_root
    for dir_path, dir_names, file_names in tqdm(os.walk(data_root)):
        for dir_name in dir_names:
            if dir_name == target_dir:
                # Construct source and destination paths
                subdirectory_src = os.path.join(dir_path, dir_name)
                subdirectory_dst = os.path.join(dataset_dir, str(i))

                # Create destination directory if it doesn't exist
                if not os.path.exists(subdirectory_dst):
                    os.makedirs(subdirectory_dst)

                # Iterate over all files in the source directory
                for file_name in os.listdir(subdirectory_src):
                    src_file = os.path.join(subdirectory_src, file_name)
                    new_name = file_name

                    # Optionally rename the file based on pivot character
                    if pivot is not None:
                        pos = new_name.find(pivot)
                        new_name = new_name[pos + 1:]

                    # Construct destination file path
                    dst_file = os.path.join(subdirectory_dst, new_name)

                    # Open the image using PIL
                    with Image.open(src_file) as img:
                        # Optionally crop the image if crop_ratio is provided
                        if crop_ratio is not None:
                            width, height = img.size
                            left = int(width * crop_ratio[0])
                            top = int(height * crop_ratio[1])
                            right = int(width * (1 - crop_ratio[2]))
                            bottom = int(height * (1 - crop_ratio[3]))
                            img = img.crop((left, top, right, bottom))

                        # Save the image to the destination directory
                        img.save(dst_file)

                    # Print progress message
                    print(f'\rWrote image {new_name}', end='')

                i += 1


def move_n_to_n_sensor(
        src_dir: str,
        src_sensor: str,
        dst_sensor: str
):
    """
    Move sensor data files from subdirectories in src_dir to dataset/sensor directories, converting to CSV if necessary.

    Parameters:
    ----------
    src_dir : str
        Root directory containing subdirectories with sensor data files.
    src_sensor : str
        Name of the sensor data file to be moved.
    dst_sensor : str
        Destination name of the sensor data file in the dataset/sensor directory.

    """
    # Define source and dataset directories
    data_root = os.path.join(src_dir, 'data')
    dataset_dir = os.path.join(src_dir, 'dataset', 'sensor')

    i = 1
    # Traverse all directories and subdirectories in data_root
    for path, dirs, files in tqdm(os.walk(data_root)):
        for file in files:
            if file == src_sensor:
                # Construct source and destination paths
                input_sensor = os.path.join(path, file)
                output_sensor = os.path.join(dataset_dir, str(i), dst_sensor)

                # Create destination directory if it doesn't exist
                if not os.path.exists(os.path.join(dataset_dir, str(i))):
                    os.makedirs(os.path.join(dataset_dir, str(i)))

                # Convert .txt to .csv if input_sensor is a text file
                if input_sensor.endswith('.txt'):
                    convert_txt_to_csv(input_sensor, output_sensor)
                elif input_sensor.endswith('.csv'):
                    # Copy .csv file directly to destination
                    shutil.copy(input_sensor, output_sensor)

                # Read, process, and save CSV file
                df = pd.read_csv(output_sensor)
                df = replace_small_numbers_in_df(df)
                df.to_csv(output_sensor, index=False)

                i += 1


def traverse_to_csv(
        src_dir: str,
):
    """
    Traverse subdirectories in src_dir containing images and write their paths to CSV files.

    Parameters:
    ----------
    src_dir : str
        Root directory containing subdirectories with images.

    """
    dataset_dir = src_dir

    # Traverse each subdirectory in dataset_dir
    for sub_dir in os.listdir(dataset_dir):
        path_image = os.path.join(dataset_dir, sub_dir)

        # List all files in the subdirectory
        image_traverse = os.listdir(path_image)

        # Filter files by the desired image file types (png, jpeg, jpg)
        image_traverse = [file for file in image_traverse if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg')]

        # Write image paths to CSV file in the same subdirectory
        write_image_paths_to_csv(image_traverse, path_image, os.path.join(path_image, 'image_path.csv'))


def repeat_sec_to_timestamp(
        sensors_dir: str,
        sensor_file: str,
        time_feature: str = None,
        formatting: bool = False
):
    """
    Repeat the conversion of Unix timestamps to formatted date strings for each subdirectory in sensors_dir.

    Parameters:
    ----------
    sensors_dir : str
        Path to the directory containing subdirectories with sensor measurements.

    sensor_file : str
        Name of the sensor file within each subdirectory to convert timestamps.

    time_feature : str, optional
        Feature containing the Unix timestamps in the sensor file.

    formatting : bool, optional
        Boolean indicating if the formatted date string conversion should be applied.

    """
    # Iterate through each subdirectory in sensors_dir
    for sub_dir in os.listdir(sensors_dir):
        path_sensor = os.path.join(sensors_dir, sub_dir)

        # Call sec_to_timestamp function for each subdirectory
        sec_to_timestamp(path_sensor, sensor_file, time_feature, formatting)


def repeat_interpolation(
        img_dir: str,
        sensors_dir: str,
        sensor_file: str,
        n: int = 1,
        interpolation: int = 1,
        timestamp: str = 'timestamp',
        sub_fixes: list = None,
        drop_column: list = None
):
    """
    Repeat the interpolation of sensor measurements for each subdirectory in img_dir.

    Parameters:
    ----------
    img_dir : str
        Path to the directory containing subdirectories with images.

    sensors_dir : str
        Path to the directory containing subdirectories with sensor measurements.

    sensor_file : str
        Name of the sensor file within each subdirectory in sensors_dir used for interpolation.

    n : int, optional
        Number of neighboring measurements to interpolate. Default is 1.

    interpolation : int, optional
        Direction for interpolation: 1 for forward, -1 for backward, 0 for both. Default is 1.

    timestamp : str, optional
        Name of the feature in sensor_file containing timestamps. Default is 'timestamp'.

    sub_fixes : list, optional
        List of characters to remove from image file names before matching with sensor timestamps. Default is None.

    drop_column : list, optional
        List of column names to drop from the sensor DataFrame before interpolation. Default is None.

    """
    # Iterate through each subdirectory in img_dir
    for sub_dir in os.listdir(img_dir):
        path_image = os.path.join(img_dir, sub_dir)
        path_sensor = os.path.join(sensors_dir, sub_dir)

        # Call interpolate_frame_sensor function for each subdirectory
        interpolate_frame_sensor(path_image, path_sensor, sensor_file, n, interpolation, timestamp, sub_fixes, drop_column)


def repeat_get_labels(
        sensors_dir: str,
        sensor_file: str,
        labels_dir: str,
        features: list[str],
        min_vals: list[float] = None,
        max_vals: list[float] = None,
        normalization_method: str = None
):
    """
    Iterate through subdirectories in `sensors_dir`, extract labels from sensor files, and save them to corresponding directories in `labels_dir`.

    Parameters:
    ----------
    sensors_dir : str
        Path to the directory containing subdirectories with sensor measurements.

    sensor_file : str
        Name of the sensor file within each subdirectory in `sensors_dir` containing measurements.

    labels_dir : str
        Path to the directory where labels files will be saved.

    features : list[str]
        List of features (columns) from `sensor_file` to extract as labels.

    min_vals : list[float], optional
        List of minimum values for normalization. Required if `normalization_method` is 'min-max' or 'z-score'. Default is None.

    max_vals : list[float], optional
        List of maximum values for normalization. Required if `normalization_method` is 'min-max'. Default is None.

    normalization_method : str, optional
        Method to normalize the features ('min-max' or 'z-score'). Default is None (no normalization).

    """
    # Iterate through each subdirectory in sensors_dir
    for sub_dir in os.listdir(sensors_dir):
        path_sensor = os.path.join(sensors_dir, sub_dir)

        # Check if the subdirectory is a directory
        if os.path.isdir(path_sensor):
            path_label = os.path.join(labels_dir, sub_dir)

            # Call get_labels function to extract labels from sensor data
            get_labels(path_sensor, sensor_file, path_label, features, min_vals, max_vals, normalization_method)


def repeat_train_val_test_split(
        image_dir: str,
        label_dir: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.05
):
    """
    Iterate through subdirectories in `image_dir` and `label_dir`, split each dataset into training, validation, and test sets, and save the splits.

    Parameters:
    ----------
    image_dir : str
        Path to the directory where all the images are located. Each subdirectory will be processed separately.

    label_dir : str
        Path to the directory where the labels are located. Corresponding subdirectories will be used for labels.

    val_ratio : float, optional
        Ratio of the validation set with respect to the training set. Default is 0.2 (20%).

    test_ratio : float, optional
        Ratio of the test set with respect to the training set. Default is 0.05 (5%).

    Raises:
    ------
    ValueError
        If there is a mismatch between image and label counts within any subdirectory.

    """
    # Iterate through each subdirectory in image_dir
    for sub_dir in os.listdir(image_dir):
        path_image = os.path.join(image_dir, sub_dir)
        path_label = os.path.join(label_dir, sub_dir)

        # Call train_val_test_split function to split each subdirectory's dataset into train, val, and test sets
        train_val_test_split(path_image, path_label, val_ratio, test_ratio)


def delete_folder(path):
    """
    Delete a folder and all its contents recursively.

    :param path: Path to the folder to be deleted.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        print(f"Folder not found: {path}")
    except PermissionError:
        print(f"Permission denied: {path}")