import glob

from tqdm import tqdm
import os
import shutil
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def move_images_with_specific_name(
        src_dir,
        dest_dirs,
        suffices
):
    """
    Organize images by suffices from a common directory src_dir to the directories specified in the dest_dirs list.
    The number and order of destination directories and suffices needs to be the same.

    :param src_dir: Source directory where all the images are found. The full path of the directory is needed.
    :param dest_dirs: List of destination directories to organize the images by suffices. the full path of the directories is needed.
    :param suffices: List of suffices used to organize the images in the dest_directories.
    """
    if len(dest_dirs) != len(suffices):
        print('ERROR: Number of destination directories and number of file suffices are different!')
        sys.exit("\nStopping the program.")

    for i in range(len(dest_dirs)):
        # Ensure the destination directory exists
        if not os.path.exists(dest_dirs[i]):
            os.makedirs(dest_dirs[i])

    # Iterate over all files in the source directory
    for file_name in os.listdir(src_dir):
        for i in range(len(suffices)):
            # Check if the file name contains the specific name
            if file_name.endswith(suffices[i]):
                # Construct full file paths
                src_file = os.path.join(src_dir, file_name)

                # Move the file
                shutil.move(src_file, dest_dirs[i])


def concatenate_files(
        input_file,
        output_file
):
    """
    Concatenate different files.
    It is used to concatenate list of measurements with the same name.

    :param input_file: File to be concatenated.
    :param output_file: File to concatenate.
    """
    with open(output_file, 'a') as outfile:
        with open(input_file, 'r') as infile:
            outfile.write('\n')
            outfile.write(infile.read())


def images_from_root(
        root_dir,
        src_dir,
        dst_dirs,
        suffices
):
    """
    Search in a root directory for the images in src_dir to organize in dst_dirs by suffices.

    :param root_dir: Root where the subdirectories are. The files should be in the '/data' folder.
    :param src_dir: Name of the subdirectories to search from the root.
    :param dst_dirs: List of destination directories to organize the images by suffices.
    :param suffices: List of suffices used to organize the images in the dst_dirs.
    """
    data_root = root_dir + '/data'

    if not os.path.exists(data_root):
        print('\nERROR: Put the raw data in the "/data" folder under the root directory!')
        sys.exit("\nStopping the program.")

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
        root_dir,
        src_dir,
        suffix
):
    """
    Move images from subdirectories with the same name to a common directory.

    :param root_dir: Root where the subdirectories are.
    :param src_dir: Name of the subdirectories to search from the root.
    :param suffix: Suffix of the images to move.
    """
    data_root = root_dir + '/data'
    dataset_dir = root_dir + '/dataset/image/raw/full'

    if not os.path.exists(data_root):
        print('\nERROR: Put the raw data in the "/data" folder under the root directory!')
        sys.exit("\nStopping the program.")

    print('\nMoving images to dataset folder...')
    for dir_path, dir_names, file_names in tqdm(os.walk(data_root)):
        for dir_name in dir_names:
            if src_dir in dir_name:
                subdirectory_src = os.path.join(dir_path, dir_name)
                move_images_with_specific_name(subdirectory_src, [dataset_dir], [suffix])


def sensor_to_dataset(
        root_dir,
        sensor_file
):
    """
    Search the root directory for files with the same name containing the data of the sensors and merges them.

    :param root_dir: Root where the files are.
    :param sensor_file: Full name of the file to search.
    """
    data_root = root_dir + '/data'
    dataset_dir = root_dir + '/dataset/csv'
    output_sensor = os.path.join(dataset_dir, sensor_file)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(data_root):
        print('\nERROR: Put the raw data in the "/data" folder under the root directory!')
        sys.exit("\nStopping the program.")

    # List all entries in the directory
    print('\nMerging sensor files...')
    for path, dirs, files in tqdm(os.walk(data_root)):
        for file in files:
            if file == sensor_file:
                input_sensor = os.path.join(path, file)
                concatenate_files(input_sensor, output_sensor)


def convert_txt_to_csv(
        input_file,
        output_file
):
    """
    Converts a txt file to a csv. TODO: do the opposite and specify the features name.

    :param input_file: File to convert.
    :param output_file: Converted file.
    """
    with open(input_file, 'r') as infile:
        # Read lines from the text file
        lines = infile.readlines()

    with open(output_file, "w", newline='') as csvfile:
        for line in lines:
            csvfile.write(line)


def add_char_before_position(
        string,
        char,
        position
):
    """
    Add characters at a given position in a string.

    :param string: String to modify.
    :param char: characters to add to the string.
    :param position: position where to add the characters.
    """
    if position < 0 or position > len(string):
        raise ValueError("Position is out of range")
    return string[:position] + char + string[position:]


def convert_timestamp_csv(
        csv_file,
        standard_len,
        csv_dst,
        csv_delimiter=',',
        remove=False,
        to_remove=''
):
    """
    Converts the timestamps to a standardized format filling with zeroes empty spaces. TODO: make it more general.

    :param csv_file: Path to the file to standardize.
    :param standard_len: length of the standard format.
    :param csv_dst: Path where to save the standardized csv.
    :param csv_delimiter: Delimiter of the columns in the csv to read.
    :param remove: Boolean that says if there are character to remove.
    :param to_remove: Character that needs to be removed from the timestamp.
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


def interpolate_frame_sensor(
        img_dir,
        sensors_dir,
        sensor_file,
        n=1,
        interpolation=1,
        sub_fixes=None
):
    """
    For each image in img_dir interpolates measurements from sensor_file and saves the in a file.
    At the end there will be a file containing a measurement of the specified sensor for each image.

    :param img_dir: Path to the directory containing the images.
    :param sensors_dir: Path to the directory containing the sensor measurements, used both to get the source sensor and to save interpolated values to the destination sensor.
    :param sensor_file: name of the sensor to use for interpolating image timestamps.
    :param n: number of neighboring measurements to interpolate.
    :param interpolation: Direction for interpolation: 1 forward, -1 backward, 0 both.
    :param sub_fixes: Character that needs to be removed from the timestamp.
    """
    if sub_fixes is None:
        sub_fixes = ['']

    if not sensor_file.endswith('.csv'):
        print('\nERROR: Convert the sensor file to CSV format!')
        sys.exit("\nStopping the program.")

    sensor_path = sensors_dir + '/' + sensor_file
    sensor_df = pd.read_csv(sensor_path)
    frame_sensor_df = pd.DataFrame(columns=sensor_df.columns)
    frame_sensor_df.to_csv(sensors_dir + '/frame_sensor_' + sensor_file, mode='a', index=False)
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

        frame_sensor_df = pd.DataFrame([frame_sensor_row])
        frame_sensor_df.to_csv(sensors_dir + '/frame_sensor_' + sensor_file, mode='a', index=False, header=False)


def interpolate_row(
        src_df,
        timestamp,
        index,
        interpolation,
        n
):
    """
    Interpolate the measurements for the image with the specific timestamp using values from src_df.
    Starting from index it computes the measurements for the image following the interpolation method and considering n neighbors.

    :param src_df: Source dataframe for the sensor measurements.
    :param timestamp: Timestamp of the image  for which to interpolate new measurements.
    :param index: Index from where to start the interpolation.
    :param interpolation: Direction for interpolation: 1 forward, -1 backward, 0 both.
    :param n: number of neighboring measurements to interpolate.
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
        print('\nERROR: Interpolation should be 1:forward, -1:backward, 0:both!')
        sys.exit("\nStopping the program.")

    interpolated_row.insert(0, 'timestamp', timestamp)
    # print(f'The nearest row added for image {timestamp} had index {index}. Now it has timestamp {interpolated_row.iloc[-1]}.\n')
    return interpolated_row


def get_labels(
        sensors_dir,
        sensor_file,
        features
):
    """
    Extract the labels from the sensor file.

    :param sensors_dir: Path to the directory where all the sensors are saved.
    :param sensor_file: File containing the specific sensor measurements.
    :param features: Features to extract and to use as labels.
    """
    sensor_path = sensors_dir + '/' + sensor_file
    sensor_df = pd.read_csv(sensor_path)

    labels = pd.DataFrame(columns=features, data=sensor_df[features])

    labels.to_csv(sensors_dir + '/labels.csv', mode='a', index=False, header=False)


def train_val_test_split(
        image_dir,
        label_dir,
        labels,
        val_ratio=0.2,
        test_ratio=0.1,
):
    """
    Splits the dataset in train, validation and test sets.

    :param image_dir: Path where all the images are located.
    :param label_dir: Path where the labels are located.
    :param val_ratio: Ratio of the validation set with respect to the training set.
    :param test_ratio: Ratio of the test set with respect to the training set.
    :param labels: Labels to use.
    """

    image_full = os.listdir(image_dir + '/full')
    train_dst = image_dir + '/train'
    val_dst = image_dir + '/val'
    test_dst = image_dir + '/test'
    label_file = label_dir + '/labels.csv'
    sensor_df = pd.read_csv(label_file, header=None)

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

    if len(labels) != len(sensor_df.columns):
        print('The number of passed labels and columns in the dataframe do not match.')
        exit()

    if len(image_full) == len(sensor_df):
        print(f'\nDataset and label dimensions are: {len(image_full)}')
    else:
        print('\nError in the dimensions between X and y in  dataset.')

    x_train, x_val, y_train, y_val = train_test_split(image_full, sensor_df, test_size=val_ratio, shuffle=False)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_ratio, shuffle=False)

    print(f'\nAfter the train-validation-test split with ratio {val_ratio} and {test_ratio} the dataset and label dimensions are:')
    if len(x_train) == len(y_train):
        print(len(x_train))
    else:
        print('\nError in the dimensions between X and y in training set.')
    if len(x_val) == len(y_val):
        print(len(x_val))
    else:
        print('\nError in the dimensions between X and y in validation set.')
    if len(x_test) == len(y_test):
        print(len(x_test))
    else:
        print('\nError in the dimensions between X and y in test set.')

    print('\nSaving train, validation and test labels to file...')
    y_train.to_csv(label_dir + '/train/labels.csv', mode='w', index=False, header=labels)
    y_val.to_csv(label_dir + '/val/labels.csv', mode='w', index=False, header=labels)
    y_test.to_csv(label_dir + '/test/labels.csv', mode='w', index=False, header=labels)

    print('\nMoving training set images...')
    for image in tqdm(x_train):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(train_dst, image)))

    print('\nMoving validation set images...')
    for image in tqdm(x_val):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(val_dst, image)))

    print('\nMoving test set images...')
    for image in tqdm(x_test):
        shutil.copy(str(os.path.join(image_dir, 'full', image)), str(os.path.join(test_dst, image)))

    if len(os.listdir(train_dst)) == len(y_train):
        print('\nError in the dimensions between X and y in training set.')
    if len(os.listdir(val_dst)) == len(y_val):
        print('\nError in the dimensions between X and y in validation set.')
    if len(os.listdir(test_dst)) == len(y_test):
        print('\nError in the dimensions between X and y in test set.')
