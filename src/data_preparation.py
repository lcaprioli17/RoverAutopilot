import csv
import os
import shutil
import sys
import pandas as pd


def move_images_with_specific_name(src_dir,
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


def concatenate_files(input_file,
                      output_file
                      ):
    with open(output_file, 'a') as outfile:
        with open(input_file, 'r') as infile:
            outfile.write('\n')
            outfile.write(infile.read())


def move_images_from_root(root_dir,
                          src_dir,
                          dst_dirs,
                          suffices
                          ):
    """
    Search in a root directories for the images in src_dir to organize in dst_dirs by suffices.
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
    for path, dirs, files in os.walk(data_root):
        for subdir in dirs:
            if subdir == src_dir:
                src = os.path.join(path, src_dir)
                dst = os.path.join(path, dst_dirs[i])
                sfx = suffices[i]
                move_images_with_specific_name(src, [dst], [sfx])
                i += i


def images_to_dataset(root_dir,
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
    dataset_dir = root_dir + '/dataset/image/raw'
    if not os.path.exists(data_root):
        print('\nERROR: Put the raw data in the "/data" folder under the root directory!')
        sys.exit("\nStopping the program.")
    for dir_path, dir_names, file_names in os.walk(data_root):
        for dir_name in dir_names:
            if src_dir in dir_name:
                subdirectory_src = os.path.join(dir_path, dir_name)
                move_images_with_specific_name(subdirectory_src, [dataset_dir], [suffix])


def sensor_to_dataset(root_dir,
                      sensor_file
                      ):
    """
    Search the root directory for files with the same name containing the data of the sensors and concatenates them.
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
    for path, dirs, files in os.walk(data_root):
        for file in files:
            if file == sensor_file:
                input_sensor = os.path.join(path, file)
                concatenate_files(input_sensor, output_sensor)


def convert_txt_to_csv(input_file,
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


def add_char_before_position(string,
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


def convert_timestamp_csv(csv_file,
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
    for index, value in df['timestamp'].items():
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


def interpolate_sensor_to_frame(img_dir,
                                sensors_dir,
                                sensor_file,
                                n=1,
                                interpolation=1,
                                sub_fixes=None
                                ):
    if sub_fixes is None:
        sub_fixes = ['']
    if not sensor_file.endswith('.csv'):
        print('\nERROR: Convert the sensor file to CSV format!')
        sys.exit("\nStopping the program.")
    sensor_path = sensors_dir + '/' + sensor_file
    sensor_df = pd.read_csv(sensor_path)
    video_sensor_df = pd.DataFrame()
    checkpoint = 0
    for filename in os.listdir(img_dir):
        # if filename.endswith('.png'):
        img_timestamp = filename
        for substring in sub_fixes:
            img_timestamp = img_timestamp.replace(substring, '')
        for index, row in sensor_df.iloc[checkpoint:].iterrows():
            if img_timestamp >= row['timestamp']:
                if interpolation == -1:
                    if index - n < 0:
                        interpolated_row = sensor_df.iloc[0:index, 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                    else:
                        interpolated_row = sensor_df.iloc[index - n:index, 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                elif interpolation == 1:
                    if index + n > len(sensor_df):
                        interpolated_row = sensor_df.iloc[index:len(sensor_df), 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        print(interpolated_row)
                        # video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                    else:
                        interpolated_row = sensor_df.iloc[index:index + n, 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        print(interpolated_row)
                        # video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                elif interpolation == 0:
                    if index - n < 0:
                        interpolated_row = sensor_df.iloc[0:index + n, 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                    elif index + n > len(sensor_df):
                        interpolated_row = sensor_df.iloc[index - n:len(sensor_df), 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                    else:
                        interpolated_row = sensor_df.iloc[index - n:index + n, 1:].mean()
                        interpolated_row['timestamp'] = img_timestamp
                        video_sensor_df = pd.concat([video_sensor_df, interpolated_row], ignore_index=True)
                else:
                    print('\nERROR: Interpolation should be 1:forward, -1:backward, 0:both!')
                    sys.exit("\nStopping the program.")
                checkpoint = index
                break
    video_sensor_df.to_csv(sensors_dir + '/frame_' + sensor_file, index=False)


kbpr_root = 'D:/Dataset/Rover/KBPR/'
kbpr_txt = 'D:/Dataset/Rover/KBPR/dataset/csv/imu.txt'
kbpr_csv = 'D:/Dataset/Rover/KBPR/dataset/csv/imu.csv'

# sensor_to_dataset(kbpr_root, 'imu.txt')
# convert_txt_to_csv(kbpr_txt, kbpr_csv)
# convert_timestamp_csv(kbpr_csv, 23, 'D:/Dataset/Rover/KBPR/dataset/csv/imu_standard_timestamp.csv')

interpolate_sensor_to_frame('D:/Dataset/Rover/KBPR/dataset/image/raw', 'D:/Dataset/Rover/KBPR/dataset/csv', 'imu_standard_timestamp.csv', n=1, interpolation=1, sub_fixes=['LocCam_', '_0.png'])
