import os

import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import cv2
import os

from tqdm import tqdm


def line_plot(
        dataframe_path: str,
        directory: str = '../../logs',
        filename: str = 'loss_plot.png'
):
    """
    Generates a line plot for the specified features in the given dataframe.

    :param dataframe_path: Path to the CSV file containing the dataframe.
    :param directory: Directory where to save the plot.
    :param filename: Name of the file to save the metrics.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        df = pd.read_csv(dataframe_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {dataframe_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {dataframe_path}")
    except pd.errors.ParserError:
        raise ValueError(f"File parsing error: {dataframe_path}")

    standard_metrics(df)
    standard_metrics_to_txt(df)

    # Plotting
    plt.figure(figsize=(10, 6))
    for feature in df.columns[1:]:
        plt.plot(df.index, df[feature], label=f'{feature}')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Line Plot of Multiple Losses')
    plt.legend()
    plt.grid(True)
    # Save plot to PNG file
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    plt.show()


def multi_line_plot(
        dataframe_path: str,
        features: list,
):
    """
    Generates multiple line plots for the specified features in the given dataframe,
    each plot in its own subplot.

    :param dataframe_path: Path to the CSV file containing the dataframe.
    :param features: List of feature names to plot.
    """
    try:
        df = pd.read_csv(dataframe_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {dataframe_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {dataframe_path}")
    except pd.errors.ParserError:
        raise ValueError(f"File parsing error: {dataframe_path}")

    if not all(feature in df.columns for feature in features):
        raise ValueError("Some features are not present in the dataframe")

    standard_metrics(df[features])
    # standard_metrics_to_txt(df[features])

    num_features = len(features)

    # Create subplots
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))

    # Plot each feature in a separate subplot
    for i, feature in enumerate(features):
        axes[i].plot(df.index, df[feature], label=feature)
        axes[i].set_ylabel(f'{feature} Value')
        axes[i].set_title(f'Line Plot of {feature}')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def standard_metrics(
        df: pd.DataFrame
):
    """
    Calculates and prints standard metrics (min, max, mean, std dev) for each column in the dataframe.

    :param df: DataFrame for which to calculate the metrics.
    """
    metrics = {
        "Metric": ["Min", "Idx Min", "Max", "Idx Max", "Mean", "Std Dev"],
    }

    for column in df.columns[1:]:
        metrics[column] = [
            df[column].min(),
            str(int(df[column].idxmin())),
            df[column].max(),
            str(int(df[column].idxmax())),
            df[column].mean(),
            df[column].std()
        ]

    metrics_df = pd.DataFrame(metrics)

    print(tabulate(metrics_df, headers='keys', tablefmt='pretty', showindex=False))


def standard_metrics_to_txt(
        df: pd.DataFrame,
        directory: str = '../../logs',
        filename: str = 'metric_summary.txt'
):
    """
    Calculates and saves standard metrics (min, max, mean, std dev) for each column in the dataframe to a text file.

    :param df: DataFrame for which to calculate the metrics.
    :param directory: Directory where the file should be saved.
    :param filename: Name of the file to save the metrics.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Calculate metrics
    metrics = {
        "Metric": ["Min", "Idx Min", "Max", "Idx Max", "Mean", "Std Dev"],
    }

    for column in df.columns[1:]:
        metrics[column] = [
            df[column].min(),
            str(int(df[column].idxmin())),
            df[column].max(),
            str(int(df[column].idxmax())),
            df[column].mean(),
            df[column].std()
        ]

    metrics_df = pd.DataFrame(metrics)

    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Write metrics to file
    with open(file_path, 'w') as f:
        f.write(tabulate(metrics_df, headers='keys', tablefmt='pretty', showindex=False))


# Function to read images from directory
def load_images_from_folder(folder):
    images = []
    for filename in tqdm(sorted(os.listdir(folder))):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# Function to generate video from images with labels
def generate_video(images_folder, csv_file, output_video):
    # Load images
    images = load_images_from_folder(images_folder)

    # Read CSV file
    labels_df = pd.read_csv(csv_file)
    labels = labels_df.values  # Assuming the CSV has one label per row, each corresponding to an image/frame

    # Check if number of images and labels match
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must be the same.")

    # Get image size
    height, width, _ = images[0].shape

    # Define video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
    fps = 25  # Adjust the frames per second as needed
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write images with labels to video
    for i in tqdm(range(len(images))):
        img = images[i]

        label = str(labels[i][0]) + '        ' + str(labels[i][1])

        # Add label to the image
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write image to video
        out.write(img)

    # Release video writer
    out.release()
