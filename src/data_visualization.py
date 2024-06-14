import os

import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate


def line_plot(
        dataframe_path: str,
        features: list
):
    """
    Generates a line plot for the specified features in the given dataframe.

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

    standard_metrics(df)

    # Plotting
    plt.figure(figsize=(10, 6))
    for feature in features:
        plt.plot(df.index, df[feature], label=f'{feature}')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Line Plot of Multiple Losses')
    plt.legend()
    plt.grid(True)
    plt.show()


def multi_line_plot(
        dataframe_path: str,
        features: list
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

    for column in df.columns:
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
