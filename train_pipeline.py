import csv
import shutil
import sys

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer, LightningDataModule
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from tqdm import tqdm
from torchvision.models import resnet18, resnet101, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import models
from torchinfo import summary
import wandb
from src import train_util


def save_model_summary(model, input_image_dimensions, filename='model_summary.txt'):
    """
    Save the model summary and trainable parameters to a text file.

    :param model: PyTorch model
    :param input_image_dimensions: Tuple representing input image dimensions
    :param filename: Name of the file to save the summary
    """
    if not os.path.exists('src/logs'):
        os.makedirs('src/logs')

    print_model_summary(model, input_image_dimensions)

    with open(os.path.join('src/logs', filename), 'w') as f:
        # Print input image dimensions
        f.write("Model summary:\n")
        f.write(f"Input image dimensions: {input_image_dimensions}\n")

        # Print model architecture
        f.write(f"{model}\n")

        # Calculate and print total trainable parameters
        total_params = 0
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
        f.write(f"Total trainable parameters: {total_params}\n")


def print_model_summary(model, input_image_dimensions):
    print("\nModel summary:")
    print(f"Input image dimensions: {input_image_dimensions}")
    print(model)
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")


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


def wait_for_user_input():
    while True:
        user_input = input("\nContinue execution? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Continuing execution...")
            return
        elif user_input == 'n':
            print("Aborting...")
            delete_folder('src/logs')
            delete_folder('src/lightning_logs')
            sys.exit()
        else:
            print("Invalid input! Please enter 'y' to continue or 'n' to abort.")


def build_trunk(input_dim, out_channels, num_blocks, batch_norm):
    layers = []
    in_ch, in_h, in_w = input_dim
    out_ch = out_channels

    for _ in range(num_blocks):
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        in_ch = out_ch
        in_h //= 2
        in_w //= 2
        out_ch *= 2

    return nn.Sequential(*layers), in_ch, in_h, in_w


def build_linear_trunk(input_dim, hidden_dim, num_layers, batch_norm, activation_fn, dropout_rate):
    layers = []
    in_features = input_dim

    for _ in range(num_layers):
        layers.append(nn.Linear(in_features, hidden_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        if activation_fn is not None:
            layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        in_features = hidden_dim

    return nn.Sequential(*layers), hidden_dim


def assign_loss(loss='mse'):
    # Condition to choose loss function based on input_variable
    if loss == 'mse':
        loss_function = nn.MSELoss()
    elif loss == 'mae':
        loss_function = nn.MSELoss()
    elif loss == 'huber':
        loss_function = nn.HuberLoss()
    else:
        raise ValueError("Invalid input_variable. Choose 'mse', 'mae', or 'huber'.")

    return loss_function


class LogEpochLossCallback(pl.Callback):
    """
    Callback to log epoch loss to a CSV file.

    Attributes:
        filename (str): Path to the CSV file to store the logs.
    """

    def __init__(self, filename, out_features):
        """
        Args:
            filename (str): The name of the CSV file where losses will be logged.
        """
        self.filename = filename
        self.out_features = out_features

        # Create and write the header to the CSV file
        if not os.path.exists('src/logs'):
            os.makedirs('src/logs')

        headers = ['epoch'] + ['train_loss'] + [f'train_loss_label_{str(i)}' for i in range(0, self.out_features)] + ['val_loss'] + [f'val_loss_label_{str(i)}' for i in range(0, self.out_features)]
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.

        Args:
            trainer (pl.Trainer): The trainer object.
            pl_module (pl.LightningModule): The Lightning module being trained.
        """
        epoch = trainer.current_epoch
        train_losses_values = [float(trainer.callback_metrics.get(f'train_loss'))] + [float(trainer.callback_metrics.get(f'train_loss_label_{str(i)}')) if trainer.callback_metrics.get(f'train_loss_label_{str(i)}') is not None else None for i in range(0, self.out_features)]
        val_losses_values = [float(trainer.callback_metrics.get(f'val_loss'))] + [float(trainer.callback_metrics.get(f'val_loss_label_{str(i)}')) if trainer.callback_metrics.get(f'val_loss_label_{str(i)}') is not None else None for i in range(0, self.out_features)]

        # Print training and validation losses
        print(f'\nEpoch {epoch} - TRAINING LOSSES: {train_losses_values[0]}    VALIDATION LOSSES: {val_losses_values[0]}\n')

        # Write the losses to the CSV file
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch] + train_losses_values + val_losses_values)

        # Log the losses to wandb
        wandb.log({'epoch': epoch, 'train_loss': train_losses_values[0], 'val_loss': val_losses_values[0]})
        for i in range(self.out_features):
            wandb.log({f'train_loss_label_{i}': train_losses_values[i+1]})
            wandb.log({f'val_loss_label_{i}': val_losses_values[i+1]})


class DelayedStartEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set start_epoch to None or 0 for no delay
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_validation_end(trainer, pl_module)


class RoverImageDataset(Dataset):
    """
    Custom Dataset for loading rover images and their corresponding labels.

    Args:
        image_path (str): Directory with all the images.
        label_dir (str): Directory with labels.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, image_path, label_dir, seq_len=5, stride=None, channel='L', transform=None):
        self.image_path = os.path.join(image_path, 'image_path.csv')
        self.labels_file = os.path.join(label_dir, 'labels.csv')
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.channel = channel
        self.transform = transform
        self.image_file = self.load_image_paths()
        self.labels = self.load_labels()

        # Debugging statement
        print(f"Loaded {len(self.image_file)} images and {len(self.labels)} labels with dim {len(self.labels[0])}.")

    def load_image_paths(self):
        """
        Load image paths from the CSV file.

        Returns:
            list: List of image paths.
        """
        df = pd.read_csv(self.image_path)
        path_list = df['image_path'].tolist()
        return path_list

    def load_labels(self):
        """
        Load labels from the CSV file.

        Returns:
            list: List of labels.
        """
        df = pd.read_csv(self.labels_file)
        labels = df.values.astype(float).tolist()
        return labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return (len(self.image_file) - self.seq_len) // self.stride + 1  # Ensure the length accounts for sequence length to avoid out of range errors

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        start_idx = idx * self.stride
        images = []
        labels = []
        for i in range(start_idx, start_idx + self.seq_len):
            img_name = self.image_file[i]
            image = Image.open(img_name).convert(self.channel)

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)
            labels.append(torch.tensor(self.labels[i], dtype=torch.float32))

        images = torch.stack(images)  # Stack images to create a sequence
        labels = torch.stack(labels)  # Use the label of the last image in the sequence

        return images, labels


class RoverDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling data loading.

    Args:
        data_dir (str): Directory with all the images.
        label_dir (str): Directory with labels.
        size (callable, optional): Optional size for transform.
        batch_size (int, optional): Batch size for data loading.
    """

    def __init__(self, data_dir: str, label_dir: str, seq_len: int = 5, stride: int = None, size=(1, 256, 256), channel: str = 'L', augment=False, batch_size: int = 32, num_workers: int = 0, split_by_traverse: bool = True, val_traverse: int = None):
        super().__init__()
        self.test = None
        self.val = None
        self.train = None
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.size = size
        self.channel = channel
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.split_by_traverse = split_by_traverse
        self.data_list = os.listdir(data_dir)
        self.label_list = os.listdir(label_dir)
        self.max_data_range = len(self.data_list)
        if val_traverse is None or val_traverse > self.max_data_range or val_traverse <= 0:
            self.val_traverse = self.max_data_range
        else:
            self.val_traverse = val_traverse

        self.transform = transforms.Compose([
            transforms.Resize((self.size[1], self.size[2])),
            transforms.ToTensor()
        ])

        self.augmentation = transforms.Compose([
            transforms.Resize((self.size[1], self.size[2])),
            transforms.RandomApply(self.random_augmentation(), p=0.5),  # Apply random augmentation with 50% probability
            transforms.ToTensor()
        ])

    def random_augmentation(self):
        # Define a list of augmentation transforms, each with its own probability
        augmentation_transforms = [
            transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),  # Small rotations to simulate camera angle changes
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.5),  # Lighting changes
            transforms.RandomApply([transforms.RandomResizedCrop((self.size[1], self.size[2]), scale=(0.8, 1.0))], p=0.5),  # Crop and resize to simulate distance changes
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),  # Blur to simulate focus issues
        ]

        return augmentation_transforms

    def create_datasets(self, stage):
        """
        Create datasets for training, validation, and testing based on the stage.

        Args:
            stage (str): Current stage - 'fit' or 'test'.

        Returns:
            tuple: Datasets for training, validation, and testing.
            float: Probability of augmentation
        """

        # TODO: Check how to call the DataModule using the stages
        if stage == 'fit':
            train_transform = self.augmentation if self.augment else self.transform
            for i in range(1, self.max_data_range + 1):
                if self.split_by_traverse:
                    try:
                        if i % self.val_traverse != 0:
                            train_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i])), os.path.join(self.label_dir, str(self.label_list[i])), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=train_transform)
                            self.train_datasets.append(train_dataset)
                            print(f"Dataset traverse {i} for training.")
                        else:
                            val_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i])), os.path.join(self.label_dir, str(self.label_list[i])), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=self.transform)
                            self.val_datasets.append(val_dataset)
                            print(f"Dataset traverse {i} for validation.")
                    except Exception as e:
                        print(f"Error loading dataset train {i}, val {i}: {e}")
                else:
                    try:
                        train_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i]), 'train'), os.path.join(self.label_dir, str(self.label_list[i]), 'train'), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=train_transform)
                        self.train_datasets.append(train_dataset)
                        val_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i]), 'val'), os.path.join(self.label_dir, str(self.label_list[i]), 'val'), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=self.transform)
                        self.val_datasets.append(val_dataset)
                        print(f"Dataset train {i}, val {i} loaded successfully.")
                    except Exception as e:
                        print(f"Error loading dataset train {i}, val {i}: {e}")

        # elif stage == 'test':
        #     for i in range(1, self.max_data_range + 1):
        #         if self.split_by_traverse:
        #         # find a dataset to use as test split
        #         else:
        #             try:
        #                 test_dataset = RoverImageDataset(os.path.join(self.data_dir, str(i), 'test'), os.path.join(self.label_dir, str(i), 'test'), seq_len=self.seq_len, transform=self.transform)
        #                 self.test_datasets.append(test_dataset)
        #                 print(f"Dataset train{i}, val{i} loaded successfully.")
        #             except Exception as e:
        #                 print(f"Error loading dataset train {i}, val {i}: {e}")

        else:
            print('\nOne or more dataset where not initialized correctly.')
            exit()

    def setup(self, stage: str):
        """
        Set up the datasets based on the stage.

        Args:
            stage (str): Current stage - 'fit' or 'test'.
        """
        # TODO: Check how to call the DataModule using the stages
        self.create_datasets(stage)

        if stage == "fit":
            self.train = ConcatDataset(self.train_datasets)
            self.val = ConcatDataset(self.val_datasets)
            print(f"Loaded {len(self.train)} and {len(self.val)} samples for training and validation.")
            print("Training and validation datasets concatenated successfully.")
        elif stage == "test":
            self.test = ConcatDataset(self.test_datasets)
            print("Test datasets concatenated successfully.")

    def train_dataloader(self):
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        """
        Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for the test set.
        """
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for regression.
    """

    def __init__(self, input_dim=(1, 256, 256), out_channel_base=4, num_blocks=2, last_timestep=False, batch_norm=False, fc_layers=1, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=1, rnn_output_size=32, dropout=0.0, rnn_dropout=0.0, backbone='scratch', out_features=2):
        super(CNNLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.out_channel_base = out_channel_base
        self.num_blocks = num_blocks
        self.last_timestep = last_timestep
        self.fc_layers = fc_layers
        self.fc_activation = fc_activation
        self.fc_output_size = fc_output_size
        self.lstm_layers = lstm_layers
        self.rnn_output_size = rnn_output_size
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.batch_norm = batch_norm
        self.backbone = backbone
        self.out_features = out_features

        # CNN layers
        if self.backbone == 'scratch':
            self.cnn, self.out_channels, self.out_height, self.out_width = build_trunk(self.input_dim, self.out_channel_base, self.num_blocks, self.batch_norm)
        elif self.backbone == 'resnet18':
            # Load pretrained ResNet-18 without classification layer
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            for param in resnet.parameters():
                param.requires_grad = False

            self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove last layers (avgpool and fc)

            self.out_channels = 512  # ResNet-18 outputs 512 channels
            self.out_height = self.input_dim[1] // 32  # Adjust based on input dimensions and ResNet-18 architecture
            self.out_width = self.input_dim[2] // 32  # Adjust based on input dimensions and ResNet-18 architecture
        elif self.backbone == 'vgg16':
            # Load pretrained VGG16 without classification layer
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            for param in vgg.parameters():
                param.requires_grad = False

            self.cnn = nn.Sequential(*list(vgg.features))  # Use the feature extractor part of VGG16

            self.out_channels = 512  # VGG16 outputs 512 channels from the feature extractor
            self.out_height = self.input_dim[1] // 32  # Adjust based on input dimensions and VGG16 architecture
            self.out_width = self.input_dim[2] // 32  # Adjust based on input dimensions and VGG16 architecture

        if self.dropout > 0.0:
            self.dropout_cnn = nn.Dropout(self.dropout)

        # Fully connected layer
        self.fc, _ = build_linear_trunk(self.out_channels * self.out_height * self.out_width, self.fc_output_size, self.fc_layers, self.batch_norm, self.fc_activation, self.dropout)

        # LSTM layers
        if self.lstm_layers > 1:
            self.lstm = nn.LSTM(self.fc_output_size, hidden_size=self.rnn_output_size, num_layers=self.lstm_layers, dropout=self.rnn_dropout)
        else:
            self.lstm = nn.LSTM(self.fc_output_size, hidden_size=self.rnn_output_size, num_layers=self.lstm_layers)

        if self.rnn_dropout > 0.0:
            self.dropout_lstm = nn.Dropout(self.rnn_dropout)

        self.output = nn.Linear(self.rnn_output_size, self.out_features)

    # noinspection PyListCreation
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, seq_len, c, h, w = x.size()

        # Reshape for CNN input
        cnn_in = x.view(batch_size * seq_len, c, h, w)

        # Apply CNN layers
        cnn_out = self.cnn(cnn_in)
        if self.dropout > 0.0:
            cnn_out = self.dropout_cnn(cnn_out)

        # Reshape for FC/LSTM input
        fc_in = cnn_out.view(batch_size, seq_len, -1)

        fc_out = self.fc(fc_in)

        # Apply LSTM layers
        lstm_out, _ = self.lstm(fc_out)
        if self.rnn_dropout > 0.0:
            lstm_out = self.dropout_lstm(lstm_out)

        # Apply fully connected layer
        if self.last_timestep:
            output = self.output(lstm_out[:, -1, :])
        else:
            output = self.output(lstm_out)

        return output


# PyTorch Lightning Module
class RoverRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for rover regression.
    """

    def __init__(self, loss='mse', learning_rate=1e-4, weight_decay=1e-5, input_dim=(1, 256, 256), out_channel_base=4, num_blocks=3, last_timestep=False, batch_norm=False, fc_layers=1, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=1, rnn_output_size=32, dropout=0.0, rnn_dropout=0.0,
                 backbone='scratch', out_features=2):
        super(RoverRegressor, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.last_timestep = last_timestep
        self.model = CNNLSTMModel(input_dim, out_channel_base, num_blocks, last_timestep, batch_norm, fc_layers, fc_activation, fc_output_size, lstm_layers, rnn_output_size, dropout, rnn_dropout, backbone, out_features)
        self.loss_fn = assign_loss(loss)
        self.out_features = out_features

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary with loss.
        """
        x, y = batch
        if self.last_timestep:
            y = y[:, -1, :].view(-1, self.out_features)
        else:
            y = y.view(-1, self.out_features)
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, self.out_features).float()
        # Compute loss for each label
        for i in range(self.out_features):
            loss = self.loss_fn(y_hat[:, i], y[:, i])
            self.log(f'train_loss_label_{str(i)}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Combine the losses
        combined_loss = self.loss_fn(y_hat, y)

        self.log('train_loss', combined_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': combined_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary with loss.
        """
        x, y = batch
        if self.last_timestep:
            y = y[:, -1, :].view(-1, self.out_features)
        else:
            y = y.view(-1, self.out_features)
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, self.out_features).float()
        # Compute loss for each label
        for i in range(self.out_features):
            loss = self.loss_fn(y_hat[:, i], y[:, i])
            self.log(f'val_loss_label_{str(i)}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Combine the losses
        combined_loss = self.loss_fn(y_hat, y)

        self.log('val_loss', combined_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': combined_loss}

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: Optimizer object.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=self.learning_rate * 1e-2),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


def main():
    """
    Main function to run the training process.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    images = 'D:/Dataset/devens_snowy_fixed'
    labels = 'D:/Dataset/devens_snowy_fixed'

    num_epoch = 20
    early_stopping_epoch = 50
    early_stopping_patience = 20

    # Initialize a new run
    wandb.init(project='Rover Autopilot')

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log_callback = LogEpochLossCallback('src/logs/epoch_loss.csv', 4)
    early_stop_callback = DelayedStartEarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=early_stopping_patience,
        verbose=False,
        mode='min',
        start_epoch=early_stopping_epoch
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        filename='best-checkpoint',  # Filename for the best model
        save_top_k=1,  # Save only the top 1 model
        mode='min'  # Mode can be 'min' or 'max' based on the metric
    )

    x = (3, 256, 128)
    data_module = RoverDataModule(images, labels, seq_len=12, stride=9, size=x, channel='RGB', augment=True, batch_size=64, num_workers=7, split_by_traverse=True, val_traverse=5)

    # Training
    model = RoverRegressor(loss='mse', learning_rate=1e-3, weight_decay=1e-4, input_dim=x, out_channel_base=8, num_blocks=4, last_timestep=True, batch_norm=False, fc_layers=3, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=3, rnn_output_size=128, dropout=0.2, rnn_dropout=0,
                           backbone='resnet18', out_features=4)
    # model.to(dev)
    # Print the model architecture using torchsummary
    train_util.save_model_summary(model, x)
    print(f'\nTraining for maximum {num_epoch} epochs. Early stopping with patience {early_stopping_patience} starting from epoch {early_stopping_epoch}')

    wait_for_user_input()

    trainer = pl.Trainer(
        max_epochs=num_epoch,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[log_callback, early_stop_callback, checkpoint_callback]
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
