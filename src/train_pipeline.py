import csv
import shutil
import sys

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer, LightningDataModule
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def save_model_summary(model, input_image_dimensions, filename='model_summary.txt'):
    """
    Save the model summary and trainable parameters to a text file.

    :param model: PyTorch model
    :param input_image_dimensions: Tuple representing input image dimensions
    :param filename: Name of the file to save the summary
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')

    print_model_summary(model, input_image_dimensions)

    with open(os.path.join('logs', filename), 'w') as f:
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
            delete_folder('logs')
            delete_folder('lightning_logs')
            sys.exit()
        else:
            print("Invalid input! Please enter 'y' to continue or 'n' to abort.")


class LogEpochLossCallback(pl.Callback):
    """
    Callback to log epoch loss to a CSV file.

    Attributes:
        filename (str): Path to the CSV file to store the logs.
    """

    def __init__(self, filename):
        self.filename = filename
        # Create and write the header to the CSV file
        if not os.path.exists('logs'):
            os.makedirs('logs')
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss_vel(x)', 'train_loss_gyro(z)', 'train_loss', 'val_loss_vel(x)', 'val_loss_gyro(z)', 'val_loss'])

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.

        Args:
            trainer (pl.Trainer): The trainer object.
            pl_module (pl.LightningModule): The Lightning module being trained.
        """
        # Get the train loss from the trainer's logged metrics
        train_loss1 = trainer.callback_metrics.get('train_loss_vel(x)')
        train_loss2 = trainer.callback_metrics.get('train_loss_gyro(z)')
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss1 = trainer.callback_metrics.get('val_loss_vel(x)')
        val_loss2 = trainer.callback_metrics.get('val_loss_gyro(z)')
        val_loss = trainer.callback_metrics.get('val_loss')

        # Convert to float if not None
        train_loss1 = float(train_loss1) if train_loss1 is not None else None
        train_loss2 = float(train_loss2) if train_loss2 is not None else None
        train_loss = float(train_loss) if train_loss is not None else None
        val_loss1 = float(val_loss1) if val_loss1 is not None else None
        val_loss2 = float(val_loss2) if val_loss2 is not None else None
        val_loss = float(val_loss) if val_loss is not None else None

        # Write the loss to the CSV file
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trainer.current_epoch, train_loss1, train_loss2, train_loss, val_loss1, val_loss2, val_loss])


class RoverImageDataset(Dataset):
    """
    Custom Dataset for loading rover images and their corresponding labels.

    Args:
        image_dir (str): Directory with all the images.
        label_dir (str): Directory with labels.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, image_dir, label_dir, seq_len=5, transform=None):
        self.image_dir = image_dir
        self.labels_file = os.path.join(label_dir, 'labels.csv')
        self.seq_len = seq_len
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.labels = self.load_labels()

        # Debugging statement
        print(f"Loaded {len(self.image_files)} images and {len(self.labels)} labels with dim {len(self.labels[0])}.")

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
        return len(self.image_files) - self.seq_len + 1  # Ensure the length accounts for sequence length to avoid out of range errors

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        images = []
        for i in range(idx, idx + self.seq_len):
            img_name = os.path.join(self.image_dir, self.image_files[i])
            image = Image.open(img_name).convert('L')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)  # Stack images to create a sequence
        label = torch.tensor(self.labels[idx + self.seq_len - 1], dtype=torch.float32)  # Use the label of the last image in the sequence

        return images, label


class RoverDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling data loading.

    Args:
        data_dir (str): Directory with all the images.
        label_dir (str): Directory with labels.
        size (callable, optional): Optional size for transform.
        batch_size (int, optional): Batch size for data loading.
    """

    def __init__(self, data_dir: str, label_dir: str, seq_len: int = 5, size=(1, 256, 256), augment=False, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.size = size
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.val = None
        self.test = None

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
            train_dataset = RoverImageDataset(os.path.join(self.data_dir, 'train'), os.path.join(self.label_dir, 'train'), seq_len=self.seq_len, transform=train_transform)
            val_dataset = RoverImageDataset(os.path.join(self.data_dir, 'val'), os.path.join(self.label_dir, 'val'), seq_len=self.seq_len, transform=self.transform)

            return train_dataset, val_dataset, None

        if stage == 'test':
            test_dataset = RoverImageDataset(os.path.join(self.data_dir, 'test'), os.path.join(self.label_dir, 'test'), seq_len=self.seq_len, transform=self.transform)
            return None, None, test_dataset

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
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train, self.val, self.test = self.create_datasets(stage)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.train, self.val, self.test = self.create_datasets(stage)

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

    def __init__(self):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dropout_cnn = nn.Dropout(0.5)

        # LSTM layers
        self.lstm_input_size = 16 * 32 * 32  # Calculated based on the output dimensions of the CNN
        self.lstm = nn.LSTM(self.lstm_input_size, 16, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)

        # Fully connected layer
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        f_in = r_out[:, -1, :]
        output = self.fc(f_in)
        return output


# PyTorch Lightning Module
class RoverRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for rover regression.
    """

    def __init__(self):
        super(RoverRegressor, self).__init__()
        self.model = CNNLSTMModel()
        self.loss_fn = nn.MSELoss()

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
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, 2).float()
        # Compute loss for each label
        loss1 = self.loss_fn(y_hat[:, 0], y[:, 0])
        loss2 = self.loss_fn(y_hat[:, 1], y[:, 1])

        # Combine the losses
        combined_loss = self.loss_fn(y_hat, y)

        # Log both individual losses and the combined loss
        self.log('train_loss_vel(x)', loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_gyro(z)', loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, 2).float()
        # Compute loss for each label
        loss1 = self.loss_fn(y_hat[:, 0], y[:, 0])
        loss2 = self.loss_fn(y_hat[:, 1], y[:, 1])

        # Combine the losses
        combined_loss = self.loss_fn(y_hat, y)

        # Log both individual losses and the combined loss
        self.log('val_loss_vel(x)', loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_gyro(z)', loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', combined_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': combined_loss}

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: Optimizer object.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-6),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


def main():
    """
    Main function to run the training process.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    images = 'D:/Dataset/Rover/CPET/dataset/image'
    labels = 'D:/Dataset/Rover/CPET/dataset/label'

    log_callback = LogEpochLossCallback('logs/epoch_loss.csv')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )
    x = (1, 256, 256)
    data_module = RoverDataModule(images, labels, seq_len=3, size=x, augment=True, batch_size=32, num_workers=7)

    # Training
    model = RoverRegressor()
    # Print the model architecture using torchsummary
    save_model_summary(model, x)
    num_epoch = 10
    print(f'\nTraining for {num_epoch} epochs.')

    wait_for_user_input()

    trainer = pl.Trainer(max_epochs=num_epoch, accelerator='gpu', callbacks=[log_callback, early_stop_callback])
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
