import csv

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
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.

        Args:
            trainer (pl.Trainer): The trainer object.
            pl_module (pl.LightningModule): The Lightning module being trained.
        """
        # Get the train loss from the trainer's logged metrics
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')

        # Convert to float if not None
        train_loss = float(train_loss) if train_loss is not None else None
        val_loss = float(val_loss) if val_loss is not None else None

        # Write the loss to the CSV file
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trainer.current_epoch, train_loss, val_loss])


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
        transform (callable, optional): Optional transform to be applied on an image sample.
        batch_size (int, optional): Batch size for data loading.
    """

    def __init__(self, data_dir: str, label_dir: str, seq_len: int = 5, transform=None, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.val = None
        self.test = None

    def create_datasets(self, stage):
        """
        Create datasets for training, validation, and testing based on the stage.

        Args:
            stage (str): Current stage - 'fit' or 'test'.

        Returns:
            tuple: Datasets for training, validation, and testing.
        """
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        # TODO: Check how to call the DataModule using the stages
        if stage == 'fit':
            train_dataset = RoverImageDataset(os.path.join(self.data_dir, 'train'), os.path.join(self.label_dir, 'train'), seq_len=self.seq_len, transform=self.transform)
            val_dataset = RoverImageDataset(os.path.join(self.data_dir, 'val'), os.path.join(self.label_dir, 'val'), seq_len=self.seq_len, transform=self.transform)

            return train_dataset, val_dataset, None

        if stage == 'test':
            test_dataset = RoverImageDataset(os.path.join(self.data_dir, 'test'), os.path.join(self.label_dir, 'test'), seq_len=self.seq_len, transform=self.transform)
            return None, None, test_dataset

        else:
            print('\nOne or more dataset where not initialized correctly.')
            exit()

        # train_dataset = RoverImageDataset(os.path.join(self.data_dir, 'train'), os.path.join(self.label_dir, 'train'), transform=self.transform)
        # val_dataset = RoverImageDataset(os.path.join(self.data_dir, 'val'), os.path.join(self.label_dir, 'val'), transform=self.transform)
        # test_dataset = RoverImageDataset(os.path.join(self.data_dir, 'test'), os.path.join(self.label_dir, 'test'), transform=self.transform)
        # return train_dataset, val_dataset, test_dataset

    def setup(self, stage: str):
        """
        Setup the datasets based on the stage.

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

        # self.train, self.val, self.test = self.create_datasets(stage)

    def train_dataloader(self):
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for the test set.
        """
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for regression.
    """

    def __init__(self):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(0.3)
        # LSTM layers
        self.lstm = nn.LSTM(128 * 16 * 16, 256, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(256, 2)

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
        # self.metric_fn = torchmetrics.MeanSquaredError()
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
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, 2).float()
        # metric = self.metric_fn(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

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
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.view(-1, 2).float()
        # metric = self.metric_fn(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: Optimizer object.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-5),
            'monitor': 'val_loss'
        }
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    """
    Main function to run the training process.
    """
    images = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
    labels = 'D:/Dataset/Rover/KBPR/dataset/label'

    log_callback = LogEpochLossCallback('logs/epoch_loss.csv')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )

    data_module = RoverDataModule(images, labels, seq_len=3, batch_size=32, num_workers=7)

    # Training
    model = RoverRegressor()
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', callbacks=[log_callback])
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
