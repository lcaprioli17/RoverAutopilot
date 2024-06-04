import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
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
from tqdm import tqdm


class RoverImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.labels_file = os.path.join(label_dir, 'labels.csv')
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.labels = self.load_labels()

        # Debugging statement
        print(f"Loaded {len(self.image_files)} images and {len(self.labels)} labels.")

    def load_labels(self):
        df = pd.read_csv(self.labels_file)
        labels = df.iloc[:, 0].astype(float).tolist()  # Assuming labels are in the first column
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class RoverDataModule(LightningDataModule):
    def __init__(self, data_dir: str, label_dir: str, transform=None, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.batch_size = batch_size
        self.train = None
        self.val = None
        self.test = None

    def create_datasets(self, stage):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        # TODO: Check how to call the DataModule using the stages
        if stage == 'fit':
            train_dataset = RoverImageDataset(os.path.join(self.data_dir, 'train'), os.path.join(self.label_dir, 'train'), transform=self.transform)
            val_dataset = RoverImageDataset(os.path.join(self.data_dir, 'val'), os.path.join(self.label_dir, 'val'), transform=self.transform)

            return train_dataset, val_dataset, None

        if stage == 'test':
            test_dataset = RoverImageDataset(os.path.join(self.data_dir, 'test'), os.path.join(self.label_dir, 'test'), transform=self.transform)
            return None, None, test_dataset

        else:
            print('\nOne or more dataset where not initialized correctly.')
            exit()

        # train_dataset = RoverImageDataset(os.path.join(self.data_dir, 'train'), os.path.join(self.label_dir, 'train'), transform=self.transform)
        # val_dataset = RoverImageDataset(os.path.join(self.data_dir, 'val'), os.path.join(self.label_dir, 'val'), transform=self.transform)
        # test_dataset = RoverImageDataset(os.path.join(self.data_dir, 'test'), os.path.join(self.label_dir, 'test'), transform=self.transform)
        # return train_dataset, val_dataset, test_dataset

    def setup(self, stage: str):
        # TODO: Check how to call the DataModule using the stages
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train, self.val, self.test = self.create_datasets(stage)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.train, self.val, self.test = self.create_datasets(stage)

        # self.train, self.val, self.test = self.create_datasets(stage)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


class CNNLSTMModel(nn.Module):
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
        # LSTM layers
        self.lstm = nn.LSTM(128 * 16 * 16, 256, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        c_in = x.view(batch_size, c, h, w)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        f_in = r_out.view(batch_size, 256)
        output = self.fc(f_in)
        return output


# PyTorch Lightning Module
class RoverRegressor(pl.LightningModule):
    def __init__(self):
        super(RoverRegressor, self).__init__()
        self.model = CNNLSTMModel()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    images = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
    labels = 'D:/Dataset/Rover/KBPR/dataset/label'

    data_module = RoverDataModule(images, labels)

    # Training
    model = RoverRegressor()
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
