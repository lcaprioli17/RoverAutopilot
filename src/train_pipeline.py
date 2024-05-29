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
from pytorch_lightning import Trainer
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


def create_dataloaders(train_paths, val_paths, test_paths, train_targets, val_targets, test_targets, transform, batch_size=32):
    train_dataset = CustomImageDataset(train_paths, train_targets, transform=transform)
    val_dataset = CustomImageDataset(val_paths, val_targets, transform=transform)
    test_dataset = CustomImageDataset(test_paths, test_targets, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LossLoggerCallback(pl.callbacks.Callback):
    def __init__(self, log_file='D:/Dataset/Rover/KBPR/dataset/csv/loss_log.txt'):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")  # Header for the log file

    def on_epoch_end(self, trainer, pl_module):
        # Get the logged losses
        train_loss = trainer.callback_metrics.get('train_loss', None)

        # Append losses to the log file
        with open(self.log_file, 'a') as f:
            f.write(f"{trainer.current_epoch},{train_loss}\n")


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        # Assuming labels are stored in a file named 'labels.txt'
        self.labels = self.load_labels(labels_file)

    def load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = [float(line.strip()) for line in f.readlines()]
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
        batch_size, _, _, _ = x.size()
        c_in = x.view(batch_size, 1, 128, 128)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        f_in = r_out.view(batch_size, 256)
        output = self.fc(f_in)
        return output


# PyTorch Lightning Module
class AccelerationRegressor(pl.LightningModule):
    def __init__(self):
        super(AccelerationRegressor, self).__init__()
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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.loss_fn(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    images = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
    labels = 'D:/Dataset/Rover/KBPR/dataset/csv/labels.csv'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CustomImageDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training
    model = AccelerationRegressor()
    # Create a custom callback instance
    loss_logger = LossLoggerCallback()
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', callbacks=[loss_logger])
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
