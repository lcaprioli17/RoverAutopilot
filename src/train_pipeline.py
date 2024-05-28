import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer


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
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CNNLSTM(pl.LightningModule):
    def __init__(self, cnn_output_size, hidden_size, num_layers, learning_rate=0.001):
        super(CNNLSTM, self).__init__()
        self.save_hyperparameters()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * (cnn_output_size // 8) * (cnn_output_size // 8), cnn_output_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.cnn(c_in)
        c_out = self.flatten(c_out)
        c_out = self.fc(c_out)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.lstm(r_in)
        r_out = self.fc_out(r_out[:, -1, :])
        return r_out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.MSELoss()(outputs, labels.view(-1, 1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    image_dir = 'D:/Dataset/Rover/KBPR/dataset/image/raw'
    labels = 'D:/Dataset/Rover/KBPR/dataset/csv/labels.csv'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(image_dir, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    cnn_output_size = 128
    hidden_size = 128
    num_layers = 2
    num_epochs = 10
    learning_rate = 0.001

    model = CNNLSTM(cnn_output_size, hidden_size, num_layers, learning_rate)
    trainer = Trainer(max_epochs=num_epochs, accelerator="gpu")
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
