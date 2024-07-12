import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights


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


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for regression.
    """

    def __init__(self, input_dim=(3, 256, 256), out_channel_base=4, num_blocks=2, last_timestep=False, batch_norm=False, fc_layers=1, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=1, rnn_output_size=32, dropout=0.0, rnn_dropout=0.0, backbone='scratch', out_features=2):
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

    def __init__(self, loss='mse', learning_rate=1e-4, weight_decay=1e-5, input_dim=(3, 256, 256), out_channel_base=4, num_blocks=3, last_timestep=False, batch_norm=False, fc_layers=1, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=1, rnn_output_size=32, dropout=0.0, rnn_dropout=0.0,
                 backbone='scratch', out_features=2, paper=False):
        super(RoverRegressor, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.last_timestep = last_timestep
        self.model = CNNLSTMModel(input_dim, out_channel_base, num_blocks, last_timestep, batch_norm, fc_layers, fc_activation, fc_output_size, lstm_layers, rnn_output_size, dropout, rnn_dropout, backbone, out_features) if not paper else PaperArch(input_dim)
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


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(batch_size * time_steps, *x.size()[2:])
        y = self.module(x_reshaped)
        y = y.contiguous().view(batch_size, time_steps, *y.size()[1:])
        return y


class PaperArch(nn.Module):
    def __init__(self, image_shape):
        super(PaperArch, self).__init__()

        self.conv1 = TimeDistributed(nn.Conv2d(image_shape[0], 24, kernel_size=5, stride=2))
        self.relu1 = nn.ReLU()
        self.conv2 = TimeDistributed(nn.Conv2d(24, 36, kernel_size=5, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = TimeDistributed(nn.Conv2d(36, 48, kernel_size=5, stride=2))
        self.relu3 = nn.ReLU()
        self.conv4 = TimeDistributed(nn.Conv2d(48, 64, kernel_size=3, stride=1))
        self.relu4 = nn.ReLU()
        self.conv5 = TimeDistributed(nn.Conv2d(64, 16, kernel_size=3, stride=2))
        self.relu5 = nn.ReLU()

        self.flatten = TimeDistributed(nn.Flatten())
        self.dense1 = TimeDistributed(nn.Linear(1248, 128))
        self.dropout1 = nn.Dropout(0.1)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.final_dense = nn.Linear(128, 4)

    def forward(self, inputs_image):
        xi = inputs_image

        xi = self.conv1(xi)
        xi = self.relu1(xi)
        xi = self.conv2(xi)
        xi = self.relu2(xi)
        xi = self.conv3(xi)
        xi = self.relu3(xi)
        xi = self.conv4(xi)
        xi = self.relu4(xi)
        xi = self.conv5(xi)
        xi = self.relu5(xi)

        xi = self.flatten(xi)
        xi = self.dense1(xi)
        xi = self.dropout1(xi)
        xi, _ = self.lstm(xi)
        xi = self.final_dense(xi)

        return xi
