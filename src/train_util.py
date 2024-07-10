import csv
import torch.nn as nn
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import wandb


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
                wandb.log({f'train_loss_label_{i}': train_losses_values[i + 1]})
                wandb.log({f'val_loss_label_{i}': val_losses_values[i + 1]})

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
