import shutil
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from src import train_util, model_util, dataloader_util, data_preparation


def wait_for_user_input():
    while True:
        user_input = input("\nContinue execution? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Continuing execution...")
            return
        elif user_input == 'n':
            print("Aborting...")
            data_preparation.delete_folder('logs')
            data_preparation.delete_folder('lightning_logs')
            sys.exit()
        else:
            print("Invalid input! Please enter 'y' to continue or 'n' to abort.")


def main():
    """
    Main function to run the training process.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    images = 'D:/Dataset/devens_snowy_fixed'
    labels = 'D:/Dataset/devens_snowy_fixed'

    num_epoch = 100
    early_stopping_epoch = 50
    early_stopping_patience = 20

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))

    log_callback = train_util.LogEpochLossCallback('logs/epoch_loss.csv', 4)
    early_stop_callback = train_util.DelayedStartEarlyStopping(
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

    x = (3, 144, 256)
    data_module = dataloader_util.RoverDataModule(images, labels, seq_len=5, stride=3, size=x, channel='RGB', augment=False, batch_size=64, num_workers=7, split_by_traverse=True, val_traverse=5)

    # Training
    model = model_util.RoverRegressor(loss='mse', learning_rate=1e-3, weight_decay=1e-4, input_dim=x, out_channel_base=4, num_blocks=4, last_timestep=False, batch_norm=False, fc_layers=1, fc_activation=nn.ReLU, fc_output_size=128, lstm_layers=1, rnn_output_size=128, dropout=0, rnn_dropout=0,
                                      backbone='scratch', out_features=4, paper=False)

    # Print the model architecture using torchsummary
    train_util.save_model_summary(model, x)
    print(f'\nTraining for maximum {num_epoch} epochs. Early stopping with patience {early_stopping_patience} starting from epoch {early_stopping_epoch}')

    wait_for_user_input()

    # Initialize a new run
    wandb.init(project='Rover Autopilot')

    trainer = pl.Trainer(
        max_epochs=num_epoch,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[log_callback, early_stop_callback, checkpoint_callback]
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
