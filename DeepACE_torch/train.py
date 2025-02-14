#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2025 Tom Gajecki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Tom Gajecki
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Dataset, collate_fn
from tqdm import tqdm
import argparse
import datetime
import shutil
import os
from utils import * 
from model import DeepACE
from losses import LossFunctionSelector

# Set random seed for reproducibility
set_seed(42)

# Create a timestamp for saving model checkpoints and logs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def main(config, config_path):
    """
    Main training function for the DeepACE model.

    Args:
        config (dict): Configuration dictionary loaded from the YAML config file.
        config_path (str): Path to the configuration file (used for backup).
    """

    # -------------------------------------------------------------------------
    # 1. Prepare Datasets and DataLoaders
    # -------------------------------------------------------------------------

    # Initialize training and validation datasets using directories specified in config
    train_dataset = Dataset(
        config['train_mixture_dir'], config['train_target_dir'],
        sample_rate=config['sample_rate'],
        stim_rate=config['stim_rate'],
        segment_length=config['segment_length']
    )
    valid_dataset = Dataset(
        config['valid_mixture_dir'], config['valid_target_dir'],
        sample_rate=config['sample_rate'],
        stim_rate=config['stim_rate'],
        segment_length=config['segment_length']
    )

    # Create DataLoaders for training and validation with the specified batch size, workers, and collate function
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    # -------------------------------------------------------------------------
    # 2. Initialize Model, Loss Function, Optimizer, and Scheduler
    # -------------------------------------------------------------------------

    # Retrieve DeepACE model parameters from the configuration
    deepace_params = config['DeepACE']

    # Instantiate the loss function using a selector
    selector = LossFunctionSelector()
    loss_fn = selector.get_loss(config['loss'])

    # Initialize the DeepACE model with parameters from the config and move it to the designated device
    model = DeepACE(
        L=deepace_params['L'],
        N=deepace_params['N'],
        P=deepace_params['P'],
        B=deepace_params['B'],
        S=deepace_params['S'],
        H=deepace_params['H'],
        R=deepace_params['R'],
        X=deepace_params['X'],
        M=deepace_params['M'],
        msk_activate=deepace_params['msk_activate'],
        causal=deepace_params['causal']
    ).to(device)

    # Print the total number of parameters in the model
    print("Model size: ", sum(p.numel() for p in model.parameters()))

    # Set up the optimizer with model parameters
    optimizer = Adam(model.parameters(), lr=float(config['learning_rate']))

    # Initialize a learning rate scheduler to reduce LR on plateau (based on validation loss)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # -------------------------------------------------------------------------
    # 3. Setup Checkpointing and Early Stopping
    # -------------------------------------------------------------------------

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5

    # Define directory for saving model checkpoints and logs; include the timestamp in the path
    model_save_dir = os.path.join('../../../results/DeepACE_V2', f"run_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Copy the configuration file to the model save directory for future reference
    config_save_path = os.path.join(model_save_dir, 'config.yaml')
    shutil.copyfile(config_path, config_save_path)

    # Define the full model checkpoint save path with timestamp
    model_save_path = os.path.join(model_save_dir, f"model_{timestamp}.pth")

    # Lists for tracking training and validation losses over epochs
    train_losses = []
    val_losses = []

    # Maximum gradient norm for gradient clipping
    max_grad_norm = 5.0

    # -------------------------------------------------------------------------
    # 4. Training and Validation Loop
    # -------------------------------------------------------------------------

    for epoch in range(config['num_epochs']):
        print_divider()  # Utility function to print a divider in the console output

        # --- Training Phase ---
        model.train()  # Set the model to training mode
        total_loss = 0  # Accumulator for the training loss

        # Use tqdm for a progress bar during training
        train_loader_tqdm = tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}', leave=False)
        for _, (mix, target) in enumerate(train_loader_tqdm):
            # Move inputs and targets to the designated device (CPU or GPU)
            mix = mix.to(device)
            target = target.to(device)

            # Forward pass: compute the model's prediction
            estimate = model(mix)

            # Compute the loss between the prediction and the target
            loss = loss_fn(estimate, target)

            # Backpropagation: zero gradients, compute new gradients, and clip them
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update model parameters
            optimizer.step()

            # Accumulate the batch loss
            total_loss += loss.item()

        # Calculate and log the average training loss for the epoch
        avg_loss = total_loss / len(train_data_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}:\nTraining Loss: {avg_loss:.4f}')

        # --- Validation Phase ---
        model.eval()  # Set the model to evaluation mode
        val_loss = 0  # Accumulator for the validation loss

        # Disable gradient computation during validation for efficiency
        with torch.no_grad():
            val_loader_tqdm = tqdm(valid_data_loader, desc=f'Validation {epoch + 1}/{config["num_epochs"]}', leave=False)
            for mix, target in val_loader_tqdm:
                mix = mix.to(device)
                target = target.to(device)

                # Compute the model's prediction and loss on the validation set
                estimate = model(mix)
                loss = loss_fn(estimate, target)
                val_loss += loss.item()

            # Calculate the average validation loss
            avg_val_loss = val_loss / len(valid_data_loader)
            val_losses.append(avg_val_loss)
            print(f'Validation Loss: {avg_val_loss:.4f}')

            # --- Early Stopping and Checkpointing ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0  # Reset the no-improvement counter
                torch.save(model.state_dict(), model_save_path)  # Save the best model checkpoint
            else:
                epochs_no_improve += 1
                print(f'Epochs without improvement: {epochs_no_improve}/{early_stop_patience}')

            # Trigger early stopping if no improvement over a set number of epochs
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered. No improvement in validation loss for {early_stop_patience} epochs.")
                break

            # Step the learning rate scheduler based on the current validation loss
            scheduler.step(avg_val_loss)
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f'Learning rate: {current_lr}')

    # After training, plot and save the loss curves for both training and validation
    plot_loss(train_losses, val_losses, os.path.join(model_save_dir, 'training_progress.png'))


if __name__ == '__main__':
    # Parse command-line arguments to get the path to the configuration file
    parser = argparse.ArgumentParser(description="Train DeepACE")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Load configuration from the YAML file
    config = load_config(args.config)

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start the training process
    main(config, args.config)
