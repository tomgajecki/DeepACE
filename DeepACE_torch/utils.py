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

import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import random
import torch
import shutil
import glob
import os

def plot_loss(train_losses, val_losses, save_path):
    """
    Plot and save the training and validation loss curves.

    Args:
        train_losses (list or array): List of training loss values per epoch.
        val_losses (list or array): List of validation loss values per epoch.
        save_path (str): Path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Training progress plot saved to {save_path}")

def load_config(config_file):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_parser():
    """
    Create and return an argument parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="ConvTasNet Training Script with SI-SDR")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    return parser

def print_divider():
    """
    Print a divider line to the terminal for better readability.
    """
    terminal_width = shutil.get_terminal_size().columns
    print("=" * terminal_width)
    print(" ")

def set_seed(seed: int):
    """
    Set random seed for reproducibility in Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_tensor_min_max(tensor):
    """
    Normalize a tensor to the range [-1, 1] using min-max normalization.

    Args:
        tensor (torch.Tensor): The tensor to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    
    # Normalize the tensor between -1 and 1
    normalized_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    return normalized_tensor

def find_trained_models():
    """
    Scan the models directory for available trained model files (.pth).

    Returns:
        list: A list of paths to the trained model files.

    Raises:
        FileNotFoundError: If no trained model directories or .pth files are found.
    """
    # Find directories matching the trained model pattern
    model_dirs = glob.glob('../../../results/DeepACE_V2/run_*')
    if not model_dirs:
        raise FileNotFoundError("No trained models found.")

    # Sort directories by modification time (oldest first)
    model_dirs.sort(key=os.path.getmtime, reverse=False)

    # Gather all .pth files from the found directories
    model_paths = []
    for model_dir in model_dirs:
        model_files = glob.glob(os.path.join(model_dir, '*.pth'))
        if model_files:
            model_paths.extend(model_files)

    if not model_paths:
        raise FileNotFoundError("No model files (.pth) found in 'models/'.")
    
    return model_paths

def display_layer_gradients(model):
    """
    Print gradient statistics (mean and standard deviation) for each layer of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
    """
    print("Gradients for individual layers:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            mean_grad = param.grad.mean().item()
            std_grad = param.grad.std().item()
            print(f"Layer: {name} | Gradient Mean: {mean_grad:.6f} | Gradient Std: {std_grad:.6f}")
        else:
            print(f"Layer: {name} | No gradients available")
    
    # Optionally print the gradient statistics of the last inspected layer
    try:
        print(f"Gradient Mean: {mean_grad:.6f}, Gradient Std: {std_grad:.6f}")
    except NameError:
        print("No gradients computed for any layer.")
