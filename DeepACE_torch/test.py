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
from dataset import Dataset, collate_fn
import os
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import *  # Contains helper functions such as load_config and find_trained_models
from model import DeepACE
import scipy.io

def test_model(model, test_mixture_dir, output_dir, config):
    """
    Runs the model on test data and saves the predictions.

    Args:
        model (torch.nn.Module): The trained DeepACE model.
        test_mixture_dir (str): Directory containing the test mixtures.
        output_dir (str): Directory where prediction files will be saved.
        config (dict): Configuration parameters (includes sample_rate, segment_length, etc.).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()

    # Create a test dataset with is_test flag set to True (no target data)
    test_dataset = Dataset(
        test_mixture_dir, 
        None,
        sample_rate=config['sample_rate'], 
        segment_length=config['segment_length'], 
        is_test=True
    )

    # Create a DataLoader for the test dataset
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Wrap the DataLoader with tqdm for a progress bar
        test_loader_tqdm = tqdm(test_data_loader, desc='Testing', leave=False)
        for _, (mix, filenames) in enumerate(test_loader_tqdm):
            # Move the input mixture to the specified device
            mix = mix.to(device)

            # Forward pass: get the model's prediction
            # The output is processed: move to CPU, convert to NumPy, and transpose dimensions
            estimate = model(mix)[0].cpu().numpy().transpose(-1, -2)

            # Apply a threshold: set values below base_level to zero
            estimate[estimate < config['base_level']] = 0

            # Generate an output filename based on the input filename
            base_filename = Path(filenames[0]).stem
            cleaned_filename = base_filename.replace("mixed", "")
            output_filename = os.path.join(output_dir, cleaned_filename + "prediction.mat")

            # Save the prediction as a MATLAB file
            scipy.io.savemat(output_filename, {'pred_lgf': estimate})

if __name__ == '__main__':
    # Automatically find all trained models in the designated models folder
    model_paths = find_trained_models()
    print(f"Found models: {model_paths}")

    # Parse command-line arguments for the configuration file path
    parser = argparse.ArgumentParser(description="Test Convdeepace")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Set the device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loop through each found model and run testing/prediction
    for model_path in model_paths:
        print(f"Processing model: {model_path}")

        # Assume each model directory contains its corresponding 'config.yaml'
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, 'config.yaml')

        # Load the configuration for this model
        config = load_config(config_path)

        # Extract DeepACE model parameters from the configuration
        deepace_params = config['DeepACE']

        # Initialize the DeepACE model with the parameters from the config
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

        # Load the saved model weights (using map_location to ensure compatibility)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Create a directory within the model folder to save the predictions
        output_dir = os.path.join(model_dir, 'predictions/')
        os.makedirs(output_dir, exist_ok=True)

        # Get the directory containing test data from the configuration
        test_dir = config['test_dir']

        # Run the test function for the current model
        test_model(model, test_dir, output_dir, config)
