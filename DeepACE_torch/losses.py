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
import torch.nn as nn


class LossFunctionSelector:
    """
    Utility class to select and return a loss function based on a provided name.
    Supported loss functions include L1, MSE, and a weighted feature loss.
    """
    def __init__(self):
        # Map loss function names to their corresponding getter methods.
        self.loss_functions = {
            "L1": self._get_l1_loss,
            "MSE": self._get_mse_loss,
            "weighted": self._get_weighted_loss
        }

    def get_loss(self, loss_name, **kwargs):
        """
        Retrieve the loss function specified by loss_name.

        Args:
            loss_name (str): Name of the loss function to retrieve.
            **kwargs: Additional keyword arguments (if any) for loss initialization.

        Returns:
            torch.nn.Module: The corresponding loss function.

        Raises:
            ValueError: If the specified loss function is not supported.
        """
        if loss_name in self.loss_functions:
            return self.loss_functions[loss_name](**kwargs)
        else:
            raise ValueError(f"Loss '{loss_name}' not found.")

    def _get_l1_loss(self):
        """Return the L1 loss function."""
        return nn.L1Loss()

    def _get_mse_loss(self):
        """Return the Mean Squared Error (MSE) loss function."""
        return nn.MSELoss()

    def _get_weighted_loss(self):
        """Return an instance of AutoWeightedFeatureLoss."""
        return AutoWeightedFeatureLoss()


class WeightedFeatureLoss(nn.Module):
    """
    Computes a weighted feature loss between input and target tensors.
    Each feature is weighted according to a predefined weights vector.
    """
    def __init__(self, weights=torch.tensor([0.5] * 11 + [1.0] * 11)):
        """
        Initialize the WeightedFeatureLoss.

        Args:
            weights (torch.Tensor): A 1D tensor of weights for each feature.
                                    Default is a tensor with 11 values of 0.5 and 11 values of 1.0.
        """
        super(WeightedFeatureLoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        """
        Compute the weighted loss between the input and target.

        Args:
            input (torch.Tensor): Predicted tensor of shape (batch, features, time_steps).
            target (torch.Tensor): Ground truth tensor of shape (batch, features, time_steps).

        Returns:
            torch.Tensor: The computed weighted loss.
        """
        # Ensure that the weights match the feature dimension.
        assert input.shape[1] == self.weights.shape[0], \
            "Weights should have the same number of elements as the feature dimension."

        # Compute the squared differences between input and target.
        loss_per_feature = (input - target) ** 2

        # Reshape weights for broadcasting and move them to the correct device.
        weighted_loss = loss_per_feature * self.weights.view(1, -1, 1).to(input.device)
        
        # Return the mean loss across all dimensions.
        return weighted_loss.mean()


class AutoWeightedFeatureLoss(nn.Module):
    """
    Computes an auto-weighted feature loss where the weights for each feature are automatically updated
    based on a running average of the per-feature errors.
    """
    def __init__(self, eps=1e-8, momentum=0.9):
        """
        Initialize the AutoWeightedFeatureLoss.

        Args:
            eps (float): Small constant to avoid division by zero.
            momentum (float): Momentum factor for updating the running error.
        """
        super(AutoWeightedFeatureLoss, self).__init__()
        self.eps = eps
        self.momentum = momentum
        # Register a buffer to hold the running error for each feature.
        self.register_buffer('running_error', torch.zeros(22))

    def forward(self, input, target):
        """
        Compute the auto-weighted loss between input and target.

        Args:
            input (torch.Tensor): Predicted tensor of shape (batch, features, time_steps).
            target (torch.Tensor): Ground truth tensor of shape (batch, features, time_steps).

        Returns:
            torch.Tensor: The computed auto-weighted loss.
        """
        # Compute the squared error per feature.
        loss_per_feature = (input - target) ** 2  # Shape: (batch, features, time_steps)
        
        # Average the loss over batch and time_steps to get error per feature.
        error_per_feature = loss_per_feature.mean(dim=(0, 2))  # Shape: (features,)

        # Ensure the error tensor is on the same device as the running_error.
        error_per_feature = error_per_feature.to(self.running_error.device)

        # Update the running error using momentum (without tracking gradients).
        with torch.no_grad():
            self.running_error.mul_(self.momentum)
            self.running_error.add_((1 - self.momentum) * error_per_feature.detach())

        # Compute normalized weights from the running error.
        weights = self.running_error / (self.running_error.sum() + self.eps)
        weights = weights.detach().view(1, -1, 1).to(input.device)

        # Apply the weights to the per-feature loss and compute the mean.
        weighted_loss = loss_per_feature * weights
        return weighted_loss.mean()
