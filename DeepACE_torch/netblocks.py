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

from typing import Optional, Tuple
import torch


class Rectifier(torch.nn.Module):
    """
    Rectifier module that applies a custom rectification operation.
    
    The module computes the positive and negative parts of the input,
    concatenates them, applies a learned linear transformation, and then
    uses a Hardtanh activation to constrain the output values.
    
    Args:
        input_dim (int): Dimensionality of the input features.
    """
    def __init__(self, input_dim):
        super(Rectifier, self).__init__()
        self.input_dim = input_dim
        # Initialize a learnable kernel with shape (input_dim * 2, input_dim)
        self.kernel = torch.nn.Parameter(torch.empty(input_dim * 2, input_dim))
        torch.nn.init.kaiming_normal_(self.kernel, mode='fan_in', nonlinearity='relu')
        # Hardtanh activation to constrain output values between [1e-6, 1.0]
        self.out_activation = torch.nn.Hardtanh(min_val=1e-6, max_val=1.0)

    def forward(self, inputs):
        """
        Forward pass of the Rectifier.
        
        Args:
            inputs (torch.Tensor): Tensor of shape (batch, input_dim, length)
                                   or (batch, length, input_dim).
        
        Returns:
            torch.Tensor: The rectified and linearly transformed output.
        """
        # If inputs are not in the shape (batch, input_dim, length), permute them.
        if inputs.shape[1] != self.input_dim:
            inputs = inputs.permute(0, 2, 1)

        # Center the inputs by subtracting the mean along the last dimension.
        inputs = inputs - torch.mean(inputs, dim=-1, keepdim=True)

        # Compute positive and negative parts using ReLU.
        pos = torch.nn.functional.relu(inputs)
        neg = torch.nn.functional.relu(-inputs)

        # Concatenate positive and negative parts along the channel dimension.
        concatenated = torch.cat([pos, neg], dim=1)
        # Use the absolute value of the kernel to ensure non-negativity.
        abs_kernel = torch.abs(self.kernel)

        # Permute to prepare for matrix multiplication.
        concatenated = concatenated.permute(0, 2, 1)
        # Apply the learned linear transformation.
        mixed = torch.matmul(concatenated, abs_kernel)
        # Permute back to (batch, channels, length)
        mixed = mixed.permute(0, 2, 1)

        # Apply the final activation function.
        output = self.out_activation(mixed)
        return output


class SELayer(torch.nn.Module):
    """
    Squeeze-and-Excitation (SE) layer for channel-wise feature recalibration.
    
    Args:
        channel (int): Number of channels in the input.
        reduction (int): Reduction ratio for the intermediate dense layer.
    """
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        # Define the fully connected (dense) layers for generating channel weights.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the SELayer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, length).
        
        Returns:
            torch.Tensor: Scaled tensor after applying channel weights.
        """
        b, c, _ = x.size()
        # Global average pooling to get channel-wise statistics.
        y = self.avg_pool(x).view(b, c)
        # Generate weights and reshape for multiplication.
        y = self.fc(y).view(b, c, 1)
        # Scale the input tensor.
        return x * y

class ChannelRebalancer(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        # A simple 1x1 convolution to mix information across channels.
        self.channel_mixer = torch.nn.Conv1d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mixer(x)
        return x


class cLN(torch.nn.Module):
    """
    Cumulative Layer Normalization (cLN) normalizes inputs based on cumulative statistics.
    
    Args:
        dimension (int): Number of channels.
        eps (float): Small constant to avoid division by zero.
        trainable (bool): Whether the normalization parameters are trainable.
    """
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = torch.nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.register_buffer('gain', torch.ones(1, dimension, 1))
            self.register_buffer('bias', torch.zeros(1, dimension, 1))

    def forward(self, input):
        """
        Forward pass for cumulative layer normalization.
        
        Args:
            input (torch.Tensor): Tensor of shape (batch, channels, time_steps).
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        B, C, T = input.size()
        # Sum across channels at each time step.
        step_sum = input.sum(1)
        # Sum of squares across channels.
        step_pow_sum = input.pow(2).sum(1)
        # Compute cumulative sums over time.
        cum_sum = torch.cumsum(step_sum, dim=1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)
        # Number of entries per time step.
        entry_cnt = torch.arange(1, T + 1, device=input.device).view(1, -1) * C
        # Calculate cumulative mean and variance.
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum / entry_cnt) - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        # Reshape to match input dimensions.
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        # Normalize and apply learnable gain and bias.
        x = (input - cum_mean) / cum_std
        return x * self.gain + self.bias


class ConvBlock(torch.nn.Module):
    """
    Convolutional block with residual and skip connections, including LAuReL augmentation.
    
    Args:
        bn_channels (int): Number of channels for the batch normalization input.
        skip_channels (int): Number of channels for the skip connection output.
        hidden_channels (int): Number of hidden channels within the block.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor for the convolution.
        no_residual (bool): If True, no residual connection is used.
        causal (bool): If True, applies causal convolution (padding only on the left).
        laurel_rank (int): Rank for low-rank augmentation.
    """
    def __init__(
        self,
        bn_channels: int,
        skip_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int = 1,
        no_residual: bool = False,
        causal: bool = False,
        laurel_rank: int = 16  # Rank for low-rank augmentation
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # 1x1 convolution: reduces/increases dimensions before depthwise convolution.
        self.conv1x1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=bn_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            cLN(hidden_channels) if causal else torch.nn.GroupNorm(1, hidden_channels, eps=1e-8)
        )

        # Depthwise convolution with specified kernel and dilation.
        self.depthwise_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=0,  # Padding will be applied manually.
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            cLN(hidden_channels) if causal else torch.nn.GroupNorm(1, hidden_channels, eps=1e-8)
        )

        # Squeeze-and-Excitation layer to recalibrate features.
        self.se_layer = SELayer(hidden_channels)

        # LAuReL components:
        # 1. Learnable scalar weights for scaling the residual.
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        # 2. Low-rank augmentation parameters.
        self.laurel_rank = laurel_rank
        self.A = torch.nn.Parameter(torch.randn(bn_channels, laurel_rank))
        self.B = torch.nn.Parameter(torch.randn(laurel_rank, bn_channels))

        # Residual connection (optional 1x1 convolution).
        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=bn_channels, kernel_size=1)
        )
        # Skip connection via 1x1 convolution.
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=skip_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for the ConvBlock.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch, channels, time_steps).
        
        Returns:
            Tuple[Optional[torch.Tensor], torch.Tensor]:
                - Residual output (or None if no residual connection).
                - Skip connection output.
        """
        # Apply appropriate padding based on causality.
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation
            input_padded = torch.nn.functional.pad(input, (padding, 0))
        else:
            padding_needed = (self.kernel_size - 1) * self.dilation
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            input_padded = torch.nn.functional.pad(input, (pad_left, pad_right))

        # Pass through 1x1 convolution and depthwise convolution.
        x = self.conv1x1(input_padded)
        x = self.depthwise_conv(x)

        # Apply SE layer.
        x = self.se_layer(x)

        # Compute residual output if applicable.
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(x)
            # LAuReL-RW: Scale the residual.
            residual = self.alpha * residual
            # LAuReL-LR: Compute low-rank augmentation from the original input.
            input_reshaped = input.view(input.shape[0], input.shape[1], -1)
            augmentation = torch.matmul(self.B, input_reshaped)
            augmentation = torch.matmul(self.A, augmentation)
            augmentation = augmentation.view_as(input)
            # Combine the augmentation with the residual.
            residual = residual + self.beta * augmentation

        # Compute skip connection output.
        skip_out = self.skip_out(x)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """
    Mask Generator module that produces a mask over input features using multiple convolutional blocks.
    
    Args:
        input_dim (int): Dimensionality of the input features.
        kernel_size (int): Convolution kernel size for the mask generator.
        num_feats_bn (int): Number of features for the batch normalization branch.
        num_feats_skip (int): Number of features for the skip connection branch.
        num_hidden (int): Number of hidden channels in the convolutional blocks.
        num_layers (int): Number of layers per stack.
        num_stacks (int): Number of convolutional stacks.
        msk_activate (str): Activation function for the mask output ("sigmoid" or "relu").
        causal (bool): If True, applies causal convolutions.
    """
    def __init__(
        self,
        input_dim: int,
        kernel_size: int,
        num_feats_bn: int,
        num_feats_skip: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
        causal: bool
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_feats_bn = num_feats_bn
        self.causal = causal

        # Input normalization: use cLN if causal, else GroupNorm.
        if self.causal:
            self.input_norm = cLN(dimension=input_dim, eps=1e-8)
        else:
            self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        # Initial 1x1 convolution to project input features.
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats_bn, kernel_size=1)

        # Build the convolutional layers using multiple stacks and layers.
        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        bn_channels=num_feats_bn,
                        skip_channels=num_feats_skip,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                        causal=self.causal,
                    )
                )
                # Update the receptive field size.
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * dilation

        # Final processing layers for generating the mask.
        self.output_prelu = torch.nn.PReLU()
        self.adjust_channels_conv = torch.nn.Conv1d(
            in_channels=num_feats_skip,
            out_channels=num_feats_bn,
            kernel_size=1,
        )
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats_bn,
            out_channels=input_dim,
            kernel_size=1,
        )
        # Set the mask activation function.
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MaskGenerator.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch, input_dim, time_steps).
        
        Returns:
            torch.Tensor: Output mask tensor of shape (batch, input_dim, time_steps).
        """
        # Normalize and apply the initial convolution.
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        # Pass through each convolutional block.
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:
                feats = feats + residual
            output = output + skip
        # Apply final activation and adjust channels if needed.
        output = self.output_prelu(output)
        if output.shape[1] != self.num_feats_bn:
            output = self.adjust_channels_conv(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output
