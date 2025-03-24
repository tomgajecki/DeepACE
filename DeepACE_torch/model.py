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
from netblocks import *  


class DeepACE(torch.nn.Module):
    """
    DeepACE Model

    Args:
        L (int): Encoder/decoder convolution kernel size (<L>).
        N (int): Number of encoder filters / feature dimensions (<N>).
        P (int): Mask generator convolution kernel size (<P>).
        B (int): Number of channels in the conv block of the mask generator (<B>).
        S (int): Number of channels in the skip connections (<Sc>).
        H (int): Number of hidden units in the conv block of the mask generator (<H>).
        X (int): Number of layers in one conv block of the mask generator (<X>).
        R (int): Number of conv block stacks in the mask generator (<R>).
        M (int): Number of output channels.
        msk_activate (str): Activation function for the mask output (Default: "sigmoid").
        causal (bool): If True, applies causality constraint (only past context is used).
    """

    def __init__(
        self,
        L: int = 16,
        N: int = 512,
        P: int = 3,
        B: int = 128,
        S: int = 128,
        H: int = 512,
        X: int = 8,
        R: int = 3,
        M: int = 22,
        msk_activate: str = "sigmoid",
        causal: bool = False
    ):
        super().__init__()

        # Save encoder parameters and calculate stride
        self.enc_num_feats = N
        self.enc_kernel_size = L
        self.enc_stride = L // 2
        self.out_channels = M

        # ---------------------------------------------------------------------
        # Encoder: Conv1D layer that transforms the input waveform into features.
        # ---------------------------------------------------------------------
        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_feats,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,  # Padding to maintain feature alignment
            bias=False,
        )

        # Apply a non-linear activation after the encoder using a custom Rectifier.
        self.input_activation = Rectifier(self.enc_num_feats)

        # ---------------------------------------------------------------------
        # Mask Generator: Estimates masks to be applied on the encoder features.
        # ---------------------------------------------------------------------
        self.mask_generator = MaskGenerator(
            input_dim=N,
            kernel_size=P,
            num_feats_bn=B,
            num_feats_skip=S,
            num_hidden=H,
            num_layers=X,
            num_stacks=R,
            msk_activate=msk_activate,
            causal=causal
        )

        # ---------------------------------------------------------------------
        # Decoder: Transposed convolution to reconstruct the output from features.
        # ---------------------------------------------------------------------
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=self.enc_num_feats,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )

        self.balance = ChannelRebalancer(self.out_channels)

        # Final activation to constrain the output values.
        self.out_activation = torch.nn.Hardtanh(min_val=1e-6, max_val=1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DeepACE model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, 1, frames).

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        # Check if input has the expected shape: 3D tensor with 1 channel.
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        # Pad the input signal to ensure compatibility with encoder kernel size.
        padded, rest = self.pad_signal(input)

        # Pass through the encoder and apply activation.
        feats = self.encoder(padded)
        feats = self.input_activation(feats)

        # Generate the mask and apply it to the encoder features.
        mask = self.mask_generator(feats)
        masked = feats * mask

        decoded = self.decoder(masked)

        # Decode the normalized features to produce the output signal.
        output = self.balance(decoded)  

        # Then apply the final activation:
        output = self.out_activation(output)

        # ---------------------------------------------------------------------
        # Remove extra frames introduced by padding.
        # ---------------------------------------------------------------------
        frames_left = self.enc_kernel_size // self.enc_stride

        # Function to perform ceiling division.
        def ceil_div(a, b):
            return -(-a // b)

        frames_right = ceil_div(rest + self.enc_stride, self.enc_stride)

        # Remove the padded frames from the output.
        if frames_right > 0:
            output = output[..., frames_left:-frames_right]
        else:
            output = output[..., frames_left:]

        return output

    def pad_signal(self, input: torch.Tensor):
        """
        Pads the input signal to ensure its length is compatible with the encoder kernel.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, 1, frames) or (batch, frames).

        Returns:
            tuple: A tuple containing the padded input and the number of padded frames (rest).
        """
        # Accept only 2D or 3D input; if 2D, add a channel dimension.
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)
        nsample = input.size(2)

        # Calculate the required padding so that the signal length fits the encoder kernel size.
        rest = self.enc_kernel_size - (self.enc_stride + nsample % self.enc_kernel_size) % self.enc_kernel_size
        if rest > 0:
            # Pad at the end of the signal.
            pad = torch.zeros(batch_size, 1, rest, device=input.device, dtype=input.dtype)
            input = torch.cat([input, pad], dim=2)

        # Add auxiliary padding at the beginning and end to maintain context.
        pad_aux = torch.zeros(batch_size, 1, self.enc_stride, device=input.device, dtype=input.dtype)
        input = torch.cat([pad_aux, input, pad_aux], dim=2)

        return input, rest
