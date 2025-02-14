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

import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset
import scipy.io


class Dataset(Dataset):
    """
    Custom Dataset class for loading mixture and source (electrodogram) audio files.

    Args:
        mixtures_dir (str): Directory containing mixture audio files (.wav).
        sources_dir (str, optional): Directory containing source electrodogram files (.mat). 
                                     Use None for test mode.
        sample_rate (int, optional): Sample rate for audio files (default: 16000).
        segment_length (float, optional): Length of each audio segment in seconds.
        stim_rate (int, optional): Cochlear implant stimulation rate.
        transform (callable, optional): Optional transform to apply on the audio sample.
        is_test (bool, optional): If True, load entire files (test mode).
    """
    def __init__(self, mixtures_dir, sources_dir=None, sample_rate=16000, segment_length=4.0,
                 stim_rate=1000, transform=None, is_test=False):
        self.mixtures_dir = mixtures_dir
        self.sources_dir = sources_dir
        self.sample_rate = sample_rate
        # Convert segment length from seconds to number of samples.
        self.segment_length = int(sample_rate * segment_length)
        self.transform = transform
        self.is_test = is_test

        # Determine block shift based on sample and stimulation rates.
        self.block_shift = int(math.ceil(sample_rate / stim_rate))
        # Calculate number of frames for electrodogram segments.
        self.num_frames = int(math.ceil(sample_rate * segment_length / self.block_shift))

        # List and sort all mixture files (wav) in the directory.
        self.mixtures = sorted(
            [os.path.join(mixtures_dir, f) for f in os.listdir(mixtures_dir) if f.endswith('.wav')]
        )

        if not self.is_test:
            # For training/validation mode, list and sort source files (.mat).
            self.sources = sorted(
                [os.path.join(sources_dir, f) for f in os.listdir(sources_dir) if f.endswith('.mat')]
            )
            # Ensure that each mixture has a corresponding source file.
            assert len(self.mixtures) == len(self.sources), "Number of mixtures and sources must be equal."

        # Create a list of segments for each file.
        if not self.is_test:
            self.segments = []
            for mix_path in self.mixtures:
                # Get information about the audio file.
                info = torchaudio.info(mix_path)
                num_samples = info.num_frames
                # Calculate number of segments needed to cover the entire file.
                num_segments = (num_samples + self.segment_length - 1) // self.segment_length
                # Find corresponding source file.
                src_path = self.sources[self.mixtures.index(mix_path)]
                for i in range(num_segments):
                    start_idx = i * self.segment_length
                    self.segments.append((mix_path, src_path, start_idx))
        else:
            # In test mode, load entire file without segmenting.
            self.segments = [(mix_path, None, None) for mix_path in self.mixtures]

    def __len__(self):
        """Return the total number of segments."""
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Retrieve a segment (or full file in test mode) from the dataset.

        Args:
            idx (int): Index of the segment to retrieve.

        Returns:
            tuple: (mix_waveform, source_electrodogram) for training mode or 
                   (mix_waveform, mix_path) for test mode.
        """
        mix_path, src_path, start_idx = self.segments[idx]

        if self.is_test:
            # In test mode, load the entire mixture waveform.
            mix_waveform, _ = torchaudio.load(mix_path)
            if self.transform:
                mix_waveform = self.transform(mix_waveform)
            return mix_waveform, mix_path
        else:
            # For training/validation, load a segment of the mixture and corresponding source.
            mix_waveform, _ = self.load_audio(mix_path, start_idx)
            src_elec = self.load_electrodogram(src_path, start_idx)

            if self.transform:
                mix_waveform = self.transform(mix_waveform)
                src_elec = self.transform(src_elec)

            return mix_waveform, src_elec

    def load_audio(self, file_path, start_idx):
        """
        Load an audio segment from a .wav file.

        Args:
            file_path (str): Path to the audio file.
            start_idx (int): Starting index (in samples) for the segment.

        Returns:
            tuple: (waveform segment, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(file_path)
        end_idx = start_idx + self.segment_length
        waveform_segment = waveform[:, start_idx:end_idx]
        return waveform_segment, sample_rate

    def load_electrodogram(self, file_path, start_idx):
        """
        Load an electrodogram segment from a .mat file.

        Args:
            file_path (str): Path to the .mat file.
            start_idx (int): Starting index (in samples) for the segment.

        Returns:
            torch.Tensor: Electrodogram segment of shape (channels, time_steps).
        """
        # Load the 'lgf' variable from the .mat file and convert to a float tensor.
        elec = torch.tensor(scipy.io.loadmat(file_path)['lgf']).float()
        # Clamp negative values to zero.
        elec = torch.clamp(elec, min=0.0)
        # Adjust indices based on block shift.
        start_idx = start_idx // self.block_shift
        end_idx = start_idx + (self.segment_length) // self.block_shift
        # Extract the segment and transpose it so that channels are the first dimension.
        elec_segment = elec[start_idx:end_idx, :]
        elec_segment = elec_segment.transpose(0, 1)
        return elec_segment


def collate_fn(batch):
    """
    Custom collate function for batching dataset samples.

    Handles both test mode (where the second element is a filename) and training/validation mode
    (where the second element is an electrodogram tensor).

    Args:
        batch (list): List of tuples returned by Dataset.__getitem__.

    Returns:
        tuple: Batched mixture waveforms and either filenames (test mode) or electrodograms.
    """
    # Check if we're in test mode (second element is a string)
    if isinstance(batch[0][1], str):
        mix_waveforms = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
    else:
        mix_waveforms = [item[0] for item in batch]
        src_elecs = [item[1] for item in batch]

    # Pad mixture waveforms to the maximum length in the batch.
    max_len = max([waveform.size(1) for waveform in mix_waveforms])
    mix_waveforms_padded = [
        torch.nn.functional.pad(w, (0, max_len - w.size(1))) for w in mix_waveforms
    ]
    mix_waveforms_batch = torch.stack(mix_waveforms_padded)

    # If in test mode, return the batch and filenames.
    if isinstance(batch[0][1], str):
        return mix_waveforms_batch, filenames
    else:
        # For training mode, also pad the electrodogram sequences.
        max_len_elec = max([w.size(1) for w in src_elecs])
        src_elecs_padded = [
            torch.nn.functional.pad(w, (0, max_len_elec - w.size(1))) for w in src_elecs
        ]
        src_elecs_batch = torch.stack(src_elecs_padded)
        return mix_waveforms_batch, src_elecs_batch
