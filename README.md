# DeepACE: Dual-Framework Implementation for Audio Signal Processing

DeepACE is a comprehensive repository that implements ´cochlear implant audio signal processing techniques using deep learning. This project offers **dual-framework implementations in both PyTorch and TensorFlow**, providing flexibility for researchers and developers to experiment and compare performance across these popular deep learning platforms.

## Overview

DeepACE is designed for tasks such as audio mixture separation, enhancement, and cochlear implant simulation. The repository includes:

- **Complete Implementations:** Training, testing, and utility modules in both PyTorch and TensorFlow.
- **Custom Loss Functions and Modules:** Tailored layers and loss functions designed specifically for audio processing.
- **Flexible Data Handling:** Custom dataset classes and collate functions for efficient loading and pre-processing of audio data.
- **Reproducibility:** Utility scripts for setting seeds, logging training progress, and saving model checkpoints.

## Features

- **Dual-Framework Support:** Use PyTorch or TensorFlow, or compare both side-by-side.
- **End-to-End Pipeline:** Includes everything from dataset preparation and model training to evaluation and testing.
- **Customizable Configurations:** Easily modify model parameters and training settings via YAML configuration files.
- **State-of-the-Art Techniques:** Implements advanced techniques such as Squeeze-and-Excitation, cumulative layer normalization, and low-rank augmentations.

# References
<p align="justify">
[1] Gajecki T, Zhang Y, Nogueira W. A Deep Denoising Sound Coding Strategy for Cochlear Implants. IEEE Trans Biomed Eng. 2023 Sep;70(9):2700-2709. doi: 10.1109/TBME.2023.3262677. Epub 2023 Aug 30. PMID: 37030808.


---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeepACE.git
cd DeepACE
