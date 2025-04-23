# hd computing for mcu

## Overview

This repository provides a Hyperdimensional (HD) Computing framework tailored for classification tasks across diverse datasets. HD Computing is a brain-inspired paradigm that leverages high-dimensional random vectors (hypervectors) to represent and process information efficiently. This implementation highlights the flexibility and effectiveness of HD Computing in addressing various machine learning classification challenges. A key feature of this project is its modular model design, which includes item hypervectors, level hypervectors, and class hypervectors. This structure facilitates lightweight deployment on resource-constrained IoT devices.

## Supported Datasets

The framework currently supports the following datasets:

1. **MNIST** - Handwritten digit recognition (28×28 grayscale images)
2. **Fashion-MNIST** - Fashion item recognition (28×28 grayscale images)
3. **UCIHAR** - Human Activity Recognition using smartphone sensor data
4. **ISOLET** - Spoken letter recognition with acoustic features
5. **CIFAR-10** - Object recognition in color images (32×32×3 RGB)
6. **Connect-4** - Board game position classification

## Getting Started

### Directory Structure

```

.
├── MNIST/              # MNIST dataset files
├── FMNIST/             # Fashion-MNIST dataset files
├── UCI_HAR/            # UCI HAR dataset files
├── ISOLET/             # ISOLET dataset files
├── CIFAR-10/           # CIFAR-10 dataset files
├── Connect-4/          # Connect-4 dataset files
├── PAMAP2/             # PAMAP2 dataset files
├── output/             # Generated model files
└── build/              # Build files

```

### Building the Project

```bash
make
```

### Running with Different Datasets

```bash

make run_mnist# Run with MNIST dataset
make run_fmnist# Run with Fashion-MNIST dataset
make run_ucihar# Run with UCI HAR dataset
make run_isolet# Run with ISOLET dataset
make run_cifar10# Run with CIFAR-10 dataset
make run_connect4# Run with Connect-4 dataset
```

### Cleaning Build Files

```bash
make clean# Remove object files and executable
make cleanall# Remove object files, executable, and generated models
```

## Implementation Details

### HD Computing Parameters

The framework's parameters can be configured in `config.h`:

- HD_DIMENSION: Dimension of hypervectors (default: 2000)
- HD_LEVEL_COUNT: Number of level vectors (default: 4)
- RANDOMNESS: Random component in level vectors (default: 0)

### Dataset Processing

Each dataset loader handles:

1. Loading dataset-specific formats
2. Normalizing features to the range [0, 1]
3. Quantizing to 8-bit values (0-255)
4. Converting class labels to zero-based indices

### Training Process

The training process involves:

1. Encoding each sample using binding and bundling operations
2. Accumulating encoded samples into class vectors
3. Applying majority voting to binarize the class vectors

### Testing and Evaluation

The evaluation provides:

1. Per-class accuracy metrics
2. Detailed Hamming distance for the first few test samples
3. Overall classification accuracy
