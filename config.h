// config.h - HD Computing configuration parameters
#ifndef HD_CONFIG_H
#define HD_CONFIG_H

// HD Computing dimensions
#define HD_DIMENSION 2000
#define HD_PACKED_DIMENSION (HD_DIMENSION / 8 + (HD_DIMENSION % 8 ? 1 : 0))

// HD Computing parameters
#define HD_LEVEL_COUNT 2
#define RANDOMNESS 0

// Dataset selection
#define DATASET_TYPE DATASET_MNIST  // Default dataset

// MNIST dataset parameters
#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
#define MNIST_NUM_CLASSES 10

// FMNIST (Fashion-MNIST) dataset parameters
#define FMNIST_IMAGE_ROWS 28
#define FMNIST_IMAGE_COLS 28
#define FMNIST_IMAGE_SIZE (FMNIST_IMAGE_ROWS * FMNIST_IMAGE_COLS)
#define FMNIST_NUM_CLASSES 10

// UCIHAR dataset parameters
#define UCIHAR_FEATURE_COUNT 561
#define UCIHAR_NUM_CLASSES 6

// ISOLET dataset parameters
#define ISOLET_FEATURE_COUNT 617
#define ISOLET_NUM_CLASSES 26

// CIFAR-10 dataset parameters
#define CIFAR10_IMAGE_ROWS 32
#define CIFAR10_IMAGE_COLS 32
#define CIFAR10_IMAGE_CHANNELS 3
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_ROWS * CIFAR10_IMAGE_COLS * CIFAR10_IMAGE_CHANNELS)
#define CIFAR10_NUM_CLASSES 10

// Connect-4 dataset parameters
#define CONNECT4_FEATURE_COUNT 42  // 7x6 board represented as a flat array
#define CONNECT4_NUM_CLASSES 3     // win, loss, draw

// File paths
// MNIST
#define MNIST_TRAIN_IMAGES "./MNIST/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS "./MNIST/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES "./MNIST/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS "./MNIST/t10k-labels-idx1-ubyte"

// FMNIST (Fashion-MNIST)
#define FMNIST_TRAIN_IMAGES "./FMNIST/train-images-idx3-ubyte"
#define FMNIST_TRAIN_LABELS "./FMNIST/train-labels-idx1-ubyte"
#define FMNIST_TEST_IMAGES "./FMNIST/t10k-images-idx3-ubyte"
#define FMNIST_TEST_LABELS "./FMNIST/t10k-labels-idx1-ubyte"

// UCIHAR
#define UCIHAR_TRAIN_FEATURES "./UCI_HAR/train/X_train.txt"
#define UCIHAR_TRAIN_LABELS "./UCI_HAR/train/y_train.txt"
#define UCIHAR_TEST_FEATURES "./UCI_HAR/test/X_test.txt"
#define UCIHAR_TEST_LABELS "./UCI_HAR/test/y_test.txt"

// ISOLET
#define ISOLET_TRAIN_FEATURES "./ISOLET/isolet1+2+3+4.data"
#define ISOLET_TEST_FEATURES "./ISOLET/isolet5.data"

// CIFAR-10
#define CIFAR10_DATA_DIR "./CIFAR-10/cifar-10-batches-bin"
#define CIFAR10_TRAIN_BATCH_PREFIX "data_batch_"
#define CIFAR10_TEST_BATCH "test_batch.bin"

// Connect-4
#define CONNECT4_DATA_FILE "./Connect-4/connect-4.data"

// Output files
#define HD_PACKED_VECTORS_FILE "./output/hd_model.h"

// Debug options
#define HD_DEBUG_PRINT 0  // Set to 0 to disable debug printing
#define WRITETESTDATA 1   // Set to 1 to write first 5 test samples to header file

// Test data output file
#define TEST_DATA_FILE "./output/test_data.h"

#endif // HD_CONFIG_H