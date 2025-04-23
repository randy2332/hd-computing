// dataset.c - Dataset Implementation
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Main dataset loading function - delegates to specific loaders
Dataset* load_dataset(DatasetType type, const char* train_or_test) {
    Dataset* dataset = NULL;
    
    switch (type) {
        case DATASET_MNIST:
            if (strcmp(train_or_test, "train") == 0) {
                dataset = load_mnist_dataset(MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS);
            } else {
                dataset = load_mnist_dataset(MNIST_TEST_IMAGES, MNIST_TEST_LABELS);
            }
            break;
            
        case DATASET_UCIHAR:
            if (strcmp(train_or_test, "train") == 0) {
                dataset = load_ucihar_dataset(UCIHAR_TRAIN_FEATURES, UCIHAR_TRAIN_LABELS);
            } else {
                dataset = load_ucihar_dataset(UCIHAR_TEST_FEATURES, UCIHAR_TEST_LABELS);
            }
            break;
            
        case DATASET_ISOLET:
            if (strcmp(train_or_test, "train") == 0) {
                dataset = load_isolet_dataset(ISOLET_TRAIN_FEATURES, "train");
            } else {
                dataset = load_isolet_dataset(ISOLET_TEST_FEATURES, "test");
            }
            break;
            
        case DATASET_CIFAR10:
            if (strcmp(train_or_test, "train") == 0) {
                dataset = load_cifar10_dataset(CIFAR10_DATA_DIR, "train");
            } else {
                dataset = load_cifar10_dataset(CIFAR10_DATA_DIR, "test");
            }
            break;
            
        case DATASET_FMNIST:
            if (strcmp(train_or_test, "train") == 0) {
                dataset = load_fmnist_dataset(FMNIST_TRAIN_IMAGES, FMNIST_TRAIN_LABELS);
            } else {
                dataset = load_fmnist_dataset(FMNIST_TEST_IMAGES, FMNIST_TEST_LABELS);
            }
            break;
            
        case DATASET_CONNECT4:
            // For Connect-4, we pass "train" or "test" to determine the split
            dataset = load_connect4_dataset(CONNECT4_DATA_FILE, train_or_test);
            break;
            
        default:
            printf("Error: Unknown dataset type %d\n", type);
            return NULL;
    }
    
    return dataset;
}

// Free dataset resources
void free_dataset(Dataset* dataset) {
    if (dataset) {
        if (dataset->features) {
            for (int i = 0; i < dataset->number_of_samples; i++) {
                free(dataset->features[i]);
            }
            free(dataset->features);
        }
        
        if (dataset->labels) {
            free(dataset->labels);
        }
        
        free(dataset);
    }
}

// Normalize floating point features to range [0, 1]
void normalize_features(float* features, int size, float min, float max) {
    float range = max - min;
    
    for (int i = 0; i < size; i++) {
        // Clamp values to the min-max range
        if (features[i] < min) features[i] = min;
        if (features[i] > max) features[i] = max;
        
        // Normalize to [0, 1]
        features[i] = (features[i] - min) / range;
    }
}

// Quantize floating point features to 8-bit unsigned values (0-255)
void quantize_features(float* features, unsigned char* quantized, int size) {
    for (int i = 0; i < size; i++) {
        // Scale from [0, 1] to [0, 255] and round to nearest integer
        quantized[i] = (unsigned char)(features[i] * 255.0f + 0.5f);
    }
}