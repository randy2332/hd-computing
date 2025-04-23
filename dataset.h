// dataset.h - Generic Dataset Interface
#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>

// Generic Dataset Structure
typedef struct {
    int number_of_samples;       // Total number of samples
    int feature_dimension;       // Number of features per sample
    int num_classes;             // Number of classes
    unsigned char **features;    // 2D array of features (quantized to 8-bit)
    unsigned char *labels;       // 1D array of labels
    
    // Dataset information
    char name[64];               // Dataset name
    int original_feature_type;   // 0 = 8-bit, 1 = float, 2 = other
} Dataset;

// Enumeration for dataset types
typedef enum {
    DATASET_MNIST = 0,
    DATASET_UCIHAR = 1,
    DATASET_ISOLET = 2,
    DATASET_CIFAR10 = 3,
    DATASET_FMNIST = 4,
    DATASET_CONNECT4 = 5,
    // Add more datasets here
    DATASET_COUNT
} DatasetType;

// Function declarations
Dataset* load_dataset(DatasetType type, const char* train_or_test);
void free_dataset(Dataset* dataset);

// Dataset-specific loaders (to be implemented in separate files)
Dataset* load_mnist_dataset(const char* image_path, const char* label_path);
Dataset* load_ucihar_dataset(const char* feature_path, const char* label_path);
Dataset* load_isolet_dataset(const char* feature_path, const char* is_test);
Dataset* load_cifar10_dataset(const char* data_dir, const char* is_test);
Dataset* load_fmnist_dataset(const char* image_path, const char* label_path);
Dataset* load_connect4_dataset(const char* data_path, const char* is_test);

// Preprocessing functions
void normalize_features(float* features, int size, float min, float max);
void quantize_features(float* features, unsigned char* quantized, int size);

#endif // DATASET_H