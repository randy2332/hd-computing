// mnist_loader.c - MNIST Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Function to swap endianness (MNIST files are big-endian)
static uint32_t swap_endian(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

// Load MNIST dataset
Dataset* load_mnist_dataset(const char* image_path, const char* label_path) {
    FILE *image_file, *label_file;
    Dataset* dataset;
    uint32_t magic_number, num_images, num_rows, num_cols;
    uint32_t label_magic_number, num_labels;
    
    // Allocate dataset structure
    dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    // Set dataset information
    strncpy(dataset->name, "MNIST", sizeof(dataset->name)-1);
    dataset->original_feature_type = 0; // 8-bit

    // Open image file
    image_file = fopen(image_path, "rb");
    if (!image_file) {
        printf("Failed to open image file: %s\n", image_path);
        free(dataset);
        return NULL;
    }

    // Read image file header
    fread(&magic_number, sizeof(uint32_t), 1, image_file);
    fread(&num_images, sizeof(uint32_t), 1, image_file);
    fread(&num_rows, sizeof(uint32_t), 1, image_file);
    fread(&num_cols, sizeof(uint32_t), 1, image_file);

    // Convert endianness
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    // Open label file
    label_file = fopen(label_path, "rb");
    if (!label_file) {
        printf("Failed to open label file: %s\n", label_path);
        fclose(image_file);
        free(dataset);
        return NULL;
    }

    // Read label file header
    fread(&label_magic_number, sizeof(uint32_t), 1, label_file);
    fread(&num_labels, sizeof(uint32_t), 1, label_file);
    
    label_magic_number = swap_endian(label_magic_number);
    num_labels = swap_endian(num_labels);

    // Verify that image and label counts match
    if (num_images != num_labels) {
        printf("Image count (%d) and label count (%d) do not match\n", num_images, num_labels);
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }

    // Set dataset properties
    dataset->number_of_samples = num_images;
    dataset->feature_dimension = num_rows * num_cols;
    dataset->num_classes = MNIST_NUM_CLASSES;

    // Allocate memory for features and read data
    dataset->features = (unsigned char**)malloc(num_images * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }

    for (uint32_t  i = 0; i < num_images; i++) {
        dataset->features[i] = (unsigned char*)malloc(num_rows * num_cols);
        if (!dataset->features[i]) {
            printf("Failed to allocate memory for image %d\n", i);
            // Free previously allocated memory
            for (uint32_t  j = 0; j < i; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            fclose(image_file);
            fclose(label_file);
            free(dataset);
            return NULL;
        }
        // Read image data
        fread(dataset->features[i], 1, num_rows * num_cols, image_file);
    }

    // Read label data
    dataset->labels = (unsigned char*)malloc(num_images);
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        for (uint32_t  i = 0; i < num_images; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }
    fread(dataset->labels, 1, num_images, label_file);

    // Close files
    fclose(image_file);
    fclose(label_file);
    
    printf("Loaded MNIST dataset: %d samples, %d features, %d classes\n", 
           dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}