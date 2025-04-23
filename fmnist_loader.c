// fmnist_loader.c - Fashion-MNIST Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * The Fashion-MNIST dataset consists of 70,000 28x28 grayscale images of fashion items in 10 classes.
 * There are 60,000 training images and 10,000 test images.
 * 
 * Format:
 * - Same format as MNIST (IDX format)
 * - Image files contain a 16-byte header followed by image data
 * - Label files contain an 8-byte header followed by label data
 * 
 * Classes:
 * 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
 * 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
 */

// Function to swap endianness (Fashion-MNIST files are big-endian)
static uint32_t swap_endian(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

// Load Fashion-MNIST dataset
Dataset* load_fmnist_dataset(const char* image_path, const char* label_path) {
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
    strncpy(dataset->name, "FMNIST", sizeof(dataset->name)-1);
    dataset->original_feature_type = 0; // 8-bit grayscale
    dataset->num_classes = FMNIST_NUM_CLASSES;
    
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

    // Verify magic number for image file (2051)
    if (magic_number != 2051) {
        printf("Invalid magic number in image file: %u\n", magic_number);
        fclose(image_file);
        free(dataset);
        return NULL;
    }

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

    // Verify magic number for label file (2049)
    if (label_magic_number != 2049) {
        printf("Invalid magic number in label file: %u\n", label_magic_number);
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }

    // Verify that image and label counts match
    if (num_images != num_labels) {
        printf("Image count (%u) and label count (%u) do not match\n", num_images, num_labels);
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }

    // Set dataset properties
    dataset->number_of_samples = num_images;
    dataset->feature_dimension = num_rows * num_cols;

    // Allocate memory for features and read data
    dataset->features = (unsigned char**)malloc(num_images * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        fclose(image_file);
        fclose(label_file);
        free(dataset);
        return NULL;
    }

    for (uint32_t i = 0; i < num_images; i++) {
        dataset->features[i] = (unsigned char*)malloc(dataset->feature_dimension * sizeof(unsigned char));
        if (!dataset->features[i]) {
            printf("Failed to allocate memory for image %u\n", i);
            // Free previously allocated memory
            for (uint32_t j = 0; j < i; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            fclose(image_file);
            fclose(label_file);
            free(dataset);
            return NULL;
        }
        // Read image data
        fread(dataset->features[i], 1, dataset->feature_dimension, image_file);
    }

    // Allocate memory for labels and read data
    dataset->labels = (unsigned char*)malloc(num_images * sizeof(unsigned char));
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        for (uint32_t i = 0; i < num_images; i++) {
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
    
    // Count samples per class
    int class_count[FMNIST_NUM_CLASSES] = {0};
    for (uint32_t i = 0; i < num_images; i++) {
        if (dataset->labels[i] < FMNIST_NUM_CLASSES) {
            class_count[dataset->labels[i]]++;
        }
    }
    
    printf("\nFashion-MNIST class distribution:\n");
    for (int c = 0; c < FMNIST_NUM_CLASSES; c++) {
        printf("Class %d: %d samples\n", c, class_count[c]);
    }
    
    printf("Loaded Fashion-MNIST dataset from %s: %d samples, %d features, %d classes\n", 
           image_path, dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}