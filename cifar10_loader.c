// cifar10_loader.c - CIFAR-10 Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes.
 * There are 50000 training images and 10000 test images.
 * 
 * Format (binary):
 * - Each image is represented as 3073 bytes:
 *   - First byte is the label (0-9)
 *   - Next 3072 bytes are the image data (32x32x3)
 * 
 * Organization:
 * - Training data is split into 5 batches (data_batch_1.bin to data_batch_5.bin)
 * - Test data is in one batch (test_batch.bin)
 * 
 * Classes:
 * 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
 * 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
 */

// Function to read a CIFAR-10 batch file
static int read_cifar10_batch(const char* filename, 
                             unsigned char** features, 
                             unsigned char* labels, 
                             int offset, 
                             int max_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open CIFAR-10 batch file %s\n", filename);
        return 0;
    }
    
    // Buffer for reading one image at a time
    // Format: [label(1) + image_data(3072)]
    unsigned char buffer[3073];
    int images_read = 0;
    
    while (images_read < max_images && 
           fread(buffer, sizeof(unsigned char), 3073, file) == 3073) {
        // Extract label (first byte)
        labels[offset + images_read] = buffer[0];
        
        // Extract image data (next 3072 bytes)
        // Each image is in format [R(1024) + G(1024) + B(1024)]
        // Convert to grayscale or keep as RGB depending on need
        
        // Allocate memory for image data if not already done
        if (features[offset + images_read] == NULL) {
            features[offset + images_read] = (unsigned char*)malloc(CIFAR10_IMAGE_SIZE * sizeof(unsigned char));
            if (!features[offset + images_read]) {
                printf("Error: Failed to allocate memory for image %d\n", offset + images_read);
                fclose(file);
                return images_read;
            }
        }
        
        // Copy image data (keeping RGB format)
        memcpy(features[offset + images_read], buffer + 1, CIFAR10_IMAGE_SIZE);
        
        images_read++;
    }
    
    fclose(file);
    return images_read;
}

// Load CIFAR-10 dataset
Dataset* load_cifar10_dataset(const char* data_dir, const char* is_test) {
    Dataset* dataset;
    int is_training = (strcmp(is_test, "train") == 0);
    
    // Determine the number of samples based on training/test
    int num_samples = is_training ? 50000 : 10000;
    
    // Allocate dataset structure
    dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    // Set dataset information
    strncpy(dataset->name, "CIFAR10", sizeof(dataset->name)-1);
    dataset->original_feature_type = 0; // Already 8-bit
    dataset->num_classes = CIFAR10_NUM_CLASSES;
    dataset->feature_dimension = CIFAR10_IMAGE_SIZE;
    dataset->number_of_samples = num_samples;
    
    // Allocate memory for features and labels
    dataset->features = (unsigned char**)malloc(num_samples * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        free(dataset);
        return NULL;
    }
    
    // Initialize feature pointers to NULL
    for (int i = 0; i < num_samples; i++) {
        dataset->features[i] = NULL;
    }
    
    dataset->labels = (unsigned char*)malloc(num_samples * sizeof(unsigned char));
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    // Path buffer for file paths
    char filepath[512];
    int images_loaded = 0;
    
    if (is_training) {
        // Load 5 training batches
        for (int batch = 1; batch <= 5; batch++) {
            sprintf(filepath, "%s/%s%d.bin", data_dir, CIFAR10_TRAIN_BATCH_PREFIX, batch);
            printf("Loading training batch %d from %s\n", batch, filepath);
            
            int batch_images = read_cifar10_batch(filepath, dataset->features, 
                                                 dataset->labels, images_loaded, 
                                                 10000); // Each batch has 10000 images
            if (batch_images == 0) {
                printf("Warning: Failed to read batch %d\n", batch);
                continue;
            }
            
            images_loaded += batch_images;
            printf("Loaded %d images from batch %d, total: %d\n", 
                   batch_images, batch, images_loaded);
        }
    } else {
        // Load test batch
        sprintf(filepath, "%s/%s", data_dir, CIFAR10_TEST_BATCH);
        printf("Loading test batch from %s\n", filepath);
        
        int batch_images = read_cifar10_batch(filepath, dataset->features, 
                                             dataset->labels, 0, 
                                             num_samples);
        if (batch_images == 0) {
            printf("Error: Failed to read test batch\n");
            // Clean up
            free(dataset->labels);
            free(dataset->features);
            free(dataset);
            return NULL;
        }
        
        images_loaded = batch_images;
        printf("Loaded %d images from test batch\n", images_loaded);
    }
    
    // Check if we loaded the expected number of images
    if (images_loaded != num_samples) {
        printf("Warning: Expected to load %d images but loaded %d\n", 
               num_samples, images_loaded);
        dataset->number_of_samples = images_loaded;
    }
    
    // Count samples per class
    int class_count[CIFAR10_NUM_CLASSES] = {0};
    for (int i = 0; i < images_loaded; i++) {
        if (dataset->labels[i] < CIFAR10_NUM_CLASSES) {
            class_count[dataset->labels[i]]++;
        }
    }
    
    printf("\nCIFAR-10 class distribution in %s set:\n", is_training ? "training" : "test");
    for (int c = 0; c < CIFAR10_NUM_CLASSES; c++) {
        printf("Class %d: %d samples\n", c, class_count[c]);
    }
    
    printf("Loaded CIFAR-10 %s dataset: %d samples, %d features, %d classes\n", 
           is_training ? "training" : "test", 
           dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}