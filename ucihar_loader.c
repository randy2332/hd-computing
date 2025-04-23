// ucihar_loader.c - UCI HAR Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * The UCI HAR (Human Activity Recognition) dataset contains smartphone sensor data
 * for recognizing human activities. Features are floating-point values from
 * accelerometer and gyroscope readings, normalized to [-1, 1].
 * 
 * Format:
 * - Each row represents one sample with multiple features (561 features)
 * - There are 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, 
 *   SITTING, STANDING, LAYING
 */

// Load UCIHAR dataset
Dataset* load_ucihar_dataset(const char* feature_path, const char* label_path) {
    FILE *feature_file, *label_file;
    Dataset* dataset;
    char line[10000];  // Buffer for reading lines (UCI HAR has 561 features)
    float *temp_features;
    int sample_count = 0;
    int feature_count = 0;
    
    // Allocate dataset structure
    dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    // Set dataset information
    strncpy(dataset->name, "UCIHAR", sizeof(dataset->name)-1);
    dataset->original_feature_type = 1; // float
    dataset->num_classes = UCIHAR_NUM_CLASSES; // 6 classes
    
    // First, count the number of samples and features
    feature_file = fopen(feature_path, "r");
    if (!feature_file) {
        printf("Failed to open feature file: %s\n", feature_path);
        free(dataset);
        return NULL;
    }
    
    // Count samples and determine feature dimension
    while (fgets(line, sizeof(line), feature_file) != NULL) {
        if (sample_count == 0) {
            // Count features in the first sample
            char *token = strtok(line, " ,\t\n");
            while (token != NULL) {
                feature_count++;
                token = strtok(NULL, " ,\t\n");
            }
        }
        sample_count++;
    }
    
    // Set dataset properties
    dataset->number_of_samples = sample_count;
    dataset->feature_dimension = feature_count;
    
    // Rewind file to beginning
    rewind(feature_file);
    
    // Allocate memory for features
    dataset->features = (unsigned char**)malloc(sample_count * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        fclose(feature_file);
        free(dataset);
        return NULL;
    }
    
    // Temporary buffer for floating-point features
    temp_features = (float*)malloc(feature_count * sizeof(float));
    if (!temp_features) {
        printf("Failed to allocate temporary feature buffer\n");
        free(dataset->features);
        fclose(feature_file);
        free(dataset);
        return NULL;
    }
    
    // Read features and convert to 8-bit
    for (int i = 0; i < sample_count; i++) {
        if (fgets(line, sizeof(line), feature_file) == NULL) {
            printf("Error reading sample %d\n", i);
            // Clean up
            free(temp_features);
            for (int j = 0; j < i; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            fclose(feature_file);
            free(dataset);
            return NULL;
        }
        
        // Allocate memory for 8-bit features
        dataset->features[i] = (unsigned char*)malloc(feature_count * sizeof(unsigned char));
        if (!dataset->features[i]) {
            printf("Failed to allocate memory for sample %d\n", i);
            // Clean up
            free(temp_features);
            for (int j = 0; j < i; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            fclose(feature_file);
            free(dataset);
            return NULL;
        }
        
        // Parse floating-point features
        char *token = strtok(line, " ,\t\n");
        for (int j = 0; j < feature_count && token != NULL; j++) {
            temp_features[j] = atof(token);
            token = strtok(NULL, " ,\t\n");
        }
        
        // Convert floating-point features to 8-bit values
        // First normalize to [0, 1] (UCIHAR features are in range [-1, 1])
        normalize_features(temp_features, feature_count, -1.0f, 1.0f);
        
        // Then quantize to 8-bit (0-255)
        quantize_features(temp_features, dataset->features[i], feature_count);
    }
    
    fclose(feature_file);
    free(temp_features);
    
    // Read labels
    label_file = fopen(label_path, "r");
    if (!label_file) {
        printf("Failed to open label file: %s\n", label_path);
        for (int i = 0; i < sample_count; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    // Allocate memory for labels
    dataset->labels = (unsigned char*)malloc(sample_count * sizeof(unsigned char));
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        for (int i = 0; i < sample_count; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        fclose(label_file);
        free(dataset);
        return NULL;
    }
    
    // Read labels (UCIHAR labels are 1-based, convert to 0-based)
    for (int i = 0; i < sample_count; i++) {
        if (fgets(line, sizeof(line), label_file) == NULL) {
            printf("Error reading label %d\n", i);
            // Clean up
            free(dataset->labels);
            for (int j = 0; j < sample_count; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            fclose(label_file);
            free(dataset);
            return NULL;
        }
        
        // Convert 1-based label to 0-based
        int label = atoi(line);
        dataset->labels[i] = (unsigned char)(label - 1);
    }
    
    fclose(label_file);
    
    printf("Loaded UCIHAR dataset: %d samples, %d features, %d classes\n", 
           dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}