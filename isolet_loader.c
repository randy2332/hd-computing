// isolet_loader.c - ISOLET Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/*
 * The ISOLET (Isolated Letter Speech Recognition) dataset consists of
 * spoken letters of the English alphabet. The dataset contains 617 features
 * extracted from audio recordings, and 26 classes (A-Z).
 * 
 * Format:
 * - Data is in comma-separated format with 618 columns (617 features + 1 label)
 * - The label is the last column and ranges from 1-26 (corresponding to A-Z)
 * - The features are real-valued and need to be normalized and quantized
 * 
 * The dataset is split into:
 * - isolet1+2+3+4.data: Training set (6238 samples)
 * - isolet5.data: Test set (1559 samples)
 */

// Function to check if a string is a valid float
static int is_valid_float(const char* str) {
    // Skip leading whitespace
    while (isspace(*str)) str++;
    
    // Check for empty string
    if (*str == '\0') return 0;
    
    // Check for optional sign
    if (*str == '+' || *str == '-') str++;
    
    // Need at least one digit
    int has_digit = 0;
    int has_decimal = 0;
    
    while (*str) {
        if (isdigit(*str)) {
            has_digit = 1;
        } else if (*str == '.' && !has_decimal) {
            has_decimal = 1;
        } else if ((*str == 'e' || *str == 'E') && has_digit) {
            // Handle scientific notation
            str++;
            if (*str == '+' || *str == '-') str++;
            if (!isdigit(*str)) return 0; // Need at least one digit after exponent
        } else {
            return 0; // Invalid character
        }
        str++;
    }
    
    return has_digit;
}

// Load ISOLET dataset
Dataset* load_isolet_dataset(const char* feature_path, const char* is_test) {
    FILE *file;
    Dataset* dataset;
    char line[10000];  // Buffer for reading lines
    float *temp_features;
    int sample_count = 0;
    int feature_count = ISOLET_FEATURE_COUNT;
    float min_vals[ISOLET_FEATURE_COUNT];
    float max_vals[ISOLET_FEATURE_COUNT];
    
    // Initialize min/max values
    for (int i = 0; i < feature_count; i++) {
        min_vals[i] = INFINITY;
        max_vals[i] = -INFINITY;
    }
    
    // Allocate dataset structure
    dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    // Set dataset information
    strncpy(dataset->name, "ISOLET", sizeof(dataset->name)-1);
    dataset->original_feature_type = 1; // float
    dataset->num_classes = ISOLET_NUM_CLASSES; // 26 classes (A-Z)
    dataset->feature_dimension = feature_count;
    
    // First pass: count samples and find min/max values
    printf("ISOLET: First pass - counting samples and finding min/max values\n");
    
    file = fopen(feature_path, "r");
    if (!file) {
        printf("Failed to open feature file: %s\n", feature_path);
        free(dataset);
        return NULL;
    }
    
    // Count samples and compute min/max values for each feature
    while (fgets(line, sizeof(line), file) != NULL) {
        // Skip empty lines
        if (strlen(line) <= 1) continue;
        
        // Replace commas with spaces for easier parsing
        for (int i = 0; line[i]; i++) {
            if (line[i] == ',') line[i] = ' ';
        }
        
        char *token = strtok(line, " \t\n");
        int feature_idx = 0;
        int valid_sample = 1;
        
        // Process all features in the line
        while (token && feature_idx < feature_count) {
            if (!is_valid_float(token)) {
                valid_sample = 0;
                break;
            }
            
            float value = atof(token);
            
            // Update min/max values
            if (value < min_vals[feature_idx]) min_vals[feature_idx] = value;
            if (value > max_vals[feature_idx]) max_vals[feature_idx] = value;
            
            feature_idx++;
            token = strtok(NULL, " \t\n");
        }
        
        // Check if we have the class label
        if (valid_sample && token && is_valid_float(token)) {
            int label = (int)atof(token);
            
            // Ensure label is in valid range (1-26)
            if (label >= 1 && label <= ISOLET_NUM_CLASSES) {
                sample_count++;
            } else {
                printf("Warning: Invalid label %d found, skipping sample\n", label);
            }
        }
    }
    
    fclose(file);
    
    printf("ISOLET: Found %d valid samples\n", sample_count);
    
    // Print feature ranges
    printf("ISOLET: Feature ranges (min, max):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Feature %d: [%f, %f]\n", i, min_vals[i], max_vals[i]);
    }
    printf("  ...\n");
    
    // Set the number of samples
    dataset->number_of_samples = sample_count;
    
    // Allocate memory for features and labels
    dataset->features = (unsigned char**)malloc(sample_count * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        free(dataset);
        return NULL;
    }
    
    for (int i = 0; i < sample_count; i++) {
        dataset->features[i] = (unsigned char*)malloc(feature_count * sizeof(unsigned char));
        if (!dataset->features[i]) {
            printf("Failed to allocate memory for sample %d\n", i);
            for (int j = 0; j < i; j++) {
                free(dataset->features[j]);
            }
            free(dataset->features);
            free(dataset);
            return NULL;
        }
    }
    
    dataset->labels = (unsigned char*)malloc(sample_count * sizeof(unsigned char));
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        for (int i = 0; i < sample_count; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    // Temporary buffer for floating-point features
    temp_features = (float*)malloc(feature_count * sizeof(float));
    if (!temp_features) {
        printf("Failed to allocate temporary feature buffer\n");
        free(dataset->labels);
        for (int i = 0; i < sample_count; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    // Second pass: load data
    printf("ISOLET: Second pass - loading and preprocessing data\n");
    
    file = fopen(feature_path, "r");
    if (!file) {
        printf("Failed to open feature file: %s\n", feature_path);
        free(temp_features);
        free(dataset->labels);
        for (int i = 0; i < sample_count; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    int current_sample = 0;
    
    while (fgets(line, sizeof(line), file) != NULL && current_sample < sample_count) {
        // Skip empty lines
        if (strlen(line) <= 1) continue;
        
        // Replace commas with spaces for easier parsing
        for (int i = 0; line[i]; i++) {
            if (line[i] == ',') line[i] = ' ';
        }
        
        char *token = strtok(line, " \t\n");
        int feature_idx = 0;
        int valid_sample = 1;
        
        // Process all features in the line
        while (token && feature_idx < feature_count) {
            if (!is_valid_float(token)) {
                valid_sample = 0;
                break;
            }
            
            temp_features[feature_idx] = atof(token);
            feature_idx++;
            token = strtok(NULL, " \t\n");
        }
        
        // Check if we have the class label
        if (valid_sample && token && is_valid_float(token)) {
            int label = (int)atof(token);
            
            // Ensure label is in valid range (1-26)
            if (label >= 1 && label <= ISOLET_NUM_CLASSES) {
                // Store label (adjust to 0-based)
                dataset->labels[current_sample] = (unsigned char)(label - 1);
                
                // Normalize and quantize features
                for (int i = 0; i < feature_count; i++) {
                    // Normalize to [0, 1]
                    float normalized = (temp_features[i] - min_vals[i]) / (max_vals[i] - min_vals[i]);
                    // Clamp to [0, 1] range
                    if (normalized < 0.0f) normalized = 0.0f;
                    if (normalized > 1.0f) normalized = 1.0f;
                    // Quantize to 8-bit (0-255)
                    dataset->features[current_sample][i] = (unsigned char)(normalized * 255.0f + 0.5f);
                }
                
                current_sample++;
            }
        }
    }
    
    fclose(file);
    free(temp_features);
    
    // Make sure we found the expected number of samples
    if (current_sample != sample_count) {
        printf("Warning: Expected %d samples but found %d in second pass\n", 
               sample_count, current_sample);
        dataset->number_of_samples = current_sample;
    }
    
    printf("Loaded ISOLET %s dataset: %d samples, %d features, %d classes\n", 
           strcmp(is_test, "test") == 0 ? "test" : "train", 
           dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}