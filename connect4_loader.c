// connect4_loader.c - Connect-4 Dataset Loader
#include "dataset.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*
 * The Connect-4 dataset contains sequences of game positions in the Connect-4 game.
 * Each instance represents a game board position, with 42 attributes (6 rows x 7 columns).
 * 
 * Format:
 * - CSV-like format with positions represented by characters
 * - Each attribute can be: 'x' (player 1), 'o' (player 2), or 'b' (blank)
 * - Class values are: 'win', 'loss', or 'draw' (from player 1's perspective)
 * 
 * The dataset contains 67,557 positions, and we'll split it 80/20 for training/testing.
 */

// Function to convert Connect-4 symbols to numeric values
static unsigned char symbol_to_value(char symbol) {
    switch(symbol) {
        case 'x': return 85;   // Player 1 - 映射到大約 1/3 的範圍 (85)
        case 'o': return 170;  // Player 2 - 映射到大約 2/3 的範圍 (170)
        case 'b': return 0;    // Blank - 保持為 0
        default:  return 0;    // Default to blank for invalid symbols
    }
}

// Function to convert class labels to numeric values
static unsigned char class_to_value(const char* class_str) {
    if (strcmp(class_str, "win") == 0) return 0;
    if (strcmp(class_str, "loss") == 0) return 1;
    if (strcmp(class_str, "draw") == 0) return 2;
    return 0; // Default to "win" for invalid classes
}

// Function to split a dataset into training and testing sets
// Returns 1 if the sample should be in the test set, 0 otherwise
static int is_test_sample(int sample_idx, float test_ratio) {
    // Use a deterministic method based on sample index
    return (sample_idx % 100) < (test_ratio * 100);
}

// Load Connect-4 dataset
Dataset* load_connect4_dataset(const char* data_path, const char* is_test) {
    FILE *file;
    Dataset* dataset;
    char line[1024];  // Buffer for reading lines
    int is_training = (strcmp(is_test, "train") == 0);
    float test_ratio = 0.2; // 20% for testing, 80% for training
    
    // First pass: count total samples and valid samples for the requested split
    printf("Connect-4: First pass - counting samples\n");
    
    file = fopen(data_path, "r");
    if (!file) {
        printf("Failed to open Connect-4 data file: %s\n", data_path);
        return NULL;
    }
    
    int total_samples = 0;
    int valid_samples = 0;
    
    while (fgets(line, sizeof(line), file) != NULL) {
        // Skip empty lines and comments
        if (strlen(line) <= 1 || line[0] == '#' || line[0] == '%') continue;
        
        // Check if this sample should be in the requested split
        if (is_test_sample(total_samples, test_ratio) != is_training) {
            total_samples++;
            continue;
        }
        
        valid_samples++;
        total_samples++;
    }
    
    fclose(file);
    
    printf("Connect-4: Found %d total samples, %d for %s set\n", 
           total_samples, valid_samples, is_training ? "training" : "test");
    
    // Allocate dataset structure
    dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    // Set dataset information
    strncpy(dataset->name, "CONNECT4", sizeof(dataset->name)-1);
    dataset->original_feature_type = 0; // Already discrete
    dataset->num_classes = CONNECT4_NUM_CLASSES;
    dataset->feature_dimension = CONNECT4_FEATURE_COUNT;
    dataset->number_of_samples = valid_samples;
    
    // Allocate memory for features and labels
    dataset->features = (unsigned char**)malloc(valid_samples * sizeof(unsigned char*));
    if (!dataset->features) {
        printf("Failed to allocate memory for features\n");
        free(dataset);
        return NULL;
    }
    
    for (int i = 0; i < valid_samples; i++) {
        dataset->features[i] = (unsigned char*)malloc(CONNECT4_FEATURE_COUNT * sizeof(unsigned char));
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
    
    dataset->labels = (unsigned char*)malloc(valid_samples * sizeof(unsigned char));
    if (!dataset->labels) {
        printf("Failed to allocate memory for labels\n");
        for (int i = 0; i < valid_samples; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset);
        return NULL;
    }
    
    // Second pass: load data
    printf("Connect-4: Second pass - loading data\n");
    
    file = fopen(data_path, "r");
    if (!file) {
        printf("Failed to open Connect-4 data file: %s\n", data_path);
        for (int i = 0; i < valid_samples; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
        free(dataset->labels);
        free(dataset);
        return NULL;
    }
    
    int sample_idx = 0;
    int current_sample = 0;
    
    while (fgets(line, sizeof(line), file) != NULL && current_sample < valid_samples) {
        // Skip empty lines and comments
        if (strlen(line) <= 1 || line[0] == '#' || line[0] == '%') continue;
        
        // Check if this sample should be in the requested split
        if (is_test_sample(sample_idx, test_ratio) != is_training) {
            sample_idx++;
            continue;
        }
        
        // Remove newline if present
        line[strcspn(line, "\n")] = 0;
        
        // Tokenize line by commas
        char *token = strtok(line, ",");
        int feature_idx = 0;
        
        // Process features (42 positions)
        while (token && feature_idx < CONNECT4_FEATURE_COUNT) {
            dataset->features[current_sample][feature_idx] = symbol_to_value(token[0]);
            feature_idx++;
            token = strtok(NULL, ",");
        }
        
        // The last token is the class label
        if (token) {
            dataset->labels[current_sample] = class_to_value(token);
        } else {
            printf("Warning: Missing class label for sample %d\n", sample_idx);
            dataset->labels[current_sample] = 0; // Default to "win"
        }
        
        current_sample++;
        sample_idx++;
    }
    
    fclose(file);
    
    // Verify that we loaded the expected number of samples
    if (current_sample != valid_samples) {
        printf("Warning: Expected to load %d samples but loaded %d\n", valid_samples, current_sample);
        dataset->number_of_samples = current_sample;
    }
    
    // Count samples per class
    int class_count[CONNECT4_NUM_CLASSES] = {0};
    for (int i = 0; i < current_sample; i++) {
        if (dataset->labels[i] < CONNECT4_NUM_CLASSES) {
            class_count[dataset->labels[i]]++;
        }
    }
    
    printf("\nConnect-4 class distribution in %s set:\n", is_training ? "training" : "test");
    const char* class_names[CONNECT4_NUM_CLASSES] = {"win", "loss", "draw"};
    for (int c = 0; c < CONNECT4_NUM_CLASSES; c++) {
        printf("Class %d (%s): %d samples\n", c, class_names[c], class_count[c]);
    }
    
    printf("Loaded Connect-4 %s dataset: %d samples, %d features, %d classes\n", 
           is_training ? "training" : "test", 
           dataset->number_of_samples, dataset->feature_dimension, dataset->num_classes);
    
    return dataset;
}