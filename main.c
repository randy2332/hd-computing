// main.c - HD Computing for Multiple Datasets

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "config.h"
#include "hd_core.h"
#include "dataset.h"

// Function to print usage information
void print_usage(char* program_name) {
    printf("Usage: %s [dataset_type]\n", program_name);
    printf("  dataset_type: 'mnist', 'fmnist', 'ucihar', 'isolet', 'cifar10', or 'connect4' (default: 'mnist')\n");
}

int main(int argc, char* argv[]) {
    DatasetType dataset_type = DATASET_MNIST; // Default to MNIST
    
    // Parse command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "mnist") == 0) {
            dataset_type = DATASET_MNIST;
        } else if (strcmp(argv[1], "ucihar") == 0) {
            dataset_type = DATASET_UCIHAR;
        } else if (strcmp(argv[1], "isolet") == 0) {
            dataset_type = DATASET_ISOLET;
        } else if (strcmp(argv[1], "cifar10") == 0) {
            dataset_type = DATASET_CIFAR10;
        } else if (strcmp(argv[1], "fmnist") == 0) {
            dataset_type = DATASET_FMNIST;
        } else if (strcmp(argv[1], "connect4") == 0) {
            dataset_type = DATASET_CONNECT4;
        } else {
            printf("Unknown dataset type: %s\n", argv[1]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("=== HD Computing for Classification ===\n\n");
    
    // Print configuration settings
    printf("Configuration:\n");
    printf("- HD Dimension: %d\n", HD_DIMENSION);
    printf("- Levels: %d\n", HD_LEVEL_COUNT);
    printf("- Encoding: Binary (0,1)\n");
    
    // Dataset-specific information
    int feature_dimension = 0;
    int num_classes = 0;
    const char* dataset_name = NULL;
    
    switch (dataset_type) {
        case DATASET_MNIST:
            feature_dimension = MNIST_IMAGE_SIZE;
            num_classes = MNIST_NUM_CLASSES;
            dataset_name = "MNIST";
            printf("- Dataset: MNIST\n");
            printf("- Feature Dimension: %d (%dx%d)\n", 
                   MNIST_IMAGE_SIZE, MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS);
            printf("- Classes: %d\n", MNIST_NUM_CLASSES);
            break;
            
        case DATASET_UCIHAR:
            feature_dimension = UCIHAR_FEATURE_COUNT;
            num_classes = UCIHAR_NUM_CLASSES;
            dataset_name = "UCIHAR";
            printf("- Dataset: UCI HAR\n");
            printf("- Feature Dimension: %d\n", UCIHAR_FEATURE_COUNT);
            printf("- Classes: %d\n", UCIHAR_NUM_CLASSES);
            break;
            
        case DATASET_ISOLET:
            feature_dimension = ISOLET_FEATURE_COUNT;
            num_classes = ISOLET_NUM_CLASSES;
            dataset_name = "ISOLET";
            printf("- Dataset: ISOLET\n");
            printf("- Feature Dimension: %d\n", ISOLET_FEATURE_COUNT);
            printf("- Classes: %d (A-Z)\n", ISOLET_NUM_CLASSES);
            break;
            
        case DATASET_CIFAR10:
            feature_dimension = CIFAR10_IMAGE_SIZE;
            num_classes = CIFAR10_NUM_CLASSES;
            dataset_name = "CIFAR10";
            printf("- Dataset: CIFAR-10\n");
            printf("- Feature Dimension: %d (%dx%dx%d)\n", 
                   CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_ROWS, CIFAR10_IMAGE_COLS, CIFAR10_IMAGE_CHANNELS);
            printf("- Classes: %d\n", CIFAR10_NUM_CLASSES);
            break;
            
        case DATASET_FMNIST:
            feature_dimension = FMNIST_IMAGE_SIZE;
            num_classes = FMNIST_NUM_CLASSES;
            dataset_name = "FMNIST";
            printf("- Dataset: Fashion-MNIST\n");
            printf("- Feature Dimension: %d (%dx%d)\n", 
                   FMNIST_IMAGE_SIZE, FMNIST_IMAGE_ROWS, FMNIST_IMAGE_COLS);
            printf("- Classes: %d\n", FMNIST_NUM_CLASSES);
            break;
            
        case DATASET_CONNECT4:
            feature_dimension = CONNECT4_FEATURE_COUNT;
            num_classes = CONNECT4_NUM_CLASSES;
            dataset_name = "CONNECT4";
            printf("- Dataset: Connect-4\n");
            printf("- Feature Dimension: %d (7x6 board)\n", CONNECT4_FEATURE_COUNT);
            printf("- Classes: %d (win, loss, draw)\n", CONNECT4_NUM_CLASSES);
            break;
            
        default:
            printf("Unsupported dataset type\n");
            return 1;
    }
    
    printf("\n");
    
    // Load training data
    printf("Loading %s training data...\n", dataset_name);
    Dataset* train_data = load_dataset(dataset_type, "train");
    
    if (!train_data) {
        printf("Failed to load training data\n");
        return 1;
    }
    
    printf("Loaded %d training samples\n", train_data->number_of_samples);
    
    // Initialize HD computing context
    printf("\nInitializing HD computing...\n");
    HDContext* hd_context = hd_init(
        HD_DIMENSION,
        HD_LEVEL_COUNT,
        RANDOMNESS,
        feature_dimension,
        num_classes,
        dataset_name
    );
    
    if (!hd_context) {
        printf("Failed to initialize HD computing\n");
        free_dataset(train_data);
        return 1;
    }
    
    // Train the model
    printf("\n=== Training Phase ===\n");
    if (!hd_train(hd_context, train_data)) {
        printf("Training failed\n");
        hd_free(hd_context);
        free_dataset(train_data);
        return 1;
    }
    
    // Load test data
    printf("\nLoading %s test data...\n", dataset_name);
    Dataset* test_data = load_dataset(dataset_type, "test");
    
    if (!test_data) {
        printf("Failed to load test data\n");
        hd_free(hd_context);
        free_dataset(train_data);
        return 1;
    }
    
    printf("Loaded %d test samples\n", test_data->number_of_samples);
    
    // Evaluate the model
    printf("\n=== Testing Phase ===\n");
    float accuracy = hd_evaluate(hd_context, test_data);
    
    // Save the model
    printf("\n=== Saving Model ===\n");
    char model_filename[256];
    sprintf(model_filename, "./output/%s_model.h", dataset_name);
    
    if (!hd_save_model(hd_context, model_filename)) {
        printf("Failed to save model\n");
    }
    
    // Clean up resources
    printf("\nCleaning up resources...\n");
    hd_free(hd_context);
    free_dataset(test_data);
    free_dataset(train_data);
    
    printf("\nProgram completed with %.2f%% accuracy\n", accuracy);
    return 0;
}