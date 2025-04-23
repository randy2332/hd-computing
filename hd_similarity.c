// hd_similarity.c - Implementation of similarity measures
#include "hd_similarity.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Calculate Hamming distance (for binary encoding)
int compute_hamming_distance(char* vec1, char* vec2, int dimension) {
    int distance = 0;
    for (int i = 0; i < dimension; i++) {
        if (vec1[i] != vec2[i]) {
            distance++;
        }
    }
    return distance;
}

// Compute similarity using Hamming distance
InferenceResult* compute_similarity(BundledVector* query, ClassVectors* cv) {
    InferenceResult* result = init_inference_result(cv->n_classes);
    if (!result) return NULL;
    
    // Use Hamming distance as similarity measure
    int min_distance = cv->dimension + 1; // Initialize to maximum possible distance
    int predicted_class = -1;
    
    for (int c = 0; c < cv->n_classes; c++) {
        // Calculate Hamming distance
        int distance = compute_hamming_distance(
            query->final_vector,
            cv->class_hvs[c],
            cv->dimension
        );
        
        // Store similarity (using negative distance, so higher is better)
        result->similarities[c] = distance;
        
        // Update best match (minimum distance)
        if (distance < min_distance) {
            min_distance = distance;
            predicted_class = c;
        }
    }
    
    result->predicted_class = predicted_class;
    return result;
}

// Evaluate test set
void evaluate_test_set(Dataset* test_data, ClassVectors* cv,
                      HDLevelVectors* hd, HDMapping* mapping,
                      char** item_memory, int dimension) {
    int correct = 0;
    int total = 0;
    int* class_correct = (int*)calloc(cv->n_classes, sizeof(int));
    int* class_total = (int*)calloc(cv->n_classes, sizeof(int));
    
    printf("\nStarting evaluation using Hamming distance...\n");

    for (int i = 0; i < test_data->number_of_samples; i++) {
        if (i % 100 == 0) {
            printf("Processing test sample %d/%d\n", i, test_data->number_of_samples);
        }

        BundledVector* test_encoded = encode_test_sample(
            test_data->features[i],
            hd, mapping, item_memory,
            test_data->feature_dimension, dimension
        );
        
        if (!test_encoded) continue;

        // Use Hamming distance
        InferenceResult* result = compute_similarity(test_encoded, cv);

        if (result) {
            int true_label = test_data->labels[i];
            class_total[true_label]++;
            total++;

            if (result->predicted_class == true_label) {
                correct++;
                class_correct[true_label]++;
            }

            if (i < 5) {  // Show detailed info for first 5 predictions
                printf("\nTest sample %d:\n", i);
                printf("True label: %d, Predicted: %d\n", true_label, result->predicted_class);
                printf("Hamming distances (lower is better):\n");
                for (int c = 0; c < cv->n_classes; c++) {
                    printf("Class %d: %d ", c, result->similarities[c]);
                    
                    // Mark the minimum distance (best match)
                    if (c == result->predicted_class) {
                        printf("(BEST)");
                    }
                    
                    // Mark the true label
                    if (c == true_label) {
                        printf("(TRUE)");
                    }
                    
                    printf("\n");
                }
            }

            free_inference_result(result);
        }

        free_bundled_vector(test_encoded);
    }

    // Print overall accuracy
    float accuracy = (float)correct / total * 100;
    printf("\nOverall Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, total);

    // Print per-class accuracy
    printf("\nPer-class Accuracy:\n");
    for (int c = 0; c < cv->n_classes; c++) {
        float class_accuracy = class_total[c] > 0 ? 
            (float)class_correct[c] / class_total[c] * 100 : 0;
        printf("Class %d: %.2f%% (%d/%d)\n", 
               c, class_accuracy, class_correct[c], class_total[c]);
    }

    free(class_correct);
    free(class_total);
}