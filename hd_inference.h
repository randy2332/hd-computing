#ifndef HD_INFERENCE_H
#define HD_INFERENCE_H

#include "dataset.h"
#include "hd_level.h"
#include "hd_mapping.h"
#include "hd_binding.h"
#include "hd_bundling.h"
#include "hd_training.h"

// Structure to store inference results
typedef struct {
    int predicted_class;    // Predicted class
    int* similarities;      // Similarity with each class
} InferenceResult;

// Function declarations
InferenceResult* init_inference_result(int n_classes);
void free_inference_result(InferenceResult* result);
BundledVector* encode_test_sample(unsigned char* features, HDLevelVectors* hd, 
                                HDMapping* mapping, char** item_memory, 
                                int feature_dimension, int dimension);

#endif // HD_INFERENCE_H