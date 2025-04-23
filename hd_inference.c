// hd_inference.c - Implementation of inference operations
#include "hd_inference.h"
#include <stdio.h>
#include <stdlib.h>

InferenceResult* init_inference_result(int n_classes) {
    InferenceResult* result = (InferenceResult*)malloc(sizeof(InferenceResult));
    if (!result) return NULL;

    result->similarities = (int*)malloc(n_classes * sizeof(int));
    if (!result->similarities) {
        free(result);
        return NULL;
    }

    result->predicted_class = -1;
    for (int i = 0; i < n_classes; i++) {
        result->similarities[i] = 0;
    }

    return result;
}

void free_inference_result(InferenceResult* result) {
    if (result) {
        if (result->similarities) {
            free(result->similarities);
        }
        free(result);
    }
}

BundledVector* encode_test_sample(unsigned char* features, HDLevelVectors* hd, 
                                HDMapping* mapping, char** item_memory, 
                                int feature_dimension, int dimension) {
    // 1. Create bound vectors
    BoundVectors* bound = init_bound_vectors(dimension, feature_dimension);
    if (!bound) {
        printf("Failed to initialize bound vectors for test sample\n");
        return NULL;
    }

    // 2. Binding
    bind_features(features, hd, mapping, item_memory, bound);

    // 3. Bundling
    BundledVector* bundle = init_bundled_vector(dimension);
    if (!bundle) {
        printf("Failed to initialize bundle vector for test sample\n");
        free_bound_vectors(bound);
        return NULL;
    }

    bundle_vectors(bound, bundle);
    free_bound_vectors(bound);

    return bundle;
}