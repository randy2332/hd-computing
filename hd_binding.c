// hd_binding.c - Implementation of binding operations
#include "hd_binding.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>

BoundVectors* init_bound_vectors(int dimension, int feature_dimension) {
    BoundVectors* bv = (BoundVectors*)malloc(sizeof(BoundVectors));
    if (!bv) return NULL;

    bv->dimension = dimension;
    bv->feature_dimension = feature_dimension;

    // Allocate memory for all binding results
    bv->bound_vectors = (char**)malloc(feature_dimension * sizeof(char*));
    if (!bv->bound_vectors) {
        free(bv);
        return NULL;
    }

    // Allocate memory for each binding result
    for (int i = 0; i < feature_dimension; i++) {
        bv->bound_vectors[i] = (char*)malloc(dimension * sizeof(char));
        if (!bv->bound_vectors[i]) {
            // Clean up already allocated memory
            for (int j = 0; j < i; j++) {
                free(bv->bound_vectors[j]);
            }
            free(bv->bound_vectors);
            free(bv);
            return NULL;
        }
    }

    return bv;
}

void free_bound_vectors(BoundVectors* bv) {
    if (bv) {
        if (bv->bound_vectors) {
            for (int i = 0; i < bv->feature_dimension; i++) {
                free(bv->bound_vectors[i]);
            }
            free(bv->bound_vectors);
        }
        free(bv);
    }
}

// Binary binding operation (XOR)
void bind_vectors(char* level_vector, char* item_vector, char* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        // XOR operation for binary encoding (0,1)
        result[i] = level_vector[i] ^ item_vector[i];
    }
}

// Bind feature vector using item memory and XOR binding
void bind_features(unsigned char* features, HDLevelVectors* hd, HDMapping* mapping, 
                  char** item_memory, BoundVectors* bound) {
    for (int i = 0; i < bound->feature_dimension; i++) {
        // Get level vector for the feature value
        char* level_vector = get_level_vector(hd, features[i], mapping);
        
        // Perform binding operation (XOR)
        for (int j = 0; j < bound->dimension; j++) {
            bound->bound_vectors[i][j] = level_vector[j] ^ item_memory[i][j];
        }
    }
}