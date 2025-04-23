// hd_binding.h - Header for binding operations
#ifndef HD_BINDING_H
#define HD_BINDING_H

#include "hd_level.h"
#include "hd_mapping.h"

// Structure for storing binding results
typedef struct {
    int dimension;          // Vector dimension
    int feature_dimension;  // Number of features
    char **bound_vectors;   // Array of bound vectors
} BoundVectors;

// Function declarations
BoundVectors* init_bound_vectors(int dimension, int feature_dimension);
void free_bound_vectors(BoundVectors* bv);
void bind_vectors(char* level_vector, char* item_vector, char* result, int dimension);
void bind_features(unsigned char* features, HDLevelVectors* hd, HDMapping* mapping, 
                  char** item_memory, BoundVectors* bound);

#endif // HD_BINDING_H