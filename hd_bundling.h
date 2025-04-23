#ifndef HD_BUNDLING_H
#define HD_BUNDLING_H

#include "hd_binding.h"

// Structure to store bundling results
typedef struct {
    int dimension;        // Vector dimension
    int* sum_vector;      // Accumulator for bundling
    char* final_vector;   // Final binarized result
} BundledVector;

// Function declarations
BundledVector* init_bundled_vector(int dimension);
void free_bundled_vector(BundledVector* bv);
void bundle_vectors(BoundVectors* bound, BundledVector* bundle);
void print_bundling_result(BundledVector* bundle);

#endif // HD_BUNDLING_H