// hd_bundling.c - Implementation of bundling operations
#include "hd_bundling.h"
#include <stdlib.h>
#include <stdio.h>

BundledVector* init_bundled_vector(int dimension) {
    BundledVector* bv = (BundledVector*)malloc(sizeof(BundledVector));
    if (!bv) return NULL;

    bv->dimension = dimension;
    
    // Allocate memory for sum vector
    bv->sum_vector = (int*)calloc(dimension, sizeof(int));  // Initialize to 0 with calloc
    if (!bv->sum_vector) {
        free(bv);
        return NULL;
    }

    // Allocate memory for final binarized vector
    bv->final_vector = (char*)malloc(dimension * sizeof(char));
    if (!bv->final_vector) {
        free(bv->sum_vector);
        free(bv);
        return NULL;
    }

    return bv;
}

void free_bundled_vector(BundledVector* bv) {
    if (bv) {
        free(bv->sum_vector);
        free(bv->final_vector);
        free(bv);
    }
}

void bundle_vectors(BoundVectors* bound, BundledVector* bundle) {
    // Reset sum vector
    for (int j = 0; j < bundle->dimension; j++) {
        bundle->sum_vector[j] = 0;
    }
    
    // Accumulation process - for binary encoding (0,1)
    for (int i = 0; i < bound->feature_dimension; i++) {
        for (int j = 0; j < bundle->dimension; j++) {
            bundle->sum_vector[j] += bound->bound_vectors[i][j];
        }
    }
    
    // Majority voting for binary encoding (threshold at n/2)
    int threshold = bound->feature_dimension / 2;
    for (int i = 0; i < bundle->dimension; i++) {
        bundle->final_vector[i] = (bundle->sum_vector[i] > threshold) ? 1 : 0;
    }
}

void print_bundling_result(BundledVector* bundle) {
    printf("\nBundling result sample (first 20 elements):\n");
    printf("Sum values: ");
    for (int i = 0; i < 20; i++) {
        printf("%4d ", bundle->sum_vector[i]);
    }
    printf("...\n");

    printf("Binarized result: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", bundle->final_vector[i]);
    }
    printf("...\n");

    // Count ones
    int ones = 0;
    for (int i = 0; i < bundle->dimension; i++) {
        if (bundle->final_vector[i] == 1) ones++;
    }
    printf("\nStatistics:\n");
    printf("Number of 1s: %d (%.2f%%)\n", ones, (float)ones * 100 / bundle->dimension);
    printf("Number of 0s: %d (%.2f%%)\n", 
           bundle->dimension - ones, 
           (float)(bundle->dimension - ones) * 100 / bundle->dimension);
}