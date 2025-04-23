#ifndef HD_SIMILARITY_H
#define HD_SIMILARITY_H

#include "hd_inference.h"
#include "hd_training.h"
#include "dataset.h"

// Calculate similarity and return prediction result
InferenceResult* compute_similarity(BundledVector* query, ClassVectors* cv);

// Evaluate an entire test set
void evaluate_test_set(Dataset* test_data, ClassVectors* cv, 
                      HDLevelVectors* hd, HDMapping* mapping, 
                      char** item_memory, int dimension);

// Calculate Hamming distance between binary vectors
int compute_hamming_distance(char* vec1, char* vec2, int dimension);

#endif // HD_SIMILARITY_H