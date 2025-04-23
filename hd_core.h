// hd_core.h - High-level API for HD Computing
#ifndef HD_CORE_H
#define HD_CORE_H

#include "config.h"
#include "dataset.h"
#include "hd_level.h"
#include "hd_mapping.h"
#include "hd_binding.h"
#include "hd_bundling.h"
#include "hd_training.h"
#include "hd_inference.h"
#include "hd_similarity.h"

// The main HD Computing context structure
typedef struct {
    // Core HD components
    HDLevelVectors* level_vectors;
    HDMapping* mapping;
    char** item_memory;
    ClassVectors* class_vectors;
    
    // Configuration
    int dimension;
    int levels;
    float randomness;
  
    int feature_dimension;   // Renamed from image_size for generality
    int n_classes;
    
    // Flags
    int is_initialized;
    int is_trained;
    
    // Dataset information
    char dataset_name[64];
} HDContext;

// Initialization and cleanup
HDContext* hd_init(int dimension, int levels, float randomness, 
                  int feature_dimension, int n_classes, const char* dataset_name);
void hd_free(HDContext* context);

// Training functions
int hd_train(HDContext* context, Dataset* train_data);
int hd_save_model(HDContext* context, const char* filename);

// Inference functions
int hd_predict(HDContext* context, unsigned char* features, int* prediction);
float hd_evaluate(HDContext* context, Dataset* test_data);

// Internal utility functions (not to be used directly by client code)
void hd_encode_sample(HDContext* context, unsigned char* features, BundledVector** result);

#endif // HD_CORE_H