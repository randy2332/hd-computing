// hd_core.c - Implementation of the high-level HD Computing API
#include "hd_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// Generate item memory with improved error handling
static char** generate_item_memory(int feature_dimension, int dimension) {
    char** item_memory = (char**)malloc(feature_dimension * sizeof(char*));
    if (!item_memory) {
        printf("Failed to allocate item memory array\n");
        return NULL;
    }

    // Initialize each item vector
    for (int i = 0; i < feature_dimension; i++) {
        item_memory[i] = (char*)malloc(dimension * sizeof(char));
        if (!item_memory[i]) {
            printf("Failed to allocate item memory vector %d\n", i);
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(item_memory[j]);
            }
            free(item_memory);
            return NULL;
        }

        // Generate random binary values (0 or 1)
        for (int j = 0; j < dimension; j++) {
            item_memory[i][j] = rand() % 2;
        }
    }

    // Debug output for verification
    if (HD_DEBUG_PRINT) {
        for (int i = 0; i < 3 && i < feature_dimension; i++) {
            printf("Item memory[%d] first 5 elements: ", i);
            for (int j = 0; j < 5 && j < dimension; j++) {
                printf("%d ", item_memory[i][j]);
            }
            printf("\n");
        }
    }

    return item_memory;
}

// Initialize the HD computing context
HDContext* hd_init(int dimension, int levels, float randomness, 
                  int feature_dimension, int n_classes, const char* dataset_name) {
    // Allocate context structure
    HDContext* context = (HDContext*)malloc(sizeof(HDContext));
    if (!context) {
        printf("Failed to allocate HD context\n");
        return NULL;
    }

    // Initialize context values
    context->dimension = dimension;
    context->levels = levels;
    context->randomness = randomness;
    context->feature_dimension = feature_dimension;
    context->n_classes = n_classes;
    context->is_initialized = 0;
    context->is_trained = 0;
    
    // Copy dataset name
    strncpy(context->dataset_name, dataset_name, sizeof(context->dataset_name)-1);
    
    // Initialize random number generator
    srand((unsigned int)time(NULL));
    
    // Initialize HD level vectors
    context->level_vectors = init_level_vectors(levels, dimension, randomness);
    if (!context->level_vectors) {
        printf("Failed to initialize HD level vectors\n");
        free(context);
        return NULL;
    }
    
    // Initialize mapping
    context->mapping = init_mapping(0, 255, levels);
    if (!context->mapping) {
        printf("Failed to initialize HD mapping\n");
        free_level_vectors(context->level_vectors);
        free(context);
        return NULL;
    }
    
    // Generate item memory
    context->item_memory = generate_item_memory(feature_dimension, dimension);
    if (!context->item_memory) {
        printf("Failed to generate item memory\n");
        free_mapping(context->mapping);
        free_level_vectors(context->level_vectors);
        free(context);
        return NULL;
    }
    
    // Initialize class vectors (will be populated during training)
    context->class_vectors = init_class_vectors(n_classes, dimension);
    if (!context->class_vectors) {
        printf("Failed to initialize class vectors\n");
        for (int i = 0; i < feature_dimension; i++) {
            free(context->item_memory[i]);
        }
        free(context->item_memory);
        free_mapping(context->mapping);
        free_level_vectors(context->level_vectors);
        free(context);
        return NULL;
    }
    
    context->is_initialized = 1;
    printf("HD Computing context initialized successfully for %s dataset\n", 
           context->dataset_name);
    return context;
}

// Free all resources associated with the HD context
void hd_free(HDContext* context) {
    if (!context) return;
    
    // Free class vectors
    if (context->class_vectors) {
        free_class_vectors(context->class_vectors);
    }
    
    // Free item memory
    if (context->item_memory) {
        for (int i = 0; i < context->feature_dimension; i++) {
            free(context->item_memory[i]);
        }
        free(context->item_memory);
    }
    
    // Free mapping and level vectors
    if (context->mapping) {
        free_mapping(context->mapping);
    }
    
    if (context->level_vectors) {
        free_level_vectors(context->level_vectors);
    }
    
    // Free the context itself
    free(context);
}

// Encode a single sample using HD computing operations
void hd_encode_sample(HDContext* context, unsigned char* features, BundledVector** result) {
    if (!context || !features || !result) {
        printf("Invalid parameters for sample encoding\n");
        *result = NULL;
        return;
    }
    
    // Initialize bound vectors
    BoundVectors* bound = init_bound_vectors(context->dimension, context->feature_dimension);
    if (!bound) {
        printf("Failed to initialize bound vectors for sample encoding\n");
        *result = NULL;
        return;
    }
    
    // Bind the features
    bind_features(features, context->level_vectors, context->mapping, 
                 context->item_memory, bound);
    
    // Bundle the bound vectors
    BundledVector* bundle = init_bundled_vector(context->dimension);
    if (!bundle) {
        printf("Failed to initialize bundle vector for sample encoding\n");
        free_bound_vectors(bound);
        *result = NULL;
        return;
    }
    
    bundle_vectors(bound, bundle);
    free_bound_vectors(bound);
    
    *result = bundle;
}

// Train the HD model using a training dataset
int hd_train(HDContext* context, Dataset* train_data) {
    if (!context || !train_data) {
        printf("Invalid parameters for training\n");
        return 0;
    }

    if (!context->is_initialized) {
        printf("HD context not properly initialized\n");
        return 0;
    }
    
    printf("\nTraining with %d samples...\n", train_data->number_of_samples);
    int progress_step = train_data->number_of_samples / 20;
    
    for (int i = 0; i < train_data->number_of_samples; i++) {
        // Show progress
        if (i % progress_step == 0) {
            printf("Training progress: %.1f%% (%d/%d)\n", 
                   (float)i * 100 / train_data->number_of_samples, 
                   i, train_data->number_of_samples);
        }
        
        // Encode the current sample
        BundledVector* bundle = NULL;
        hd_encode_sample(context, train_data->features[i], &bundle);
        
        if (bundle) {
            // Accumulate the encoded sample into the class vectors
            accumulate_training_vector(context->class_vectors, 
                                      train_data->labels[i], 
                                      bundle);
            free_bundled_vector(bundle);
        }
    }
    
    if (HD_DEBUG_PRINT) {
        print_class_vector_stats(context->class_vectors);
    }
    
    context->is_trained = 1;
    printf("Training completed.\n");
    return 1;
}

// Save the model to a header file
int hd_save_model(HDContext* context, const char* filename) {
    if (!context || !filename) {
        printf("Invalid parameters for model saving\n");
        return 0;
    }
    
    if (!context->is_trained) {
        printf("Model not trained yet\n");
        return 0;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error opening file for writing: %s\n", filename);
        return 0;
    }
    
    // Calculate packed dimension
    int packed_dim = (context->dimension / 8) + (context->dimension % 8 ? 1 : 0);
    
    // Write header file preamble
    fprintf(fp, "#ifndef PACKED_VECTORS_H\n");
    fprintf(fp, "#define PACKED_VECTORS_H\n\n");
    fprintf(fp, "#include <stdint.h>\n\n");
    
    // Write dimension definitions
    fprintf(fp, "#define HD_DIMENSION %d\n", context->dimension);
    fprintf(fp, "#define PACKED_DIMENSION %d\n", packed_dim);
    fprintf(fp, "#define FEATURE_DIMENSION %d\n", context->feature_dimension);
    fprintf(fp, "#define NUM_CLASSES %d\n", context->n_classes);
    fprintf(fp, "#define DATASET_NAME \"%s\"\n\n", context->dataset_name);
    
    // Helper macro to pack a vector
    #define PACK_AND_WRITE(vector, packed, dim) do { \
        memset(packed, 0, packed_dim); \
        for (int bit = 0; bit < dim; bit++) { \
            if (vector[bit]) { \
                packed[bit / 8] |= (1 << (bit % 8)); \
            } \
        } \
    } while(0)
    
    // Write item memory
    fprintf(fp, "const uint8_t packed_item_memory[%d][%d] = {\n", 
            context->feature_dimension, packed_dim);
    
    uint8_t* packed = (uint8_t*)malloc(packed_dim);
    if (!packed) {
        printf("Failed to allocate memory for packing\n");
        fclose(fp);
        return 0;
    }
    
    for (int i = 0; i < context->feature_dimension; i++) {
        PACK_AND_WRITE(context->item_memory[i], packed, context->dimension);
        
        fprintf(fp, "    {");
        for (int j = 0; j < packed_dim; j++) {
            fprintf(fp, "0x%02X%s", packed[j], j < packed_dim - 1 ? "," : "");
        }
        fprintf(fp, "}%s\n", i < context->feature_dimension - 1 ? "," : "");
    }
    fprintf(fp, "};\n\n");
    
    // Write level vectors
    fprintf(fp, "const uint8_t packed_level_vectors[%d][%d] = {\n", 
            context->levels, packed_dim);
    
    for (int i = 0; i < context->levels; i++) {
        PACK_AND_WRITE(context->level_vectors->vectors[i], packed, context->dimension);
        
        fprintf(fp, "    {");
        for (int j = 0; j < packed_dim; j++) {
            fprintf(fp, "0x%02X%s", packed[j], j < packed_dim - 1 ? "," : "");
        }
        fprintf(fp, "}%s\n", i < context->levels - 1 ? "," : "");
    }
    fprintf(fp, "};\n\n");
    
    // Write class HVs
    fprintf(fp, "const uint8_t packed_class_hvs[%d][%d] = {\n", 
            context->n_classes, packed_dim);
    
    for (int i = 0; i < context->n_classes; i++) {
        PACK_AND_WRITE(context->class_vectors->class_hvs[i], packed, context->dimension);
        
        fprintf(fp, "    {");
        for (int j = 0; j < packed_dim; j++) {
            fprintf(fp, "0x%02X%s", packed[j], j < packed_dim - 1 ? "," : "");
        }
        fprintf(fp, "}%s\n", i < context->n_classes - 1 ? "," : "");
    }
    fprintf(fp, "};\n\n");
    
    fprintf(fp, "#endif // PACKED_VECTORS_H\n");
    fclose(fp);
    free(packed);
    
    printf("Generated packed vectors header file: %s\n", filename);
    printf("Packed dimension: %d bytes\n", packed_dim);
    printf("Total memory usage:\n");
    printf("- Item Memory: %d bytes\n", context->feature_dimension * packed_dim);
    printf("- Level Vectors: %d bytes\n", context->levels * packed_dim);
    printf("- Class HVs: %d bytes\n", context->n_classes * packed_dim);
    printf("Total: %d bytes\n", 
           (context->feature_dimension + context->levels + context->n_classes) * packed_dim);
    
    return 1;
}

// Predict the class of a single sample
int hd_predict(HDContext* context, unsigned char* features, int* prediction) {
    if (!context || !features || !prediction) {
        printf("Invalid parameters for prediction\n");
        return 0;
    }
    
    if (!context->is_trained) {
        printf("Model not trained yet\n");
        return 0;
    }
    
    // Initialize prediction to -1 (invalid)
    *prediction = -1;
    
    // Encode the features
    BundledVector* encoded = NULL;
    hd_encode_sample(context, features, &encoded);
    if (!encoded) {
        printf("Failed to encode sample for prediction\n");
        return 0;
    }
    
    // Compute similarity and get prediction
    InferenceResult* result = compute_similarity(encoded, context->class_vectors);
    if (!result) {
        printf("Failed to compute similarity for prediction\n");
        free_bundled_vector(encoded);
        return 0;
    }
    
    *prediction = result->predicted_class;
    free_inference_result(result);
    free_bundled_vector(encoded);
    
    return 1;
}

// Evaluate the model on a test dataset
float hd_evaluate(HDContext* context, Dataset* test_data) {
    if (!context || !test_data) {
        printf("Invalid parameters for evaluation\n");
        return 0.0f;
    }
    
    if (!context->is_trained) {
        printf("Model not trained yet\n");
        return 0.0f;
    }
    
    int correct = 0;
    int total = 0;
    
    printf("\nEvaluating model on %d test samples...\n", test_data->number_of_samples);
    
    // If WRITETESTDATA is defined, write the first 5 test samples to a header file
    #if WRITETESTDATA
    FILE *test_fp = fopen(TEST_DATA_FILE, "w");
    if (test_fp) {
        int num_samples = (test_data->number_of_samples < 5) ? 
                          test_data->number_of_samples : 5;
        
        fprintf(test_fp, "#ifndef TEST_DATA_H\n");
        fprintf(test_fp, "#define TEST_DATA_H\n\n");
        fprintf(test_fp, "#include <stdint.h>\n\n");
        fprintf(test_fp, "#define NUM_TEST_SAMPLES %d\n", num_samples);
        fprintf(test_fp, "#define FEATURE_SIZE %d\n\n", test_data->feature_dimension);
        
        // Write test features
        fprintf(test_fp, "const uint8_t test_features[NUM_TEST_SAMPLES][FEATURE_SIZE] = {\n");
        
        for (int i = 0; i < num_samples; i++) {
            fprintf(test_fp, "    {");
            for (int j = 0; j < test_data->feature_dimension; j++) {
                fprintf(test_fp, "%d%s", 
                        test_data->features[i][j],
                        (j < test_data->feature_dimension - 1) ? ", " : "");
            }
            fprintf(test_fp, "}%s\n", (i < num_samples - 1) ? "," : "");
        }
        
        fprintf(test_fp, "};\n\n");
        
        // Write test labels
        fprintf(test_fp, "const uint8_t test_labels[NUM_TEST_SAMPLES] = {");
        
        for (int i = 0; i < num_samples; i++) {
            fprintf(test_fp, "%d%s", 
                   test_data->labels[i],
                   (i < num_samples - 1) ? ", " : "");
        }
        
        fprintf(test_fp, "};\n\n");
        
        fprintf(test_fp, "#endif // TEST_DATA_H\n");
        
        fclose(test_fp);
        printf("Wrote first %d test samples to %s\n", num_samples, TEST_DATA_FILE);
    } else {
        printf("Failed to open file for writing test data: %s\n", TEST_DATA_FILE);
    }
    #endif
    
    for (int i = 0; i < test_data->number_of_samples; i++) {
        if (i % 100 == 0) {
            printf("Processing test sample %d/%d\n", i, test_data->number_of_samples);
        }
        
        // Encode the features
        BundledVector* encoded = NULL;
        hd_encode_sample(context, test_data->features[i], &encoded);
        if (!encoded) {
            printf("Failed to encode test sample %d\n", i);
            continue;
        }
        
        // Compute similarity and get prediction
        InferenceResult* result = compute_similarity(encoded, context->class_vectors);
        if (!result) {
            printf("Failed to compute similarity for test sample %d\n", i);
            free_bundled_vector(encoded);
            continue;
        }
        
        int true_label = test_data->labels[i];
        int predicted_class = result->predicted_class;
        
        // Update statistics
        total++;
        if (predicted_class == true_label) {
            correct++;
        }
        
        // Display detailed information for the first 5 samples
        if (i < 5) {
            printf("\nTest sample %d:\n", i);
            printf("True label: %d, Predicted: %d\n", true_label, predicted_class);
            printf("Hamming distances (lower is better):\n");
            
            for (int c = 0; c < context->n_classes; c++) {
                printf("Class %d: %d ", c, result->similarities[c]);
                
                // Mark the minimum distance (best match)
                if (c == predicted_class) {
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
        free_bundled_vector(encoded);
    }
    
    float accuracy = (float)correct / total * 100.0f;
    printf("\nOverall Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, total);
    
    return accuracy;
}