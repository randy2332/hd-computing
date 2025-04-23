#ifndef HD_TRAINING_H
#define HD_TRAINING_H

#include "hd_bundling.h"
#include "mnist_loader.h"

// 存儲類別向量的結構
typedef struct {
    int n_classes;        // 類別數量
    int dimension;        // 向量維度
    int *class_counts;    // 每個類別的樣本數量
    int **accumulators;   // 存儲每個類別的累加結果
    char **class_hvs;     // 最終的類別超維向量
} ClassVectors;

// 函數聲明
ClassVectors* init_class_vectors(int n_classes, int dimension);
void free_class_vectors(ClassVectors* cv);
void accumulate_training_vector(ClassVectors* cv, int class_label, BundledVector* bundle);
void quantize_class_vectors(ClassVectors* cv, int bits);
void print_class_vector_stats(ClassVectors* cv);

#endif

